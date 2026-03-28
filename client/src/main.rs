//! telemuze-listen — streaming transcription client with desktop integration.
//!
//! Captures audio from the default input device, runs Silero VAD locally to
//! detect speech segments, then sends each segment to a Telemuze server for
//! transcription.
//!
//! Run with no subcommand to start listening (foreground, for CLI/systemd).
//! Use `toggle` to pause/resume, `flush` to force a segment boundary,
//! and `stop` to shut down.

mod tray;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::cell::Cell;
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::mpsc;
use tray::TrayStatus;
use vad_rs::Vad;

// ── Audio constants ────────────────────────────────────────────────────────

const SAMPLE_RATE: u32 = 16_000;
const FRAME_SIZE: usize = 480; // 30ms at 16kHz

// ── VAD defaults ───────────────────────────────────────────────────────────

const DEFAULT_POSITIVE_THRESHOLD: f32 = 0.3;
const DEFAULT_NEGATIVE_THRESHOLD: f32 = 0.3;
const DEFAULT_MIN_SPEECH_MS: u32 = 250;
const DEFAULT_SILENCE_MS: u32 = 800;
const DEFAULT_PREFILL_MS: u32 = 450;
const DEFAULT_MAX_SPEECH_SECS: u32 = 300;

const DEFAULT_SCROLL_TICKS: u32 = 5;
const DEFAULT_SOCKET: &str = "/tmp/telemuze-listen.sock";

// ── Event types ────────────────────────────────────────────────────────────

enum Event {
    Audio(Vec<f32>),
    Ipc(IpcCommand),
}

enum IpcCommand {
    Toggle,
    Flush,
    Stop,
}

// ── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "telemuze-listen",
    about = "Streaming transcription client for Telemuze.\n\n\
             Run with no subcommand to start the daemon. Use subcommands to control it."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Telemuze server URL
    #[arg(long, env = "TELEMUZE_URL", default_value = "http://127.0.0.1:7313")]
    url: String,

    /// Path to Silero VAD ONNX model file.
    /// Defaults to ~/.local/share/telemuze/models/silero_vad.onnx
    /// and auto-downloads if missing.
    #[arg(long, env = "TELEMUZE_VAD_MODEL_PATH")]
    vad_model_path: Option<String>,

    /// Use the /v1/dictate/smart endpoint (LLM correction, slower).
    #[arg(long)]
    smart: bool,

    /// Type transcribed text into the focused window instead of printing to stdout.
    #[arg(long)]
    type_text: bool,

    /// Show a system tray icon indicating recording/processing state.
    #[arg(long)]
    tray: bool,

    /// Show desktop notifications for recording/processing state.
    #[arg(long)]
    notify: bool,

    /// Play notification sounds on transcription success/error.
    #[arg(long, env = "TELEMUZE_SOUND")]
    sound: bool,

    /// Display server for text injection: "wayland" or "x11" (auto-detected if unset).
    #[arg(long, env = "TELEMUZE_TYPE")]
    display_server: Option<String>,

    /// Unix socket path for IPC between daemon and controller.
    #[arg(long, default_value = DEFAULT_SOCKET)]
    socket: String,

    /// VAD positive threshold (0.0–1.0). Higher = harder to trigger speech start.
    #[arg(long, default_value_t = DEFAULT_POSITIVE_THRESHOLD)]
    vad_positive: f32,

    /// VAD negative threshold (0.0–1.0). Higher = quicker to end speech.
    #[arg(long, default_value_t = DEFAULT_NEGATIVE_THRESHOLD)]
    vad_negative: f32,

    /// Minimum speech duration in ms before a segment is valid.
    #[arg(long, default_value_t = DEFAULT_MIN_SPEECH_MS)]
    min_speech_ms: u32,

    /// Silence duration in ms to end a speech segment.
    #[arg(long, default_value_t = DEFAULT_SILENCE_MS)]
    silence_ms: u32,

    /// Audio to include before speech onset in ms (captures breaths, soft starts).
    #[arg(long, default_value_t = DEFAULT_PREFILL_MS)]
    prefill_ms: u32,

    /// Maximum speech segment length in seconds before force-flushing.
    #[arg(long, default_value_t = DEFAULT_MAX_SPEECH_SECS)]
    max_speech_secs: u32,

    /// Start in paused state (model loaded but not listening). Use `toggle` to begin.
    #[arg(long)]
    paused: bool,

    /// Lowercase the first letter of a segment when the previous segment did not
    /// end with sentence-ending punctuation (. ! ?). Reduces spurious capitals
    /// when VAD splits speech mid-sentence.
    #[arg(long)]
    continuation_lowercase: bool,

    /// Number of scroll ticks per "scroll up" / "scroll down" voice command.
    #[arg(long, default_value_t = DEFAULT_SCROLL_TICKS)]
    scroll_ticks: u32,

    /// Enable verbose logging (parsed actions, key injection details).
    #[arg(long, short, env = "TELEMUZE_VERBOSE")]
    verbose: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Toggle listening on/off (pause/resume). Bind to a hotkey.
    Toggle {
        /// Unix socket path.
        #[arg(long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    /// Flush the current speech chunk (keep listening).
    Flush {
        /// Unix socket path.
        #[arg(long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
    /// Stop the daemon (flush any in-progress speech first).
    Stop {
        /// Unix socket path.
        #[arg(long, default_value = DEFAULT_SOCKET)]
        socket: String,
    },
}

// ── VAD state machine ──────────────────────────────────────────────────────

enum VadEvent {
    None,
    SpeechStarted,
    SpeechEnded(usize),
}

struct VadConfig {
    positive_threshold: f32,
    negative_threshold: f32,
    min_speech_frames: usize,
    redemption_frames: usize,
    prefill_samples: usize,
    max_speech_samples: usize,
}

impl VadConfig {
    fn from_cli(cli: &Cli) -> Self {
        let frame_ms = (FRAME_SIZE as f32 / SAMPLE_RATE as f32) * 1000.0;
        Self {
            positive_threshold: cli.vad_positive,
            negative_threshold: cli.vad_negative,
            min_speech_frames: (cli.min_speech_ms as f32 / frame_ms).ceil() as usize,
            redemption_frames: (cli.silence_ms as f32 / frame_ms).ceil() as usize,
            prefill_samples: (cli.prefill_ms as usize * SAMPLE_RATE as usize) / 1000,
            max_speech_samples: cli.max_speech_secs as usize * SAMPLE_RATE as usize,
        }
    }
}

/// Number of recent VAD probability frames to keep for smart splitting.
/// 100 frames × 30ms = 3 seconds of lookback.
const LOOKBACK_FRAMES: usize = 100;

struct VadState {
    in_speech: bool,
    speech_start: Option<usize>,
    speech_frames: usize,
    redemption_count: usize,
    prob_history: VecDeque<f32>,
}

impl VadState {
    fn new() -> Self {
        Self {
            in_speech: false,
            speech_start: None,
            speech_frames: 0,
            redemption_count: 0,
            prob_history: VecDeque::with_capacity(LOOKBACK_FRAMES),
        }
    }

    fn update(&mut self, prob: f32, current_pos: usize, cfg: &VadConfig) -> VadEvent {
        if !self.in_speech {
            if prob > cfg.positive_threshold {
                self.in_speech = true;
                self.speech_start = Some(current_pos.saturating_sub(cfg.prefill_samples));
                self.speech_frames = 1;
                self.redemption_count = 0;
                return VadEvent::SpeechStarted;
            }
            VadEvent::None
        } else {
            self.speech_frames += 1;

            // Track probability for smart force-splitting.
            self.prob_history.push_back(prob);
            if self.prob_history.len() > LOOKBACK_FRAMES {
                self.prob_history.pop_front();
            }

            if prob < cfg.negative_threshold {
                self.redemption_count += 1;
                if self.redemption_count > cfg.redemption_frames {
                    return match self.end_speech(cfg) {
                        Some(start) => VadEvent::SpeechEnded(start),
                        None => VadEvent::None,
                    };
                }
            } else {
                self.redemption_count = 0;
            }
            VadEvent::None
        }
    }

    /// Find the sample offset (back from current position) of the
    /// lowest-probability frame in the lookback window.
    fn best_split_offset(&self) -> usize {
        let len = self.prob_history.len();
        if len == 0 {
            return 0;
        }
        let (min_idx, _) = self
            .prob_history
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        (len - 1 - min_idx) * FRAME_SIZE
    }

    fn end_speech(&mut self, cfg: &VadConfig) -> Option<usize> {
        let result = if self.speech_frames >= cfg.min_speech_frames {
            self.speech_start
        } else {
            None
        };
        self.in_speech = false;
        self.speech_start = None;
        self.speech_frames = 0;
        self.redemption_count = 0;
        self.prob_history.clear();
        result
    }
}

// ── Application context ────────────────────────────────────────────────────

struct AppContext {
    http_client: reqwest::blocking::Client,
    endpoint_url: String,
    smart: bool,
    type_text: bool,
    notify: bool,
    sound: bool,
    verbose: bool,
    continuation_lowercase: bool,
    scroll_ticks: u32,
    last_ended_with_punctuation: Cell<bool>,
    display_server: String,
    notify_id: Cell<u32>,
    tray_handle: Option<tray::TrayHandle>,
}

impl AppContext {
    fn new(cli: &Cli) -> Self {
        let base_url = cli.url.trim_end_matches('/');
        let endpoint_url = if cli.smart {
            format!("{base_url}/v1/dictate/smart")
        } else {
            format!("{base_url}/v1/audio/transcriptions")
        };

        let display_server = cli
            .display_server
            .clone()
            .unwrap_or_else(detect_display_server);

        Self {
            http_client: reqwest::blocking::Client::new(),
            endpoint_url,
            smart: cli.smart,
            type_text: cli.type_text,
            notify: cli.notify,
            sound: cli.sound,
            verbose: cli.verbose,
            continuation_lowercase: cli.continuation_lowercase,
            scroll_ticks: cli.scroll_ticks,
            last_ended_with_punctuation: Cell::new(true),
            display_server,
            notify_id: Cell::new(0),
            tray_handle: None,
        }
    }

    fn set_tray_status(&self, status: TrayStatus) {
        if let Some(handle) = &self.tray_handle {
            handle.update(status);
        }
    }
}

// ── Desktop integration (shell out) ────────────────────────────────────────

fn detect_display_server() -> String {
    if std::env::var("WAYLAND_DISPLAY").is_ok() {
        "wayland".into()
    } else {
        "x11".into()
    }
}

fn type_into_window(text: &str, display_server: &str, verbose: bool) {
    if verbose {
        eprintln!("  -> type_into_window({text:?}, {display_server})");
    }
    let result = if display_server == "wayland" {
        std::process::Command::new("wtype")
            .arg(format!("{text} "))
            .status()
    } else {
        std::process::Command::new("xdotool")
            .args(["type", "--clearmodifiers", &format!("{text} ")])
            .status()
    };
    if let Err(e) = result {
        eprintln!("Text injection failed: {e}");
    }
}

fn send_key(key: &str, display_server: &str, verbose: bool) {
    if verbose {
        eprintln!("  -> send_key({key:?}, {display_server})");
    }
    let result = if display_server == "wayland" {
        std::process::Command::new("wtype")
            .args(["-k", key])
            .status()
    } else {
        std::process::Command::new("xdotool")
            .args(["key", "--clearmodifiers", key])
            .status()
    };
    if let Err(e) = result {
        eprintln!("Key injection failed: {e}");
    }
}

fn send_modified_key(modifiers: &[&str], key: &str, display_server: &str, verbose: bool) {
    if verbose {
        eprintln!("  -> send_modified_key({modifiers:?}+{key:?}, {display_server})");
    }
    let result = if display_server == "wayland" {
        // wtype: -M mod -k key -m mod (press modifier, tap key, release modifier)
        let mut args: Vec<String> = Vec::new();
        for m in modifiers {
            args.push("-M".into());
            args.push(m.to_string());
        }
        args.push("-k".into());
        args.push(key.into());
        for m in modifiers.iter().rev() {
            args.push("-m".into());
            args.push(m.to_string());
        }
        std::process::Command::new("wtype")
            .args(&args)
            .status()
    } else {
        // xdotool: "ctrl+Return" style
        let combo = format!("{}+{}", modifiers.join("+"), key);
        std::process::Command::new("xdotool")
            .args(["key", "--clearmodifiers", &combo])
            .status()
    };
    if let Err(e) = result {
        eprintln!("Key injection failed: {e}");
    }
}

fn click_quadrant(right: bool, bottom: bool) {
    // Get screen dimensions via xdotool
    let output = match std::process::Command::new("xdotool")
        .args(["getdisplaygeometry"])
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Failed to get display geometry: {e}");
            return;
        }
    };
    let geometry = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = geometry.trim().split_whitespace().collect();
    if parts.len() < 2 {
        eprintln!("Unexpected geometry output: {geometry}");
        return;
    }
    let (width, height) = match (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
        (Ok(w), Ok(h)) => (w, h),
        _ => {
            eprintln!("Failed to parse geometry: {geometry}");
            return;
        }
    };

    let x = if right { width * 3 / 4 } else { width / 4 };
    let y = if bottom { height * 3 / 4 } else { height / 4 };

    let result = std::process::Command::new("xdotool")
        .args(["mousemove", "--sync", &x.to_string(), &y.to_string(), "click", "1"])
        .status();
    if let Err(e) = result {
        eprintln!("Mouse click failed: {e}");
    }
}

fn scroll(up: bool, ticks: u32) {
    let button = if up { "4" } else { "5" };
    let repeat = ticks.to_string();
    let result = std::process::Command::new("xdotool")
        .args(["click", "--repeat", &repeat, "--delay", "10", button])
        .status();
    if let Err(e) = result {
        eprintln!("Scroll failed: {e}");
    }
}

// ── Voice commands ──────────────────────────────────────────────────────

/// An action that a voice command can trigger.
#[derive(Clone, Debug)]
enum VoiceAction {
    /// Simulate a keypress (e.g. "Return", "Tab").
    Key(&'static str),
    /// Simulate a modified keypress (e.g. Ctrl+Return).
    ModifiedKey {
        modifiers: &'static [&'static str],
        key: &'static str,
    },
    /// Move the mouse to a screen quadrant and click.
    ClickQuadrant {
        /// Horizontal position: false = left, true = right.
        right: bool,
        /// Vertical position: false = top, true = bottom.
        bottom: bool,
    },
    /// Scroll the mouse wheel.
    Scroll {
        /// true = scroll up, false = scroll down.
        up: bool,
    },
}

/// A voice command: a phrase (lowercased words) mapped to an action.
struct VoiceCommand {
    /// The trigger words, e.g. &["press", "enter"].
    words: &'static [&'static str],
    action: VoiceAction,
}

fn voice_commands() -> &'static [VoiceCommand] {
    static COMMANDS: &[VoiceCommand] = &[
        VoiceCommand {
            words: &["press", "control", "enter"],
            action: VoiceAction::ModifiedKey {
                modifiers: &["ctrl"],
                key: "Return",
            },
        },
        VoiceCommand {
            words: &["press", "ctrl", "enter"],
            action: VoiceAction::ModifiedKey {
                modifiers: &["ctrl"],
                key: "Return",
            },
        },
        VoiceCommand {
            words: &["press", "shift", "tab"],
            action: VoiceAction::ModifiedKey {
                modifiers: &["shift"],
                key: "Tab",
            },
        },
        VoiceCommand {
            words: &["press", "enter"],
            action: VoiceAction::Key("Return"),
        },
        VoiceCommand {
            words: &["press", "tab"],
            action: VoiceAction::Key("Tab"),
        },
        VoiceCommand {
            words: &["press", "up"],
            action: VoiceAction::Key("Up"),
        },
        VoiceCommand {
            words: &["press", "down"],
            action: VoiceAction::Key("Down"),
        },
        VoiceCommand {
            words: &["press", "left"],
            action: VoiceAction::Key("Left"),
        },
        VoiceCommand {
            words: &["press", "right"],
            action: VoiceAction::Key("Right"),
        },
        VoiceCommand {
            words: &["click", "upper", "left"],
            action: VoiceAction::ClickQuadrant { right: false, bottom: false },
        },
        VoiceCommand {
            words: &["click", "upper", "right"],
            action: VoiceAction::ClickQuadrant { right: true, bottom: false },
        },
        VoiceCommand {
            words: &["click", "lower", "left"],
            action: VoiceAction::ClickQuadrant { right: false, bottom: true },
        },
        VoiceCommand {
            words: &["click", "lower", "right"],
            action: VoiceAction::ClickQuadrant { right: true, bottom: true },
        },
        VoiceCommand {
            words: &["scroll", "up"],
            action: VoiceAction::Scroll { up: true },
        },
        VoiceCommand {
            words: &["scroll", "down"],
            action: VoiceAction::Scroll { up: false },
        },
    ];
    COMMANDS
}

/// A segment of processed transcription output.
#[derive(Debug)]
enum TextAction<'a> {
    /// Literal text to type or print.
    Text(&'a str),
    /// Transformed text to type or print (owned, from input transformations).
    OwnedText(String),
    /// A voice command was recognized.
    Command(VoiceAction),
}

/// Returns true if `s` contains at least one alphanumeric character (i.e. is
/// not purely punctuation/whitespace left over after stripping a command).
fn has_word_chars(s: &str) -> bool {
    s.chars().any(|c| c.is_alphanumeric())
}

/// Try to find "slash command <word>" at position `from` in `lower`.
/// Returns `(start, end, slash_word)` where `slash_word` is the captured word
/// lowercased with a leading `/`.
fn find_slash_command(lower: &str, from: usize) -> Option<(usize, usize, String)> {
    let trigger = &["slash", "command"];
    if let Some((start, end)) = find_phrase(lower, from, trigger) {
        // Skip whitespace/punctuation after "slash command"
        let rest = &lower[end..];
        let skip = rest
            .chars()
            .take_while(|c| {
                c.is_whitespace()
                    || matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '-' | '\'' | '"')
            })
            .map(|c| c.len_utf8())
            .sum::<usize>();
        let after = &rest[skip..];
        // Capture the next word (alphanumeric chars)
        let word_len = after
            .chars()
            .take_while(|c| c.is_alphanumeric())
            .map(|c| c.len_utf8())
            .sum::<usize>();
        if word_len > 0 {
            let word = &after[..word_len];
            return Some((start, end + skip + word_len, format!("/{word}")));
        }
    }
    None
}

/// Scan `text` for voice command phrases (case-insensitive, tolerant of
/// punctuation between words) and split into text segments and command actions.
fn process_voice_commands(text: &str) -> Vec<TextAction<'_>> {
    let lower = text.to_lowercase();
    let commands = voice_commands();
    let mut actions: Vec<TextAction<'_>> = Vec::new();
    let mut cursor = 0;

    while cursor < text.len() {
        // Try each command at every position
        let mut best: Option<(usize, usize, VoiceAction)> = None; // (start, end, action)

        for cmd in commands {
            if let Some((start, end)) = find_phrase(&lower, cursor, cmd.words) {
                if best.as_ref().is_none_or(|b| start < b.0) {
                    best = Some((start, end, cmd.action.clone()));
                }
            }
        }

        // Try "slash command <word>" only at the very beginning of the text
        let slash = if cursor == 0 {
            find_slash_command(&lower, 0)
                .filter(|(start, ..)| !has_word_chars(&text[..*start]))
        } else {
            None
        };

        // Determine whether a voice command or slash command comes first
        let use_voice = match (&best, &slash) {
            (Some((vs, ..)), Some((ss, ..))) => vs <= ss,
            (Some(_), None) => true,
            _ => false,
        };

        if use_voice {
            let (start, end, action) = best.unwrap();
            let before = text[cursor..start].trim();
            if has_word_chars(before) {
                actions.push(TextAction::Text(before));
            }
            actions.push(TextAction::Command(action));
            cursor = end;
        } else if let Some((start, end, slash_text)) = slash {
            let before = text[cursor..start].trim();
            if has_word_chars(before) {
                actions.push(TextAction::Text(before));
            }
            actions.push(TextAction::OwnedText(slash_text));
            cursor = end;
        } else {
            let rest = text[cursor..].trim();
            if has_word_chars(rest) {
                actions.push(TextAction::Text(rest));
            }
            break;
        }
    }

    actions
}

/// Find a multi-word phrase in `lower` starting from `from`, allowing optional
/// punctuation and whitespace between words. Returns (start, end) byte offsets
/// into `lower`.
fn find_phrase(lower: &str, from: usize, words: &[&str]) -> Option<(usize, usize)> {
    if words.is_empty() {
        return None;
    }

    let haystack = &lower[from..];
    let first = words[0];
    let mut search_start = 0;

    while let Some(p) = haystack[search_start..].find(first) {
        let abs_start = from + search_start + p;
        let mut pos = search_start + p + first.len();

        let mut matched = true;
        for &word in &words[1..] {
            // Skip punctuation and whitespace between words
            let rest = &haystack[pos..];
            let skip = rest
                .chars()
                .take_while(|c| {
                    c.is_whitespace()
                        || matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '-' | '\'' | '"')
                })
                .map(|c| c.len_utf8())
                .sum::<usize>();

            if skip == 0 || !haystack[pos + skip..].starts_with(word) {
                matched = false;
                break;
            }
            pos = pos + skip + word.len();
        }

        if matched {
            return Some((abs_start, from + pos));
        }

        search_start = search_start + p + first.len();
    }

    None
}

fn show_notification(summary: &str, body: &str, replaces_id: u32) -> u32 {
    let output = std::process::Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.freedesktop.Notifications",
            "--object-path",
            "/org/freedesktop/Notifications",
            "--method",
            "org.freedesktop.Notifications.Notify",
            "Telemuze",
            &replaces_id.to_string(),
            "audio-input-microphone",
            summary,
            body,
            "[]",
            "{\"urgency\": <byte 1>}",
            "0",
        ])
        .output();

    match output {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout);
            s.split_whitespace()
                .find_map(|tok| {
                    tok.trim_end_matches([',', ')'])
                        .parse::<u32>()
                        .ok()
                        .filter(|&n| n > 0)
                })
                .unwrap_or(0)
        }
        Err(_) => 0,
    }
}

fn dismiss_notification(notify_id: u32) {
    if notify_id == 0 {
        return;
    }
    let _ = std::process::Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--dest",
            "org.freedesktop.Notifications",
            "--object-path",
            "/org/freedesktop/Notifications",
            "--method",
            "org.freedesktop.Notifications.CloseNotification",
            &notify_id.to_string(),
        ])
        .output();
}

fn play_sound(name: &str) {
    let path = format!("/usr/share/sounds/freedesktop/stereo/{name}.oga");
    let _ = std::process::Command::new("paplay")
        .arg(&path)
        .spawn();
}

// ── WAV encoding ───────────────────────────────────────────────────────────

fn encode_wav(samples: &[f32]) -> Result<Vec<u8>> {
    let mut buf = io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut buf, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        writer.write_sample((clamped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(buf.into_inner())
}

// ── Server communication ───────────────────────────────────────────────────

fn send_to_server(
    client: &reqwest::blocking::Client,
    url: &str,
    wav_data: Vec<u8>,
    smart: bool,
) -> Result<String> {
    let part = reqwest::blocking::multipart::Part::bytes(wav_data)
        .file_name("audio.wav")
        .mime_str("audio/wav")?;
    let form = reqwest::blocking::multipart::Form::new().part("file", part);

    let response = client
        .post(url)
        .multipart(form)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .context("Failed to send audio to server")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        anyhow::bail!("Server returned {status}: {body}");
    }

    if smart {
        Ok(response.text()?)
    } else {
        let json: serde_json::Value = response.json()?;
        Ok(json["text"].as_str().unwrap_or("").to_string())
    }
}

// ── VAD model download ─────────────────────────────────────────────────────

fn download_vad_model(dest: &std::path::Path) -> Result<()> {
    const URL: &str =
        "https://github.com/thewh1teagle/vad-rs/releases/download/v0.1.0/silero_vad.onnx";

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    eprintln!("Downloading Silero VAD model...");
    let client = reqwest::blocking::Client::new();
    let response = client.get(URL).send().context("Failed to download VAD model")?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with status {}", response.status());
    }

    let bytes = response.bytes()?;
    let mut file = std::fs::File::create(dest)?;
    file.write_all(&bytes)?;

    eprintln!("VAD model saved to {}", dest.display());
    Ok(())
}

// ── IPC ────────────────────────────────────────────────────────────────────

fn send_ipc_command(socket_path: &str, command: &str) -> Result<()> {
    let mut stream = UnixStream::connect(socket_path)
        .context("No listener running (cannot connect to socket)")?;
    stream.write_all(command.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    Ok(())
}

/// Spawn a thread that accepts connections on the Unix socket and forwards
/// parsed commands through the event channel.
fn spawn_ipc_listener(listener: UnixListener, tx: mpsc::SyncSender<Event>) {
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let mut reader = io::BufReader::new(stream);
            let mut line = String::new();
            if reader.read_line(&mut line).is_ok() {
                let cmd = match line.trim() {
                    "toggle" => Some(IpcCommand::Toggle),
                    "flush" => Some(IpcCommand::Flush),
                    "stop" => Some(IpcCommand::Stop),
                    _ => None,
                };
                if let Some(cmd) = cmd {
                    if tx.send(Event::Ipc(cmd)).is_err() {
                        break; // Main thread gone
                    }
                }
            }
        }
    });
}

/// Spawn a thread that translates SIGINT/SIGTERM into Stop events.
fn spawn_signal_handler(tx: mpsc::SyncSender<Event>) {
    use signal_hook::iterator::Signals;
    std::thread::spawn(move || {
        let mut signals =
            Signals::new([signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM])
                .expect("Failed to register signal handlers");
        for _ in signals.forever() {
            let _ = tx.send(Event::Ipc(IpcCommand::Stop));
            break;
        }
    });
}

// ── Continuation lowercasing ──────────────────────────────────────────────

fn lowercase_first_letter(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

fn ends_with_sentence_punctuation(s: &str) -> bool {
    s.trim_end().ends_with(['.', '!', '?'])
}

// ── Segment flushing ───────────────────────────────────────────────────────

fn flush_segment(samples: &[f32], ctx: &AppContext) {
    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    eprint!("[{duration_secs:.1}s] ");

    ctx.set_tray_status(TrayStatus::Processing);
    if ctx.notify {
        let id = show_notification("Processing...", "Transcribing audio", ctx.notify_id.get());
        ctx.notify_id.set(id);
    }

    match encode_wav(samples) {
        Ok(wav) => {
            match send_to_server(&ctx.http_client, &ctx.endpoint_url, wav, ctx.smart) {
                Ok(text) => {
                    let text = text.trim();
                    if !text.is_empty() {
                        eprintln!("OK");
                        let text = if ctx.continuation_lowercase
                            && !ctx.last_ended_with_punctuation.get()
                        {
                            lowercase_first_letter(text)
                        } else {
                            text.to_string()
                        };
                        if ctx.continuation_lowercase {
                            ctx.last_ended_with_punctuation
                                .set(ends_with_sentence_punctuation(&text));
                        }
                        let actions = process_voice_commands(&text);
                        if ctx.verbose {
                            eprintln!("  actions: {actions:?}");
                        }
                        let mut just_typed = false;
                        for action in &actions {
                            match action {
                                TextAction::Text(t) => {
                                    if ctx.type_text {
                                        type_into_window(t, &ctx.display_server, ctx.verbose);
                                        just_typed = true;
                                    } else {
                                        print!("{t} ");
                                    }
                                }
                                TextAction::OwnedText(t) => {
                                    if ctx.type_text {
                                        type_into_window(t, &ctx.display_server, ctx.verbose);
                                        just_typed = true;
                                    } else {
                                        print!("{t} ");
                                    }
                                }
                                TextAction::Command(VoiceAction::Key(key)) => {
                                    if ctx.type_text {
                                        if just_typed {
                                            std::thread::sleep(std::time::Duration::from_millis(50));
                                        }
                                        send_key(key, &ctx.display_server, ctx.verbose);
                                    } else {
                                        println!();
                                    }
                                    just_typed = false;
                                }
                                TextAction::Command(VoiceAction::ModifiedKey { modifiers, key }) => {
                                    if ctx.type_text {
                                        if just_typed {
                                            std::thread::sleep(std::time::Duration::from_millis(50));
                                        }
                                        send_modified_key(modifiers, key, &ctx.display_server, ctx.verbose);
                                    } else {
                                        println!();
                                    }
                                    just_typed = false;
                                }
                                TextAction::Command(VoiceAction::ClickQuadrant { right, bottom }) => {
                                    if ctx.type_text {
                                        click_quadrant(*right, *bottom);
                                    } else {
                                        let h = if *bottom { "lower" } else { "upper" };
                                        let v = if *right { "right" } else { "left" };
                                        println!("[click {h} {v}]");
                                    }
                                    just_typed = false;
                                }
                                TextAction::Command(VoiceAction::Scroll { up }) => {
                                    if ctx.type_text {
                                        scroll(*up, ctx.scroll_ticks);
                                    } else {
                                        let dir = if *up { "up" } else { "down" };
                                        println!("[scroll {dir}]");
                                    }
                                    just_typed = false;
                                }
                            }
                        }
                        if !ctx.type_text {
                            let _ = io::stdout().flush();
                        }
                        if ctx.sound {
                            play_sound("message-new-instant");
                        }
                    } else {
                        eprintln!("(empty)");
                    }
                    ctx.set_tray_status(TrayStatus::Listening);
                    if ctx.notify {
                        let id = show_notification(
                            "Listening...",
                            "Telemuze is active",
                            ctx.notify_id.get(),
                        );
                        ctx.notify_id.set(id);
                    }
                }
                Err(e) => {
                    eprintln!("send error: {e}");
                    ctx.set_tray_status(TrayStatus::Listening);
                    if ctx.notify {
                        let id = show_notification(
                            "Dictation failed",
                            &e.to_string(),
                            ctx.notify_id.get(),
                        );
                        ctx.notify_id.set(id);
                    }
                    if ctx.sound {
                        play_sound("dialog-error");
                    }
                }
            }
        }
        Err(e) => eprintln!("encode error: {e}"),
    }
}

// ── Socket cleanup guard ───────────────────────────────────────────────────

struct SocketGuard<'a> {
    path: &'a str,
    notify_id: &'a Cell<u32>,
    notify: bool,
    tray_handle: Option<tray::TrayHandle>,
}

impl Drop for SocketGuard<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.path);
        if self.notify {
            dismiss_notification(self.notify_id.get());
        }
        if let Some(handle) = self.tray_handle.take() {
            handle.shutdown();
        }
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle subcommands that send IPC to a running daemon
    match &cli.command {
        Some(Command::Toggle { socket }) => return send_ipc_command(socket, "toggle"),
        Some(Command::Flush { socket }) => return send_ipc_command(socket, "flush"),
        Some(Command::Stop { socket }) => return send_ipc_command(socket, "stop"),
        None => {} // Start the daemon
    }

    // ── Start the daemon ───────────────────────────────────────────────

    // Clean up stale socket
    let _ = std::fs::remove_file(&cli.socket);

    let listener = UnixListener::bind(&cli.socket)
        .with_context(|| format!("Failed to bind socket: {}", cli.socket))?;

    let mut ctx = AppContext::new(&cli);
    let vad_cfg = VadConfig::from_cli(&cli);

    // Unified event channel for audio and IPC
    let (tx, rx) = mpsc::sync_channel::<Event>(64);

    // Spawn system tray if requested
    if cli.tray {
        let initial = if cli.paused {
            TrayStatus::Idle
        } else {
            TrayStatus::Listening
        };
        match tray::spawn_tray(tx.clone(), initial) {
            Ok(handle) => ctx.tray_handle = Some(handle),
            Err(e) => eprintln!("Tray unavailable: {e}"),
        }
    }

    // Socket cleanup on exit
    let _guard = SocketGuard {
        path: &cli.socket,
        notify_id: &ctx.notify_id,
        notify: ctx.notify,
        tray_handle: ctx.tray_handle.clone(),
    };

    // Spawn IPC listener thread (blocks on accept, sends events)
    spawn_ipc_listener(listener, tx.clone());

    // Spawn signal handler thread (SIGINT/SIGTERM → Stop event)
    spawn_signal_handler(tx.clone());

    // Resolve VAD model path
    let vad_path = match &cli.vad_model_path {
        Some(p) => std::path::PathBuf::from(p),
        None => {
            let default = dirs_next::data_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("telemuze/models/silero_vad.onnx");
            if !default.exists() {
                download_vad_model(&default)?;
            }
            default
        }
    };

    eprintln!("Loading VAD model from {}", vad_path.display());
    let mut vad = Vad::new(&vad_path, SAMPLE_RATE as usize)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("Failed to load VAD model")?;

    // Set up audio capture
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input audio device available")?;

    eprintln!(
        "Using input device: {}",
        device.name().unwrap_or_else(|_| "unknown".into())
    );

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let audio_tx = tx; // Move remaining sender to audio callback
    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let _ = audio_tx.send(Event::Audio(data.to_vec()));
            },
            |err| eprintln!("Audio stream error: {err}"),
            None,
        )
        .context("Failed to build audio input stream")?;

    let mut listening = !cli.paused;
    if listening {
        stream.play().context("Failed to start audio stream")?;
    }

    if ctx.notify && listening {
        let id = show_notification("Listening...", "Telemuze is active", 0);
        ctx.notify_id.set(id);
    }

    if listening {
        eprintln!("Listening... (use `telemuze-listen stop` to shut down)");
    } else {
        eprintln!("Ready (paused). Use `telemuze-listen toggle` to start listening.");
    }
    eprintln!("Endpoint: {}", ctx.endpoint_url);
    eprintln!();

    // Main processing loop
    let mut audio_buf: Vec<f32> = Vec::new();
    let mut vad_pos: usize = 0;
    let mut vad_state = VadState::new();

    loop {
        match rx.recv() {
            Ok(Event::Audio(chunk)) => {
                if !listening {
                    continue; // Discard audio while paused
                }
                audio_buf.extend_from_slice(&chunk);

                // Process all complete 30ms frames
                while vad_pos + FRAME_SIZE <= audio_buf.len() {
                    let frame = &audio_buf[vad_pos..vad_pos + FRAME_SIZE];

                    let prob = match vad.compute(frame) {
                        Ok(result) => result.prob,
                        Err(e) => {
                            eprintln!("VAD error at sample {vad_pos}: {e}");
                            if let Some(start) = vad_state.end_speech(&vad_cfg) {
                                flush_segment(&audio_buf[start..vad_pos], &ctx);
                            }
                            vad.reset();
                            audio_buf.drain(..vad_pos + FRAME_SIZE);
                            vad_pos = 0;
                            continue;
                        }
                    };

                    match vad_state.update(prob, vad_pos, &vad_cfg) {
                        VadEvent::SpeechStarted => {
                            ctx.set_tray_status(TrayStatus::Recording);
                            if ctx.notify {
                                let id = show_notification(
                                    "Recording...",
                                    "Speech detected",
                                    ctx.notify_id.get(),
                                );
                                ctx.notify_id.set(id);
                            }
                        }
                        VadEvent::SpeechEnded(start) => {
                            let end = vad_pos + FRAME_SIZE;
                            flush_segment(&audio_buf[start..end], &ctx);
                            audio_buf.drain(..end);
                            vad_pos = 0;
                            continue;
                        }
                        VadEvent::None => {}
                    }

                    // Force-flush if speech is too long, splitting at the
                    // lowest-probability frame in the lookback window.
                    if vad_state.in_speech {
                        if let Some(start) = vad_state.speech_start {
                            if vad_pos + FRAME_SIZE - start >= vad_cfg.max_speech_samples {
                                let offset_back = vad_state.best_split_offset();
                                let split_end = (vad_pos + FRAME_SIZE) - offset_back;
                                flush_segment(&audio_buf[start..split_end], &ctx);
                                let frames_back = offset_back / FRAME_SIZE;
                                audio_buf.drain(..split_end);
                                vad_pos = audio_buf.len() - offset_back;
                                vad_state.speech_start = Some(0);
                                vad_state.speech_frames = frames_back;
                                vad_state.prob_history.clear();
                                vad.reset();
                                continue;
                            }
                        }
                    }

                    vad_pos += FRAME_SIZE;
                }

                // Prevent unbounded memory growth during silence and reset
                // VAD internal state to avoid accumulated ORT tensor drift.
                if !vad_state.in_speech && vad_pos > SAMPLE_RATE as usize * 5 {
                    let keep = vad_cfg.prefill_samples.min(vad_pos);
                    audio_buf.drain(..vad_pos - keep);
                    vad_pos = keep;
                    vad.reset();
                }
            }

            Ok(Event::Ipc(IpcCommand::Toggle)) => {
                if listening {
                    // Pause: flush in-progress speech, stop audio capture
                    if let Some(start) = vad_state.end_speech(&vad_cfg) {
                        let end = vad_pos.min(audio_buf.len());
                        if start < end {
                            flush_segment(&audio_buf[start..end], &ctx);
                        }
                    }
                    stream.pause().ok();
                    audio_buf.clear();
                    vad_pos = 0;
                    vad.reset();
                    listening = false;
                    eprintln!("Paused");
                    ctx.set_tray_status(TrayStatus::Idle);
                    if ctx.notify {
                        dismiss_notification(ctx.notify_id.get());
                        ctx.notify_id.set(0);
                    }
                } else {
                    // Resume
                    stream.play().ok();
                    listening = true;
                    eprintln!("Listening...");
                    ctx.set_tray_status(TrayStatus::Listening);
                    if ctx.notify {
                        let id = show_notification(
                            "Listening...",
                            "Telemuze is active",
                            ctx.notify_id.get(),
                        );
                        ctx.notify_id.set(id);
                    }
                }
            }

            Ok(Event::Ipc(IpcCommand::Flush)) => {
                if vad_state.in_speech {
                    if let Some(start) = vad_state.end_speech(&vad_cfg) {
                        let end = vad_pos.min(audio_buf.len());
                        if start < end {
                            flush_segment(&audio_buf[start..end], &ctx);
                        }
                        audio_buf.drain(..end);
                        vad_pos = 0;
                        vad.reset();
                    }
                }
            }

            Ok(Event::Ipc(IpcCommand::Stop)) => {
                // Flush any in-progress speech
                if let Some(start) = vad_state.end_speech(&vad_cfg) {
                    let end = vad_pos.min(audio_buf.len());
                    if start < end {
                        flush_segment(&audio_buf[start..end], &ctx);
                    }
                }
                eprintln!("Stopping...");
                break;
            }

            Err(_) => break, // All senders dropped
        }
    }

    Ok(())
}
