//! telemuze-listen — streaming transcription client with desktop integration.
//!
//! Captures audio from the default input device, runs Silero VAD locally to
//! detect speech segments, then sends each segment to a Telemuze server for
//! transcription.
//!
//! Run with no subcommand to start listening (foreground, for CLI/systemd).
//! Use `toggle` to pause/resume, `flush` to force a segment boundary,
//! and `stop` to shut down.

mod config;
mod tray;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use config::{ResolvedAliases, ResolvedConfig};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use std::cell::Cell;
use std::io::{self, BufRead, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use tray::TrayStatus;

// ── Audio constants ────────────────────────────────────────────────────────

const SAMPLE_RATE: u32 = 16_000;
const WINDOW_SIZE: usize = 512; // sherpa-onnx Silero VAD frame size
const VAD_RESET_SECS: u64 = 45; // Must be < buffer_size_in_seconds (60)

// ── VAD defaults ───────────────────────────────────────────────────────────

const DEFAULT_VAD_THRESHOLD: f32 = 0.5;
const DEFAULT_IDLE_SILENCE: f32 = 0.4;
const DEFAULT_DICTATION_SILENCE: f32 = 1.5;
const DEFAULT_MIN_SPEECH: f32 = 0.1;
const DEFAULT_MAX_SPEECH: f32 = 15.0;
const DEFAULT_DICTATION_MAX_SPEECH: f32 = 30.0;
const DEFAULT_PREFILL_MS: u32 = 450;
const DEFAULT_LOWERCASE_TIMEOUT: f32 = 5.0;

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

    /// Path to TOML configuration file.
    #[arg(long, env = "TELEMUZE_CONFIG")]
    config: Option<PathBuf>,

    /// Print the full resolved configuration as TOML (with comments) and exit.
    /// Reads the current config file, fills in any missing options with defaults.
    #[arg(long)]
    dump_config: bool,

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

    /// VAD speech detection threshold (0.0–1.0). Higher = harder to trigger.
    #[arg(long, default_value_t = DEFAULT_VAD_THRESHOLD)]
    vad_threshold: f32,

    /// Silence duration (seconds) to end a segment in idle/command mode.
    #[arg(long, default_value_t = DEFAULT_IDLE_SILENCE)]
    idle_silence: f32,

    /// Silence duration (seconds) to end a segment in dictation mode.
    #[arg(long, default_value_t = DEFAULT_DICTATION_SILENCE)]
    dictation_silence: f32,

    /// Minimum speech duration (seconds) before a segment is valid.
    #[arg(long, default_value_t = DEFAULT_MIN_SPEECH)]
    min_speech: f32,

    /// Maximum speech segment length (seconds) in idle mode before force-flushing.
    #[arg(long, default_value_t = DEFAULT_MAX_SPEECH)]
    max_speech: f32,

    /// Audio to include before speech onset in ms (captures breaths, soft starts).
    #[arg(long, default_value_t = DEFAULT_PREFILL_MS)]
    prefill_ms: u32,

    /// Maximum dictation utterance length (seconds) before force-flushing.
    #[arg(long, default_value_t = DEFAULT_DICTATION_MAX_SPEECH)]
    dictation_max_speech: f32,

    /// Comma-separated list of words to boost during dictation (hotwords).
    #[arg(long, env = "TELEMUZE_HOTWORDS")]
    hotwords: Option<String>,

    /// Score boost for command hotwords in idle mode (0.0 = disabled). Range: 1.0–4.0.
    #[arg(long, env = "TELEMUZE_COMMAND_HOTWORDS_SCORE", default_value_t = 1.5)]
    command_hotwords_score: f32,

    /// Score boost for user-provided dictation hotwords (0.0 = disabled). Range: 1.0–4.0.
    #[arg(long, env = "TELEMUZE_DICTATION_HOTWORDS_SCORE", default_value_t = 1.5)]
    dictation_hotwords_score: f32,

    /// Seconds after last output before continuation lowercasing resets
    /// (treats next segment as a fresh utterance).
    #[arg(long, default_value_t = DEFAULT_LOWERCASE_TIMEOUT)]
    lowercase_timeout: f32,

    /// Start in paused state (model loaded but not listening). Use `toggle` to begin.
    #[arg(long)]
    paused: bool,

    /// Lowercase the first letter of a segment in idle mode when the previous
    /// segment did not end with sentence-ending punctuation (. ! ?).
    #[arg(long)]
    continuation_lowercase: bool,

    /// Number of scroll ticks per "scroll up" / "scroll down" voice command.
    #[arg(long, default_value_t = DEFAULT_SCROLL_TICKS)]
    scroll_ticks: u32,

    /// Minimum number of words for a dictation output to be kept (shorter outputs
    /// are silently dropped). Set to 1 to keep everything.
    #[arg(long, default_value_t = 2)]
    min_dictation_words: usize,

    /// Dump each audio segment to a WAV file in this directory for debugging.
    #[arg(long)]
    dump_audio: Option<PathBuf>,

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

// ── Listen mode state machine ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum ListenMode {
    /// Aggressive VAD. Checks STT result for command keywords.
    Idle,
    /// Relaxed VAD. Outputs text, returns to Idle on silence timeout.
    Dictation,
}

// ── VAD configuration helpers ─────────────────────────────────────────────

fn make_vad_config(
    model_path: &str,
    threshold: f32,
    min_silence: f32,
    min_speech: f32,
    max_speech: f32,
) -> VadModelConfig {
    VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(model_path.to_owned()),
            threshold,
            min_silence_duration: min_silence,
            min_speech_duration: min_speech,
            max_speech_duration: max_speech,
            window_size: WINDOW_SIZE as i32,
        },
        sample_rate: SAMPLE_RATE as i32,
        num_threads: 1,
        debug: false,
        ..Default::default()
    }
}

fn create_vad(config: &VadModelConfig) -> Result<VoiceActivityDetector> {
    // Buffer size in seconds — generous to avoid overflow.
    VoiceActivityDetector::create(config, 60.0)
        .context("Failed to create VAD detector")
}

/// Safety: VadModelConfig is just data. VoiceActivityDetector wraps C++ but
/// is only used from the main thread.
struct Vad {
    idle: VoiceActivityDetector,
    dictation: VoiceActivityDetector,
    use_dictation: bool,
}

unsafe impl Send for Vad {}

impl Vad {
    fn new(idle_config: &VadModelConfig, dictation_config: &VadModelConfig) -> Result<Self> {
        Ok(Self {
            idle: create_vad(idle_config)?,
            dictation: create_vad(dictation_config)?,
            use_dictation: false,
        })
    }

    fn detector(&self) -> &VoiceActivityDetector {
        if self.use_dictation {
            &self.dictation
        } else {
            &self.idle
        }
    }

    fn switch_to_idle(&mut self) {
        self.use_dictation = false;
        self.idle.reset();
    }

    fn switch_to_dictation(&mut self) {
        self.use_dictation = true;
        self.dictation.reset();
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
    continuation_lowercase: bool,
    scroll_ticks: u32,
    prefill_samples: usize,
    dictation_silence_secs: f32,
    lowercase_timeout_secs: f32,
    dictation_max_speech_samples: usize,
    last_ended_with_punctuation: Cell<bool>,
    last_output_time: Cell<Option<Instant>>,
    /// Number of characters typed in the last segment (for undo via backspace).
    last_typed_chars: Cell<usize>,
    display_server: String,
    notify_id: Cell<u32>,
    tray_handle: Option<tray::TrayHandle>,
    /// Hotwords derived from command vocabulary, used in idle mode.
    command_hotwords: String,
    command_hotwords_score: f32,
    /// User-provided hotwords from --hotwords flag, used in dictation mode.
    dictation_hotwords: Option<String>,
    dictation_hotwords_score: f32,
    min_dictation_words: usize,
    dump_audio_dir: Option<PathBuf>,
    dump_audio_counter: AtomicU32,
    /// Voice command trigger aliases (from config file or defaults).
    aliases: ResolvedAliases,
    /// Modifier key mappings (spoken word → canonical name).
    modifiers: Vec<(String, String)>,
}

impl AppContext {
    fn new(cfg: ResolvedConfig) -> Self {
        let base_url = cfg.url.trim_end_matches('/');
        let endpoint_url = if cfg.smart {
            format!("{base_url}/v1/dictate/smart")
        } else {
            format!("{base_url}/v1/audio/transcriptions")
        };

        let display_server = cfg
            .display_server
            .unwrap_or_else(detect_display_server);

        let command_hotwords = build_command_hotwords(&cfg.aliases, &cfg.modifiers);

        Self {
            http_client: reqwest::blocking::Client::new(),
            endpoint_url,
            smart: cfg.smart,
            type_text: cfg.type_text,
            notify: cfg.notify,
            sound: cfg.sound,
            continuation_lowercase: cfg.continuation_lowercase,
            scroll_ticks: cfg.scroll_ticks,
            prefill_samples: (cfg.prefill_ms as usize * SAMPLE_RATE as usize) / 1000,
            dictation_silence_secs: cfg.dictation_silence,
            lowercase_timeout_secs: cfg.lowercase_timeout,
            dictation_max_speech_samples: (cfg.dictation_max_speech * SAMPLE_RATE as f32) as usize,
            last_ended_with_punctuation: Cell::new(true),
            last_output_time: Cell::new(None),
            last_typed_chars: Cell::new(0),
            display_server,
            notify_id: Cell::new(0),
            tray_handle: None,
            command_hotwords,
            command_hotwords_score: cfg.command_hotwords_score,
            dictation_hotwords: cfg.hotwords,
            dictation_hotwords_score: cfg.dictation_hotwords_score,
            min_dictation_words: cfg.min_dictation_words,
            dump_audio_dir: cfg.dump_audio,
            dump_audio_counter: AtomicU32::new(0),
            aliases: cfg.aliases,
            modifiers: cfg.modifiers,
        }
    }

    /// If `--dump-audio` is set, write the WAV data to a sequentially numbered file.
    fn maybe_dump_wav(&self, wav: &[u8], label: &str) {
        if let Some(dir) = &self.dump_audio_dir {
            let n = self.dump_audio_counter.fetch_add(1, Ordering::Relaxed);
            let path = dir.join(format!("{n:04}_{label}.wav"));
            match std::fs::write(&path, wav) {
                Ok(()) => info!(?path, "Dumped audio segment"),
                Err(e) => warn!(?path, "Failed to dump audio: {e}"),
            }
        }
    }

    fn set_tray_status(&self, status: TrayStatus) {
        if let Some(handle) = &self.tray_handle {
            handle.update(status);
        }
    }

    /// Should we lowercase the first letter of this segment?
    fn should_lowercase_continuation(&self, mode: ListenMode) -> bool {
        if mode == ListenMode::Dictation {
            // In dictation mode, always lowercase continuations unless
            // the previous segment ended with terminal punctuation or
            // enough time has passed to treat this as a fresh utterance.
            if self.last_ended_with_punctuation.get() {
                return false;
            }
            if let Some(t) = self.last_output_time.get() {
                if t.elapsed().as_secs_f32() > self.lowercase_timeout_secs {
                    return false;
                }
            } else {
                return false; // First segment ever
            }
            true
        } else {
            // In idle mode, respect the CLI flag.
            self.continuation_lowercase && !self.last_ended_with_punctuation.get()
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

fn type_into_window(text: &str, display_server: &str) {
    debug!(text, display_server, "type_into_window");
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
        warn!("Text injection failed: {e}");
    }
}

fn send_key(key: &str, display_server: &str) {
    debug!(key, display_server, "send_key");
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
        warn!("Key injection failed: {e}");
    }
}

fn send_modified_key(modifiers: &[&str], key: &str, display_server: &str) {
    debug!(?modifiers, key, display_server, "send_modified_key");
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
        warn!("Modified key injection failed: {e}");
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
            warn!("Failed to get display geometry: {e}");
            return;
        }
    };
    let geometry = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = geometry.trim().split_whitespace().collect();
    if parts.len() < 2 {
        warn!("Unexpected geometry output: {geometry}");
        return;
    }
    let (width, height) = match (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
        (Ok(w), Ok(h)) => (w, h),
        _ => {
            warn!("Failed to parse geometry: {geometry}");
            return;
        }
    };

    let x = if right { width * 3 / 4 } else { width / 4 };
    let y = if bottom { height * 3 / 4 } else { height / 4 };

    let result = std::process::Command::new("xdotool")
        .args(["mousemove", "--sync", &x.to_string(), &y.to_string(), "click", "1"])
        .status();
    if let Err(e) = result {
        warn!("Mouse click failed: {e}");
    }
}

fn click_coordinate(x_pct: u32, y_pct: u32) {
    let output = match std::process::Command::new("xdotool")
        .args(["getdisplaygeometry"])
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            warn!("Failed to get display geometry: {e}");
            return;
        }
    };
    let geometry = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = geometry.trim().split_whitespace().collect();
    if parts.len() < 2 {
        warn!("Unexpected geometry output: {geometry}");
        return;
    }
    let (width, height) = match (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
        (Ok(w), Ok(h)) => (w, h),
        _ => {
            warn!("Failed to parse geometry: {geometry}");
            return;
        }
    };

    let x = width * x_pct.min(100) / 100;
    let y = height * y_pct.min(100) / 100;

    let result = std::process::Command::new("xdotool")
        .args(["mousemove", "--sync", &x.to_string(), &y.to_string(), "click", "1"])
        .status();
    if let Err(e) = result {
        warn!("Mouse click failed: {e}");
    }
}

fn scroll(up: bool, ticks: u32) {
    let button = if up { "4" } else { "5" };
    let repeat = ticks.to_string();
    let result = std::process::Command::new("xdotool")
        .args(["click", "--repeat", &repeat, "--delay", "10", button])
        .status();
    if let Err(e) = result {
        warn!("Scroll failed: {e}");
    }
}

// ── Voice commands ──────────────────────────────────────────────────────

/// Find the earliest occurrence of any trigger word from `triggers` in
/// `haystack[search_start..]`.  Returns `(position_relative_to_search_start, matched_len)`.
fn find_any_trigger(haystack: &str, search_start: usize, triggers: &[String]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for trigger in triggers {
        if let Some(p) = haystack[search_start..].find(trigger.as_str()) {
            if best.as_ref().is_none_or(|&(bp, _)| p < bp) {
                best = Some((p, trigger.len()));
            }
        }
    }
    best
}

/// An action that a voice command can trigger.
#[derive(Clone, Debug)]
enum VoiceAction {
    /// Simulate a keypress (e.g. "Return", "Tab").
    Key(String),
    /// Simulate a modified keypress (e.g. Ctrl+Return).
    ModifiedKey {
        modifiers: Vec<String>,
        key: String,
    },
    /// Move the mouse to a screen quadrant and click.
    ClickQuadrant {
        /// Horizontal position: false = left, true = right.
        right: bool,
        /// Vertical position: false = top, true = bottom.
        bottom: bool,
    },
    /// Move the mouse to a percentage-based coordinate and click.
    ClickCoordinate {
        /// X position as a percentage (0–100) of screen width.
        x: u32,
        /// Y position as a percentage (0–100) of screen height.
        y: u32,
    },
    /// Scroll the mouse wheel.
    Scroll {
        /// true = scroll up, false = scroll down.
        up: bool,
        /// Number of times to apply the scroll tick count.
        repeats: u32,
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

/// Build a comma-separated hotwords string from the command vocabulary.
/// These are all words that might appear in voice commands, used to boost
/// recognition accuracy when we don't yet know if an utterance is a command.
fn build_command_hotwords(aliases: &ResolvedAliases, modifiers: &[(String, String)]) -> String {
    let mut words: Vec<String> = Vec::new();

    // Command trigger keywords (canonical + aliases)
    words.extend(["press", "click", "scroll", "slash", "command", "undo"].iter().map(|s| s.to_string()));
    words.extend(aliases.click.iter().cloned());
    words.extend(aliases.press.iter().cloned());
    words.extend(aliases.scroll.iter().cloned());
    words.extend(aliases.undo.iter().cloned());
    for trigger in &aliases.slash_command {
        words.extend(trigger.iter().cloned());
    }

    // Click quadrant words
    words.extend(["upper", "lower", "left", "right"].iter().map(|s| s.to_string()));

    // Scroll directions
    words.extend(["up", "down", "top", "bottom"].iter().map(|s| s.to_string()));

    // Key names (named keys only; single letters are too ambiguous)
    words.extend([
        "enter", "return", "tab", "space", "backspace", "delete",
        "escape", "home", "end",
    ].iter().map(|s| s.to_string()));

    // Modifier keys
    for (alias, _) in modifiers {
        words.push(alias.clone());
    }

    // Deduplicate (some words appear in multiple roles, e.g. "left", "right", "up", "down")
    words.sort_unstable();
    words.dedup();

    // Multi-word command phrases — boosting these as phrases gives stronger
    // recognition than individual words alone.
    let mut phrases: Vec<String> = vec![
        "slash command".into(),
        "click upper left".into(),
        "click upper right".into(),
        "click lower left".into(),
        "click lower right".into(),
        "scroll up".into(),
        "scroll down".into(),
        "scroll top".into(),
        "scroll bottom".into(),
        "press enter".into(),
        "press tab".into(),
        "press space".into(),
        "press backspace".into(),
        "press delete".into(),
        "press escape".into(),
        "press control".into(),
        "press shift".into(),
        "press alt".into(),
        "press super".into(),
    ];

    // Add alias phrases for slash command triggers
    for trigger in &aliases.slash_command {
        phrases.push(trigger.join(" "));
    }
    // Add alias phrases for click quadrant commands
    for trigger in &aliases.click {
        for dir in &["upper left", "upper right", "lower left", "lower right"] {
            phrases.push(format!("{trigger} {dir}"));
        }
    }

    phrases.sort();
    phrases.dedup();

    let mut all = words;
    all.extend(phrases);
    all.join(",")
}

/// Returns true if `s` contains at least one alphanumeric character (i.e. is
/// not purely punctuation/whitespace left over after stripping a command).
fn has_word_chars(s: &str) -> bool {
    s.chars().any(|c| c.is_alphanumeric())
}

/// Try to find "slash command <word>" at position `from` in `lower`.
/// Returns `(start, end, slash_word)` where `slash_word` is the captured word
/// lowercased with a leading `/`.
fn find_slash_command(lower: &str, from: usize, slash_triggers: &[Vec<String>]) -> Option<(usize, usize, String)> {
    let mut best: Option<(usize, usize, String)> = None;
    for trigger in slash_triggers {
        let trigger_refs: Vec<&str> = trigger.iter().map(|s| s.as_str()).collect();
        if let Some((start, end)) = find_phrase(lower, from, &trigger_refs) {
            if best.as_ref().is_none_or(|b| start < b.0) {
                // Skip whitespace/punctuation after the trigger phrase
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
                    best = Some((start, end + skip + word_len, format!("/{word}")));
                }
            }
        }
    }
    best
}

/// Parse a spoken number word into its value. Returns None if the word is not a number.
fn parse_number_word(word: &str) -> Option<u32> {
    match word {
        "zero" | "oh" => Some(0),
        "one" => Some(1),
        "two" | "to" | "too" => Some(2),
        "three" => Some(3),
        "four" | "for" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        "thirty" => Some(30),
        "forty" => Some(40),
        "fifty" => Some(50),
        "sixty" => Some(60),
        "seventy" => Some(70),
        "eighty" => Some(80),
        "ninety" => Some(90),
        "hundred" => Some(100),
        _ => None,
    }
}

/// Try to parse a spoken number (0–100) from words starting at `word_idx`.
/// Returns (value, number_of_words_consumed) or None.
fn parse_spoken_number(words: &[&str], word_idx: usize) -> Option<(u32, usize)> {
    let first = words.get(word_idx)?;
    let first_val = parse_number_word(first)?;

    // "one hundred"
    if first_val == 1 {
        if let Some(&next) = words.get(word_idx + 1) {
            if next == "hundred" {
                return Some((100, 2));
            }
        }
    }

    // "hundred" alone
    if *first == "hundred" {
        return Some((100, 1));
    }

    // Tens word (20, 30, ..., 90) optionally followed by a ones word (1–9)
    if first_val >= 20 && first_val <= 90 && first_val % 10 == 0 {
        if let Some(&next) = words.get(word_idx + 1) {
            if let Some(ones) = parse_number_word(next) {
                if ones >= 1 && ones <= 9 {
                    return Some((first_val + ones, 2));
                }
            }
        }
        return Some((first_val, 1));
    }

    // Single number word (0–19)
    if first_val <= 19 {
        return Some((first_val, 1));
    }

    None
}

/// Try to find "click <number> <number>" in `lower` starting from `from`.
/// Returns (start_byte, end_byte, x, y) or None.
fn find_click_coordinate(lower: &str, from: usize, click_triggers: &[String]) -> Option<(usize, usize, u32, u32)> {
    let haystack = &lower[from..];

    let mut search_start = 0;
    while let Some((p, matched_len)) = find_any_trigger(haystack, search_start, click_triggers) {
        let click_start = from + search_start + p;
        let after_click = click_start + matched_len;

        // Collect the words after the trigger, skipping punctuation/whitespace
        let rest = &lower[after_click..];
        let word_chars: Vec<&str> = rest
            .split(|c: char| c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '-' | '\'' | '"'))
            .filter(|s| !s.is_empty())
            .take(4) // at most 4 words needed (2 per number)
            .collect();

        if let Some((y, y_consumed)) = parse_spoken_number(&word_chars, 0) {
            if y <= 100 {
                if let Some((x, x_consumed)) = parse_spoken_number(&word_chars, y_consumed) {
                    if x <= 100 {
                        // Find the byte offset of the end of the last consumed word
                        let total_words = y_consumed + x_consumed;
                        let mut end = after_click;
                        let mut words_found = 0;
                        let mut in_word = false;
                        for (i, c) in lower[after_click..].char_indices() {
                            let is_sep = c.is_whitespace()
                                || matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '-' | '\'' | '"');
                            if in_word && is_sep {
                                words_found += 1;
                                if words_found == total_words {
                                    end = after_click + i;
                                    break;
                                }
                                in_word = false;
                            } else if !is_sep {
                                in_word = true;
                            }
                        }
                        if words_found < total_words && in_word {
                            // Last word runs to end of string
                            end = lower.len();
                        }
                        return Some((click_start, end, x, y));
                    }
                }
            }
        }

        search_start = search_start + p + matched_len;
    }

    None
}

/// Try to find "press <modifier> <key>" in `lower` starting from `from`.
/// Supports any modifier (ctrl, shift, alt, super) and recognises single
/// letters a–z plus common named keys.  "press" is required.
/// Returns (start_byte, end_byte, VoiceAction) or None.
fn find_modified_key(lower: &str, from: usize, press_triggers: &[String], modifiers: &[(String, String)]) -> Option<(usize, usize, VoiceAction)> {
    let haystack = &lower[from..];
    let mut best: Option<(usize, usize, VoiceAction)> = None;

    // Scan for "press" (or aliases) — then check if the next word is a modifier
    let mut search_start = 0;
    while let Some((p, matched_len)) = find_any_trigger(haystack, search_start, press_triggers) {
        let press_start = from + search_start + p;
        let after_press = press_start + matched_len;

        // The word after "press" must be a modifier alias
        if let Some((modifier_word, mod_word_end)) = next_word(lower, after_press) {
            if let Some(canonical) = modifier_canonical(modifier_word, modifiers) {
                // The word after the modifier must be a key name
                if let Some((key_name, word_end)) = next_key_word(lower, mod_word_end) {
                    let candidate = (
                        press_start,
                        word_end,
                        VoiceAction::ModifiedKey {
                            modifiers: vec![canonical.into()],
                            key: key_name.into(),
                        },
                    );
                    if best.as_ref().is_none_or(|b| press_start < b.0) {
                        best = Some(candidate);
                    }
                }
            }
        }

        search_start = search_start + p + matched_len;
    }

    best
}

/// Try to find "press <key>" (without a modifier) in `lower` starting from
/// `from`.  Returns (start_byte, end_byte, VoiceAction) or None.
fn find_key_press(lower: &str, from: usize, press_triggers: &[String]) -> Option<(usize, usize, VoiceAction)> {
    let haystack = &lower[from..];
    let mut best: Option<(usize, usize, VoiceAction)> = None;

    let mut search_start = 0;
    while let Some((p, matched_len)) = find_any_trigger(haystack, search_start, press_triggers) {
        let press_start = from + search_start + p;
        let after_press = press_start + matched_len;

        // The word after "press" must not be a modifier — those are handled by
        // find_modified_key and we don't want "press control c" to fire as
        // Key("control").
        if let Some((key_name, word_end)) = next_key_word(lower, after_press) {
            let candidate = (
                press_start,
                word_end,
                VoiceAction::Key(key_name.into()),
            );
            if best.as_ref().is_none_or(|b| press_start < b.0) {
                best = Some(candidate);
            }
        }

        search_start = search_start + p + matched_len;
    }

    best
}

/// Skip separators after `pos` in `lower`, read the next word, and try to map
/// it to a key name.  Returns `(key_name, end_byte_offset)` or None.
fn next_key_word(lower: &str, pos: usize) -> Option<(&'static str, usize)> {
    let rest = &lower[pos..];
    let skip = rest
        .chars()
        .take_while(|c| is_separator(*c))
        .map(|c| c.len_utf8())
        .sum::<usize>();

    if skip == 0 {
        return None;
    }

    let after_sep = pos + skip;
    let word_end = lower[after_sep..]
        .find(|c: char| c.is_whitespace() || is_separator(c))
        .map_or(lower.len(), |i| after_sep + i);
    let word = &lower[after_sep..word_end];

    key_name(word).map(|k| (k, word_end))
}

/// Skip separators after `pos` in `lower` and read the next word.
/// Returns `(word_slice, end_byte_offset)` or None.
fn next_word(lower: &str, pos: usize) -> Option<(&str, usize)> {
    let rest = &lower[pos..];
    let skip: usize = rest
        .chars()
        .take_while(|c| is_separator(*c))
        .map(|c| c.len_utf8())
        .sum();

    if skip == 0 {
        return None;
    }

    let after_sep = pos + skip;
    let word_end = lower[after_sep..]
        .find(|c: char| c.is_whitespace() || is_separator(c))
        .map_or(lower.len(), |i| after_sep + i);

    if after_sep == word_end {
        return None;
    }

    Some((&lower[after_sep..word_end], word_end))
}

/// If `word` is a modifier alias, return its canonical xdotool/wtype name.
fn modifier_canonical<'a>(word: &str, modifiers: &'a [(String, String)]) -> Option<&'a str> {
    modifiers
        .iter()
        .find(|(alias, _)| alias == word)
        .map(|(_, canonical)| canonical.as_str())
}

fn is_separator(c: char) -> bool {
    c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '-' | '\'' | '"')
}

/// Map a spoken word to its xdotool/wtype key name.
/// Returns `None` if the word is not a recognised key.
fn key_name(word: &str) -> Option<&'static str> {
    // Single letters a–z
    if word.len() == 1 && word.as_bytes()[0].is_ascii_lowercase() {
        return Some(match word {
            "a" => "a", "b" => "b", "c" => "c", "d" => "d",
            "e" => "e", "f" => "f", "g" => "g", "h" => "h",
            "i" => "i", "j" => "j", "k" => "k", "l" => "l",
            "m" => "m", "n" => "n", "o" => "o", "p" => "p",
            "q" => "q", "r" => "r", "s" => "s", "t" => "t",
            "u" => "u", "v" => "v", "w" => "w", "x" => "x",
            "y" => "y", "z" => "z",
            _ => unreachable!(),
        });
    }
    // Common named keys
    match word {
        "enter" | "return" => Some("Return"),
        "tab" => Some("Tab"),
        "space" => Some("space"),
        "backspace" => Some("BackSpace"),
        "delete" => Some("Delete"),
        "escape" | "esc" => Some("Escape"),
        "up" => Some("Up"),
        "down" => Some("Down"),
        "left" => Some("Left"),
        "right" => Some("Right"),
        "home" => Some("Home"),
        "end" => Some("End"),
        _ => None,
    }
}

/// Try to find "scroll up/down/top/bottom [up/down...]" in `lower` starting
/// from `from`.  Repeated direction words multiply the scroll count.
/// "top" and "bottom" scroll 100 repeats in the respective direction.
/// Returns (start_byte, end_byte, VoiceAction) or None.
fn find_scroll(lower: &str, from: usize, scroll_triggers: &[String]) -> Option<(usize, usize, VoiceAction)> {
    let haystack = &lower[from..];
    let mut best: Option<(usize, usize, VoiceAction)> = None;

    let mut search_start = 0;
    while let Some((p, matched_len)) = find_any_trigger(haystack, search_start, scroll_triggers) {
        let scroll_start = from + search_start + p;
        let after_scroll = scroll_start + matched_len;

        // Collect words after "scroll", consuming direction words greedily
        let mut pos = after_scroll;
        let mut up: Option<bool> = None;
        let mut repeats: u32 = 0;

        loop {
            // Skip separators
            let rest = &lower[pos..];
            let skip: usize = rest
                .chars()
                .take_while(|c| is_separator(*c))
                .map(|c| c.len_utf8())
                .sum();

            if skip == 0 && pos != after_scroll {
                break;
            }
            let word_start = pos + skip;
            let word_end = lower[word_start..]
                .find(|c: char| c.is_whitespace() || is_separator(c))
                .map_or(lower.len(), |i| word_start + i);
            if word_start == word_end {
                break;
            }
            let word = &lower[word_start..word_end];

            match word {
                "up" => {
                    if up == Some(false) {
                        break; // direction changed — stop
                    }
                    up = Some(true);
                    repeats += 1;
                    pos = word_end;
                }
                "down" => {
                    if up == Some(true) {
                        break;
                    }
                    up = Some(false);
                    repeats += 1;
                    pos = word_end;
                }
                "top" => {
                    if up.is_some() {
                        break; // already have a direction
                    }
                    up = Some(true);
                    repeats = 100;
                    pos = word_end;
                    break; // "top" is terminal
                }
                "bottom" => {
                    if up.is_some() {
                        break;
                    }
                    up = Some(false);
                    repeats = 100;
                    pos = word_end;
                    break;
                }
                _ => break,
            }
        }

        if let Some(up) = up {
            if repeats > 0 {
                let candidate = (
                    scroll_start,
                    pos,
                    VoiceAction::Scroll { up, repeats },
                );
                if best.as_ref().is_none_or(|b| scroll_start < b.0) {
                    best = Some(candidate);
                }
            }
        }

        search_start = search_start + p + matched_len;
    }

    best
}

/// Returns true when the entire segment matches one of the undo triggers
/// (possibly followed by punctuation), e.g. "Undo", "undo.", "Undo!".
fn is_undo_command(text: &str, undo_triggers: &[String]) -> bool {
    let stripped: String = text
        .chars()
        .filter(|c| !c.is_whitespace() && !matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | '-' | '\'' | '"'))
        .collect();
    undo_triggers.iter().any(|t| stripped.eq_ignore_ascii_case(t))
}

/// Scan `text` for voice command phrases (case-insensitive, tolerant of
/// punctuation between words) and split into text segments and command actions.
fn process_voice_commands<'a>(text: &'a str, ctx: &AppContext) -> Vec<TextAction<'a>> {
    let lower = text.to_lowercase();
    let commands = voice_commands();
    let mut actions: Vec<TextAction<'_>> = Vec::new();
    let mut cursor = 0;
    // Commands are only recognised at the front of the input — once real
    // dictated text appears, the rest is treated as literal text.  This
    // prevents words like "press", "click", or "scroll" in normal speech
    // from being swallowed.  A slash command at the very start also enters
    // command mode.
    let mut seen_text = false;

    while cursor < text.len() {
        if seen_text {
            // Command mode is over — emit everything remaining as text.
            let rest = text[cursor..].trim();
            if has_word_chars(rest) {
                actions.push(TextAction::Text(rest));
            }
            break;
        }

        let mut best: Option<(usize, usize, VoiceAction)> = None;

        // Only look for commands if there is no word-character text between
        // the cursor and the candidate — that would mean dictated text
        // precedes the command, ending command mode.
        let no_text_before = |start: usize| !has_word_chars(&text[cursor..start]);

        // Static commands (click quadrant) — try each click alias
        for cmd in commands {
            for trigger in &ctx.aliases.click {
                let mut words: Vec<&str> = cmd.words.to_vec();
                words[0] = trigger.as_str();
                if let Some((start, end)) = find_phrase(&lower, cursor, &words) {
                    if no_text_before(start) && best.as_ref().is_none_or(|b| start < b.0) {
                        best = Some((start, end, cmd.action.clone()));
                    }
                }
            }
        }

        // "scroll up/down/top/bottom [up/down...]" dynamic command
        if let Some((start, end, action)) = find_scroll(&lower, cursor, &ctx.aliases.scroll) {
            if no_text_before(start) && best.as_ref().is_none_or(|b| start < b.0) {
                best = Some((start, end, action));
            }
        }

        // "click <number> <number>" coordinate command
        if let Some((start, end, x, y)) = find_click_coordinate(&lower, cursor, &ctx.aliases.click) {
            if no_text_before(start) && best.as_ref().is_none_or(|b| start < b.0) {
                best = Some((start, end, VoiceAction::ClickCoordinate { x, y }));
            }
        }

        // "[press] <modifier> <key>" dynamic command
        if let Some((start, end, action)) = find_modified_key(&lower, cursor, &ctx.aliases.press, &ctx.modifiers) {
            if no_text_before(start) && best.as_ref().is_none_or(|b| start < b.0) {
                best = Some((start, end, action));
            }
        }

        // "press <key>" (unmodified) dynamic command
        if let Some((start, end, action)) = find_key_press(&lower, cursor, &ctx.aliases.press) {
            if no_text_before(start) && best.as_ref().is_none_or(|b| start < b.0) {
                best = Some((start, end, action));
            }
        }

        // "slash command <word>" at the very beginning of the text
        let slash = if cursor == 0 {
            find_slash_command(&lower, 0, &ctx.aliases.slash_command)
                .filter(|(start, ..)| !has_word_chars(&text[..*start]))
        } else {
            None
        };

        // Pick whichever matched earliest
        let use_voice = match (&best, &slash) {
            (Some((vs, ..)), Some((ss, ..))) => vs <= ss,
            (Some(_), None) => true,
            _ => false,
        };

        if use_voice {
            let (start, end, action) = best.unwrap();
            let before = text[cursor..start].trim();
            if has_word_chars(before) {
                // Text before the first command — emit it and leave command mode
                actions.push(TextAction::Text(before));
                seen_text = true;
                continue;
            }
            actions.push(TextAction::Command(action.clone()));
            cursor = end;

            // After a key-press command, greedily consume continuation keys
            // so that "press enter tab escape" produces three key actions
            // and "press enter control c" produces Enter then Ctrl+C.
            if matches!(action, VoiceAction::Key(_) | VoiceAction::ModifiedKey { .. }) {
                loop {
                    // Try modifier+key first (e.g. "control c")
                    if let Some((mod_word, mod_end)) = next_word(&lower, cursor) {
                        if let Some(canonical) = modifier_canonical(mod_word, &ctx.modifiers) {
                            if let Some((kn, key_end)) = next_key_word(&lower, mod_end) {
                                actions.push(TextAction::Command(VoiceAction::ModifiedKey {
                                    modifiers: vec![canonical.into()],
                                    key: kn.into(),
                                }));
                                cursor = key_end;
                                continue;
                            }
                        }
                    }
                    // Try plain key name (e.g. "tab")
                    if let Some((kn, key_end)) = next_key_word(&lower, cursor) {
                        actions.push(TextAction::Command(VoiceAction::Key(kn.into())));
                        cursor = key_end;
                        continue;
                    }
                    break;
                }
            }
        } else if let Some((start, end, slash_text)) = slash {
            let before = text[cursor..start].trim();
            if has_word_chars(before) {
                actions.push(TextAction::Text(before));
                seen_text = true;
                continue;
            }
            actions.push(TextAction::OwnedText(slash_text));
            cursor = end;
        } else {
            // No command found — emit the rest as text
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
    hotwords: Option<&str>,
    hotwords_score: f32,
) -> Result<String> {
    let part = reqwest::blocking::multipart::Part::bytes(wav_data)
        .file_name("audio.wav")
        .mime_str("audio/wav")?;
    let mut form = reqwest::blocking::multipart::Form::new().part("file", part);
    if let Some(hw) = hotwords {
        form = form.text("hotwords", hw.to_string());
        if hotwords_score > 0.0 {
            form = form.text("hotwords_score", hotwords_score.to_string());
        }
    }

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
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    info!("Downloading Silero VAD model");
    let client = reqwest::blocking::Client::new();
    let response = client.get(URL).send().context("Failed to download VAD model")?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with status {}", response.status());
    }

    let bytes = response.bytes()?;
    let mut file = std::fs::File::create(dest)?;
    file.write_all(&bytes)?;

    info!(path = %dest.display(), "VAD model downloaded");
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

/// Result of classifying a speech segment (without outputting).
enum ClassifyResult {
    /// Transcription contains voice commands only (no text to dictate).
    Command(String),
    /// Transcription is regular text (not a command). Not output in idle mode.
    Text,
    /// Empty transcription.
    Empty,
    /// Server or encoding error.
    Error,
}

/// Send audio to the server for STT and classify the result as command or text.
/// Does NOT output anything — use `output_transcription` for that.
fn classify_segment(samples: &[f32], ctx: &AppContext, mode: ListenMode) -> ClassifyResult {
    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    info!(duration_secs, "Classifying speech segment");

    ctx.set_tray_status(TrayStatus::Processing);
    if ctx.notify {
        let id = show_notification("Processing...", "Classifying speech", ctx.notify_id.get());
        ctx.notify_id.set(id);
    }

    let wav = match encode_wav(samples) {
        Ok(w) => w,
        Err(e) => {
            error!("WAV encode failed: {e}");
            finish_segment_ui(ctx, mode);
            return ClassifyResult::Error;
        }
    };
    ctx.maybe_dump_wav(&wav, "idle");

    let text = match send_to_server(&ctx.http_client, &ctx.endpoint_url, wav, ctx.smart, Some(&ctx.command_hotwords), ctx.command_hotwords_score) {
        Ok(t) => t,
        Err(e) => {
            error!("Classification request failed: {e}");
            finish_segment_ui(ctx, mode);
            if ctx.sound {
                play_sound("dialog-error");
            }
            return ClassifyResult::Error;
        }
    };

    let text = text.trim().to_string();
    if text.is_empty() {
        info!("Classification returned empty");
        finish_segment_ui(ctx, mode);
        return ClassifyResult::Empty;
    }

    info!(text = %text, "Classification result");
    finish_segment_ui(ctx, mode);

    // Undo is a special command not handled by process_voice_commands.
    if is_undo_command(&text, &ctx.aliases.undo) {
        return ClassifyResult::Command(text);
    }

    // Check if the text starts with a command keyword.
    let actions = process_voice_commands(&text, ctx);
    let has_text = actions.iter().any(|a| matches!(a, TextAction::Text(_) | TextAction::OwnedText(_)));

    if has_text {
        ClassifyResult::Text
    } else {
        ClassifyResult::Command(text)
    }
}

/// Execute voice commands from a transcription (used in idle mode).
fn execute_commands(text: &str, ctx: &AppContext) {
    if is_undo_command(text, &ctx.aliases.undo) {
        let n = ctx.last_typed_chars.get();
        if n > 0 && ctx.type_text {
            debug!(count = n, "Sending undo backspaces");
            for _ in 0..n {
                send_key("BackSpace", &ctx.display_server);
            }
            ctx.last_typed_chars.set(0);
        } else if !ctx.type_text {
            println!("[undo]");
        }
        return;
    }

    let actions = process_voice_commands(text, ctx);
    debug!(?actions, "Parsed voice actions");

    let mut just_typed = false;
    for action in &actions {
        match action {
            TextAction::Command(VoiceAction::Key(ref key)) => {
                if ctx.type_text {
                    if just_typed {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    send_key(key, &ctx.display_server);
                } else {
                    println!();
                }
                just_typed = false;
            }
            TextAction::Command(VoiceAction::ModifiedKey { ref modifiers, ref key }) => {
                if ctx.type_text {
                    if just_typed {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    let mod_refs: Vec<&str> = modifiers.iter().map(|s| s.as_str()).collect();
                    send_modified_key(&mod_refs, key, &ctx.display_server);
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
            TextAction::Command(VoiceAction::ClickCoordinate { x, y }) => {
                if ctx.type_text {
                    click_coordinate(*x, *y);
                } else {
                    println!("[click {x} {y}]");
                }
                just_typed = false;
            }
            TextAction::Command(VoiceAction::Scroll { up, repeats }) => {
                if ctx.type_text {
                    scroll(*up, ctx.scroll_ticks * repeats);
                } else {
                    let dir = if *up { "up" } else { "down" };
                    println!("[scroll {dir} x{repeats}]");
                }
                just_typed = false;
            }
            // Commands shouldn't have text actions, but handle gracefully
            TextAction::Text(_) | TextAction::OwnedText(_) => {}
        }
    }

    if ctx.sound {
        play_sound("message-new-instant");
    }
}

/// Send accumulated dictation audio to the server, output the transcription.
fn flush_utterance(utterance_audio: &mut Vec<f32>, ctx: &AppContext, tray_mode: ListenMode) {
    if utterance_audio.is_empty() {
        return;
    }

    let duration_secs = utterance_audio.len() as f64 / SAMPLE_RATE as f64;
    info!(duration_secs, "Transcribing dictation segment");

    if ctx.notify {
        let id = show_notification("Processing...", "Transcribing dictation", ctx.notify_id.get());
        ctx.notify_id.set(id);
    }

    let wav = match encode_wav(utterance_audio) {
        Ok(w) => w,
        Err(e) => {
            error!("WAV encode failed: {e}");
            utterance_audio.clear();
            finish_segment_ui(ctx, tray_mode);
            return;
        }
    };
    ctx.maybe_dump_wav(&wav, "dictation");

    let text = match send_to_server(&ctx.http_client, &ctx.endpoint_url, wav, ctx.smart, ctx.dictation_hotwords.as_deref(), ctx.dictation_hotwords_score) {
        Ok(t) => t,
        Err(e) => {
            error!("Dictation request failed: {e}");
            utterance_audio.clear();
            finish_segment_ui(ctx, tray_mode);
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
            return;
        }
    };

    utterance_audio.clear();

    let text = text.trim();
    if text.is_empty() {
        info!("Dictation returned empty");
        finish_segment_ui(ctx, tray_mode);
        return;
    }

    let word_count = text.split_whitespace().count();
    if word_count < ctx.min_dictation_words {
        info!(word_count, min = ctx.min_dictation_words, text, "Dictation too short, dropping");
        finish_segment_ui(ctx, tray_mode);
        return;
    }

    info!("Dictation transcribed successfully");

    // Apply continuation lowercasing for dictation.
    let text = if ctx.should_lowercase_continuation(ListenMode::Dictation) {
        lowercase_first_letter(text)
    } else {
        text.to_string()
    };

    ctx.last_ended_with_punctuation
        .set(ends_with_sentence_punctuation(&text));
    ctx.last_output_time.set(Some(Instant::now()));

    // In dictation mode, output all text (commands at start are still processed).
    let actions = process_voice_commands(&text, ctx);
    debug!(?actions, "Parsed dictation actions");

    let mut just_typed = false;
    let mut chars_typed: usize = 0;
    for action in &actions {
        match action {
            TextAction::Text(t) => {
                if ctx.type_text {
                    type_into_window(t, &ctx.display_server);
                    chars_typed += t.len() + 1;
                    just_typed = true;
                } else {
                    print!("{t} ");
                }
            }
            TextAction::OwnedText(t) => {
                if ctx.type_text {
                    type_into_window(t, &ctx.display_server);
                    chars_typed += t.len() + 1;
                    just_typed = true;
                } else {
                    print!("{t} ");
                }
            }
            TextAction::Command(VoiceAction::Key(ref key)) => {
                if ctx.type_text {
                    if just_typed {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    send_key(key, &ctx.display_server);
                } else {
                    println!();
                }
                just_typed = false;
            }
            TextAction::Command(VoiceAction::ModifiedKey { ref modifiers, ref key }) => {
                if ctx.type_text {
                    if just_typed {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    let mod_refs: Vec<&str> = modifiers.iter().map(|s| s.as_str()).collect();
                    send_modified_key(&mod_refs, key, &ctx.display_server);
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
            TextAction::Command(VoiceAction::ClickCoordinate { x, y }) => {
                if ctx.type_text {
                    click_coordinate(*x, *y);
                } else {
                    println!("[click {x} {y}]");
                }
                just_typed = false;
            }
            TextAction::Command(VoiceAction::Scroll { up, repeats }) => {
                if ctx.type_text {
                    scroll(*up, ctx.scroll_ticks * repeats);
                } else {
                    let dir = if *up { "up" } else { "down" };
                    println!("[scroll {dir} x{repeats}]");
                }
                just_typed = false;
            }
        }
    }

    ctx.last_typed_chars.set(chars_typed);
    if !ctx.type_text {
        let _ = io::stdout().flush();
    }

    if ctx.sound {
        play_sound("message-new-instant");
    }

    finish_segment_ui(ctx, tray_mode);
}

fn finish_segment_ui(ctx: &AppContext, mode: ListenMode) {
    ctx.set_tray_status(match mode {
        ListenMode::Idle => TrayStatus::Listening,
        ListenMode::Dictation => TrayStatus::Dictating,
    });
    if ctx.notify {
        let id = show_notification(
            "Listening...",
            "Telemuze is active",
            ctx.notify_id.get(),
        );
        ctx.notify_id.set(id);
    }
}

/// Flush any in-progress speech. In dictation mode, sends accumulated
/// utterance audio. In idle mode, processes remaining VAD segments as commands.
fn drain_vad(vad: &mut Vad, utterance_audio: &mut Vec<f32>, ctx: &AppContext, mode: ListenMode) {
    // In dictation mode, send accumulated audio if any.
    if mode == ListenMode::Dictation {
        ctx.set_tray_status(TrayStatus::Processing);
        flush_utterance(utterance_audio, ctx, ListenMode::Idle);
    }

    // Drain any remaining VAD segments (idle mode commands).
    vad.detector().flush();
    while !vad.detector().is_empty() {
        let segment_data = vad.detector().front().and_then(|seg| {
            let samples = seg.samples().to_vec();
            (mode == ListenMode::Idle).then_some(samples)
        });
        vad.detector().pop();

        if let Some(samples) = segment_data {
            if samples.len() >= (SAMPLE_RATE as usize / 10) {
                if let ClassifyResult::Command(text) = classify_segment(&samples, ctx, ListenMode::Idle) {
                    execute_commands(&text, ctx);
                }
            }
        }
    }
    vad.detector().reset();
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
    let matches = <Cli as clap::CommandFactory>::command().get_matches();
    let cli = <Cli as clap::FromArgMatches>::from_arg_matches(&matches)?;

    // Handle subcommands that send IPC to a running daemon
    match &cli.command {
        Some(Command::Toggle { socket }) => return send_ipc_command(socket, "toggle"),
        Some(Command::Flush { socket }) => return send_ipc_command(socket, "flush"),
        Some(Command::Stop { socket }) => return send_ipc_command(socket, "stop"),
        None => {} // Start the daemon
    }

    // Merge config file + CLI args
    let (cfg, config_path) = config::resolve(&cli, &matches)?;

    if cli.dump_config {
        print!("{}", config::dump(&cfg));
        return Ok(());
    }

    // ── Logging ───────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    if cfg.verbose {
                        "telemuze_listen=debug".into()
                    } else {
                        "telemuze_listen=info".into()
                    }
                }),
        )
        .init();

    if let Some(path) = &config_path {
        info!(path = %path.display(), "Loaded config");
    }

    // ── Start the daemon ───────────────────────────────────────────────

    let socket_path = cfg.socket.clone();
    let paused = cfg.paused;
    let tray_enabled = cfg.tray;
    let vad_model_path = cfg.vad_model_path.clone();
    let vad_threshold = cfg.vad_threshold;
    let idle_silence = cfg.idle_silence;
    let dictation_silence = cfg.dictation_silence;
    let min_speech = cfg.min_speech;
    let max_speech = cfg.max_speech;

    // Clean up stale socket
    let _ = std::fs::remove_file(&socket_path);

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("Failed to bind socket: {}", socket_path))?;

    let mut ctx = AppContext::new(cfg);

    if let Some(dir) = &ctx.dump_audio_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create dump-audio directory: {}", dir.display()))?;
        info!(?dir, "Audio dump enabled");
    }

    // Unified event channel for audio and IPC
    let (tx, rx) = mpsc::sync_channel::<Event>(64);

    // Spawn system tray if requested
    if tray_enabled {
        let initial = if paused {
            TrayStatus::Idle
        } else {
            TrayStatus::Listening
        };
        match tray::spawn_tray(tx.clone(), initial) {
            Ok(handle) => ctx.tray_handle = Some(handle),
            Err(e) => warn!("System tray unavailable: {e}"),
        }
    }

    // Socket cleanup on exit
    let _guard = SocketGuard {
        path: &socket_path,
        notify_id: &ctx.notify_id,
        notify: ctx.notify,
        tray_handle: ctx.tray_handle.clone(),
    };

    // Spawn IPC listener thread (blocks on accept, sends events)
    spawn_ipc_listener(listener, tx.clone());

    // Spawn signal handler thread (SIGINT/SIGTERM → Stop event)
    spawn_signal_handler(tx.clone());

    // Resolve VAD model path
    let vad_path = match &vad_model_path {
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

    let vad_path_str = vad_path.to_string_lossy().into_owned();

    // Build VAD configs for each mode.
    let idle_vad_config = make_vad_config(
        &vad_path_str,
        vad_threshold,
        idle_silence,
        min_speech,
        max_speech,
    );
    let dictation_vad_config = make_vad_config(
        &vad_path_str,
        vad_threshold,
        dictation_silence,
        min_speech,
        DEFAULT_DICTATION_MAX_SPEECH,
    );

    info!(path = %vad_path.display(), "Loading VAD model");
    let mut vad = Vad::new(&idle_vad_config, &dictation_vad_config)?;

    // Set up audio capture
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input audio device available")?;

    info!(
        device = %device.name().unwrap_or_else(|_| "unknown".into()),
        "Using input device"
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
            |err| error!("Audio stream error: {err}"),
            None,
        )
        .context("Failed to build audio input stream")?;

    let mut listening = !paused;
    if listening {
        stream.play().context("Failed to start audio stream")?;
    }

    if ctx.notify && listening {
        let id = show_notification("Listening...", "Telemuze is active", 0);
        ctx.notify_id.set(id);
    }

    if listening {
        info!("Listening (use `telemuze-listen stop` to shut down)");
    } else {
        info!("Ready (paused), use `telemuze-listen toggle` to start listening");
    }
    info!(endpoint = %ctx.endpoint_url, "Server endpoint");

    // ── Main processing loop ──────────────────────────────────────────

    let mut audio_buf: Vec<f32> = Vec::new();
    let mut vad_pos: usize = 0;
    let mut was_detected = false;
    let mut mode = ListenMode::Idle;
    let mut last_speech_time: Option<Instant> = None;
    let mut utterance_audio: Vec<f32> = Vec::new();
    let mut recording_hold_until: Option<Instant> = None;
    let mut last_vad_reset = Instant::now();
    let prefill_samples = ctx.prefill_samples;

    loop {
        match rx.recv() {
            Ok(Event::Audio(chunk)) => {
                if !listening {
                    continue;
                }

                // ── Dictation mode: check silence thresholds ─────────
                if mode == ListenMode::Dictation {
                    let silence = last_speech_time
                        .map(|t| t.elapsed().as_secs_f32())
                        .unwrap_or(f32::MAX);

                    if silence > ctx.dictation_silence_secs
                        && !utterance_audio.is_empty()
                    {
                        // Speech ended → flush and return to idle so
                        // commands are accepted immediately.
                        debug!(silence, "Dictation silence, flushing and returning to idle");
                        ctx.set_tray_status(TrayStatus::Processing);
                        flush_utterance(&mut utterance_audio, &ctx, ListenMode::Idle);
                        mode = ListenMode::Idle;
                        vad.switch_to_idle();
                        audio_buf.clear();
                        vad_pos = 0;
                        was_detected = false;
                        last_vad_reset = Instant::now();
                        last_speech_time = None;
                        recording_hold_until = None;
                    }

                    // Force-flush if utterance is too long.
                    if utterance_audio.len() > ctx.dictation_max_speech_samples {
                        let secs = utterance_audio.len() as f32 / SAMPLE_RATE as f32;
                        debug!(duration = secs, "Dictation max speech, force-flushing");
                        ctx.set_tray_status(TrayStatus::RecordingProcessing);
                        flush_utterance(&mut utterance_audio, &ctx, ListenMode::Dictation);
                        // Reset so the next speech frame properly re-enters Recording.
                        was_detected = false;
                        recording_hold_until = None;
                    }
                }

                // ── Accumulate audio in dictation mode ────────────────
                // Only append when we have an utterance in progress or
                // speech is currently detected (to start a new one).
                if mode == ListenMode::Dictation
                    && (!utterance_audio.is_empty() || vad.detector().detected())
                {
                    utterance_audio.extend_from_slice(&chunk);
                }

                audio_buf.extend_from_slice(&chunk);

                // ── Process frames through VAD ────────────────────────
                while vad_pos + WINDOW_SIZE <= audio_buf.len() {
                    let frame = &audio_buf[vad_pos..vad_pos + WINDOW_SIZE];
                    vad.detector().accept_waveform(frame);
                    vad_pos += WINDOW_SIZE;

                    // Track speech transitions for UI and silence timing.
                    let now_detected = vad.detector().detected();
                    if now_detected {
                        last_speech_time = Some(Instant::now());
                        recording_hold_until = None; // cancel pending transition
                        if !was_detected {
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
                    } else if was_detected {
                        // Start debounce hold instead of immediately transitioning.
                        recording_hold_until = Some(Instant::now() + Duration::from_millis(300));
                    }
                    // Check debounce expiry.
                    if let Some(deadline) = recording_hold_until {
                        if !now_detected && Instant::now() >= deadline {
                            recording_hold_until = None;
                            let status = match mode {
                                ListenMode::Idle => TrayStatus::Listening,
                                ListenMode::Dictation => TrayStatus::Dictating,
                            };
                            ctx.set_tray_status(status);
                        }
                    }
                    was_detected = now_detected;

                    // ── Handle VAD segments ───────────────────────────
                    // Safety: always drop(seg) + pop() before any reset or
                    // mode switch to avoid use-after-free of the C segment.
                    while !vad.detector().is_empty() {
                        match mode {
                            ListenMode::Idle => {
                                let segment_data = vad.detector().front().map(|seg| {
                                    let start = seg.start() as usize;
                                    let samples = seg.samples().to_vec();
                                    (start, samples)
                                });
                                vad.detector().pop();

                                let Some((seg_start, seg_samples)) = segment_data else {
                                    continue;
                                };

                                // Prepend prefill audio.
                                let prefill_start = seg_start.saturating_sub(prefill_samples);
                                let samples = if prefill_start < seg_start && prefill_start < audio_buf.len() {
                                    let prefill_end = seg_start.min(audio_buf.len());
                                    let mut pre = audio_buf[prefill_start..prefill_end].to_vec();
                                    pre.extend_from_slice(&seg_samples);
                                    pre
                                } else {
                                    seg_samples
                                };

                                // Skip tiny segments.
                                if samples.len() < (SAMPLE_RATE as usize / 10) {
                                    continue;
                                }

                                match classify_segment(&samples, &ctx, ListenMode::Idle) {
                                    ClassifyResult::Command(text) => {
                                        execute_commands(&text, &ctx);
                                        last_speech_time = Some(Instant::now());
                                        recording_hold_until = None;
                                        // Reset VAD and drain audio so internal
                                        // buffer doesn't accumulate across commands.
                                        audio_buf.clear();
                                        vad_pos = 0;
                                        vad.detector().reset();
                                        last_vad_reset = Instant::now();
                                    }
                                    ClassifyResult::Text => {
                                        // Text detected → enter dictation mode.
                                        debug!("Switching to dictation mode");
                                        utterance_audio = samples;
                                        mode = ListenMode::Dictation;
                                        last_speech_time = Some(Instant::now());
                                        recording_hold_until = None;
                                        ctx.set_tray_status(TrayStatus::Dictating);

                                        vad.switch_to_dictation();
                                        audio_buf.clear();
                                        vad_pos = 0;
                                        was_detected = false;
                                        last_vad_reset = Instant::now();
                                        break;
                                    }
                                    ClassifyResult::Empty | ClassifyResult::Error => {}
                                }
                            }
                            ListenMode::Dictation => {
                                // Discard VAD segments in dictation mode —
                                // we use our own silence tracking.
                                vad.detector().pop();
                            }
                        }
                    }

                    // If we just switched to dictation, break the frame loop
                    // so the next chunk starts accumulating in utterance_audio.
                    if mode == ListenMode::Dictation && audio_buf.is_empty() {
                        break;
                    }
                }

                // Prevent VAD internal buffer overflow (60s limit).
                // One rule: always reset before VAD_RESET_SECS regardless of
                // mode or detection state.
                if last_vad_reset.elapsed() > Duration::from_secs(VAD_RESET_SECS) {
                    // Drain any pending segments before resetting.
                    vad.detector().flush();
                    while !vad.detector().is_empty() {
                        if mode == ListenMode::Idle {
                            if let Some(seg) = vad.detector().front() {
                                let samples = seg.samples().to_vec();
                                drop(seg);
                                vad.detector().pop();
                                if samples.len() >= (SAMPLE_RATE as usize / 10) {
                                    if let ClassifyResult::Command(text) =
                                        classify_segment(&samples, &ctx, ListenMode::Idle)
                                    {
                                        execute_commands(&text, &ctx);
                                    }
                                }
                                continue;
                            }
                        }
                        vad.detector().pop();
                    }
                    audio_buf.clear();
                    vad_pos = 0;
                    vad.detector().reset();
                    last_vad_reset = Instant::now();
                    was_detected = false;
                }
            }

            Ok(Event::Ipc(IpcCommand::Toggle)) => {
                if listening {
                    // Pause: flush any in-progress speech, stop audio.
                    drain_vad(&mut vad, &mut utterance_audio, &ctx, mode);
                    stream.pause().ok();
                    audio_buf.clear();
                    vad_pos = 0;
                    was_detected = false;
                    mode = ListenMode::Idle;
                    last_speech_time = None;
                    recording_hold_until = None;
                    vad.switch_to_idle();
                    last_vad_reset = Instant::now();
                    listening = false;
                    info!("Paused");
                    ctx.set_tray_status(TrayStatus::Idle);
                    if ctx.notify {
                        dismiss_notification(ctx.notify_id.get());
                        ctx.notify_id.set(0);
                    }
                } else {
                    // Resume
                    stream.play().ok();
                    listening = true;
                    info!("Listening");
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
                drain_vad(&mut vad, &mut utterance_audio, &ctx, mode);
                audio_buf.clear();
                vad_pos = 0;
                was_detected = false;
                recording_hold_until = None;
                vad.detector().reset();
                last_vad_reset = Instant::now();
            }

            Ok(Event::Ipc(IpcCommand::Stop)) => {
                drain_vad(&mut vad, &mut utterance_audio, &ctx, mode);
                info!("Stopping");
                break;
            }

            Err(_) => break, // All senders dropped
        }
    }

    Ok(())
}
