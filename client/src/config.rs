//! Configuration file loading and CLI/config merging for telemuze-listen.

use anyhow::{Context, Result};
use clap::ArgMatches;
use serde::Deserialize;

/// Dictation segmenting mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SegmentingMode {
    /// Silence-based flush (current behavior).
    #[default]
    Vad,
    /// Server-side decode with silence-gated output.
    Speculative,
}

impl std::fmt::Display for SegmentingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vad => write!(f, "vad"),
            Self::Speculative => write!(f, "speculative"),
        }
    }
}

impl std::str::FromStr for SegmentingMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "vad" => Ok(Self::Vad),
            "speculative" | "spec" => Ok(Self::Speculative),
            _ => anyhow::bail!("Unknown segmenting mode '{s}' — expected 'vad' or 'speculative'"),
        }
    }
}

/// Initial listening state when the client starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StartMode {
    /// Actively listening for speech and commands.
    #[default]
    Active,
    /// Sleeping — only wake phrases are processed.
    Sleeping,
    /// Fully paused — audio capture is stopped, nothing is processed.
    Paused,
}

impl std::fmt::Display for StartMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Sleeping => write!(f, "sleeping"),
            Self::Paused => write!(f, "paused"),
        }
    }
}

impl std::str::FromStr for StartMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "active" => Ok(Self::Active),
            "sleeping" | "sleep" => Ok(Self::Sleeping),
            "paused" | "pause" => Ok(Self::Paused),
            _ => anyhow::bail!("Unknown start-mode '{s}' — expected 'active', 'sleeping', or 'paused'"),
        }
    }
}
/// How transcribed text reaches the target window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMethod {
    /// Simulate keystrokes (xdotool type / wtype).
    #[default]
    Type,
    /// Paste via Ctrl+V.
    Paste,
    /// Paste via Ctrl+Shift+V (e.g. terminals).
    PasteShift,
}

impl std::fmt::Display for OutputMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Type => write!(f, "type"),
            Self::Paste => write!(f, "paste"),
            Self::PasteShift => write!(f, "paste-shift"),
        }
    }
}

impl std::str::FromStr for OutputMethod {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "type" => Ok(Self::Type),
            "paste" => Ok(Self::Paste),
            "paste-shift" | "paste_shift" => Ok(Self::PasteShift),
            _ => anyhow::bail!("Unknown output-method '{s}' — expected 'type', 'paste', or 'paste-shift'"),
        }
    }
}

/// Per-selection-buffer configuration (CLIPBOARD or PRIMARY).
#[derive(Debug, Clone, Copy)]
pub struct SelectionConfig {
    /// Write dictation text to this selection (even if not used for paste trigger).
    pub populate: bool,
    /// Restore the original content after pasting.
    pub restore: bool,
}

/// TOML-level per-selection config (all fields optional).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct FileSelectionConfig {
    pub populate: Option<bool>,
    pub restore: Option<bool>,
}

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::Cli;

// ── TOML file structs ────────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "kebab-case", default)]
pub struct FileConfig {
    pub url: Option<String>,
    pub vad_model_path: Option<String>,
    pub smart: Option<bool>,
    pub type_text: Option<bool>,
    pub tray: Option<bool>,
    pub notify: Option<bool>,
    pub sound: Option<bool>,
    pub display_server: Option<String>,
    pub socket: Option<String>,
    pub verbose: Option<bool>,
    pub vad_threshold: Option<f32>,
    pub vad_energy_gate: Option<f32>,
    pub fast_silence: Option<f32>,
    pub slow_silence: Option<f32>,
    pub segmenting: Option<String>,
    pub min_speech: Option<f32>,
    pub max_speech: Option<f32>,
    pub dictation_max_speech: Option<f32>,
    pub prefill_ms: Option<u32>,
    pub hotwords: Option<String>,
    pub dictation_hotwords_score: Option<f32>,
    pub sentence_continuation: Option<bool>,
    pub sentence_continuation_timeout: Option<f32>,
    pub min_dictation_words: Option<usize>,
    pub paused: Option<bool>,
    pub start_mode: Option<String>,
    pub scroll_ticks: Option<u32>,
    pub dump_audio: Option<PathBuf>,

    // ── Text output method ──────────────────────────────────────────────
    pub output_method: Option<String>,

    #[serde(default)]
    pub clipboard: FileSelectionConfig,
    #[serde(default)]
    pub primary: FileSelectionConfig,

    /// Per-application output method overrides keyed by window class name.
    #[serde(default)]
    pub output_rules: HashMap<String, String>,

    // ── Local command detection (110m transducer) ─────────────────────────
    pub no_cmd: Option<bool>,
    pub cmd_model_dir: Option<PathBuf>,
    pub cmd_boost_first: Option<f32>,
    pub cmd_boost_phrase: Option<f32>,
    pub cmd_boost_vocab: Option<f32>,
    pub cmd_first_pass_ms: Option<u32>,
    pub cmd_prefill_ms: Option<u32>,
    pub cmd_silence_ms: Option<u32>,
    pub cmd_spec_silence_ms: Option<u32>,
    pub cmd_threads: Option<i32>,
    pub cmd_beam_width: Option<i32>,

    // ── End-of-utterance model ───────────────────────────────────────────
    pub no_eou: Option<bool>,
    pub eou_model_dir: Option<PathBuf>,
    pub eou_threads: Option<i32>,
    pub eou_blank_penalty: Option<f32>,
    pub eou_no_int8: Option<bool>,

    #[serde(default)]
    pub aliases: AliasConfig,

    #[serde(default)]
    pub modifiers: HashMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "kebab-case", default)]
pub struct AliasConfig {
    pub click: Option<Vec<String>>,
    pub press: Option<Vec<String>>,
    pub scroll: Option<Vec<String>>,
    pub slash_command: Option<Vec<String>>,
    pub undo: Option<Vec<String>>,
    pub wake: Option<Vec<String>>,
    pub sleep: Option<Vec<String>>,
}

// ── Resolved (merged) config ─────────────────────────────────────────────

/// Final configuration after merging config file + CLI args.
pub struct ResolvedConfig {
    pub url: String,
    pub vad_model_path: Option<String>,
    pub smart: bool,
    pub type_text: bool,
    pub tray: bool,
    pub notify: bool,
    pub sound: bool,
    pub display_server: Option<String>,
    pub socket: String,
    pub verbose: bool,
    pub vad_threshold: f32,
    pub vad_energy_gate: f32,
    pub fast_silence: f32,
    pub slow_silence: f32,
    pub segmenting: SegmentingMode,
    pub min_speech: f32,
    pub max_speech: f32,
    pub dictation_max_speech: f32,
    pub prefill_ms: u32,
    pub hotwords: Option<String>,
    pub dictation_hotwords_score: f32,
    pub sentence_continuation: bool,
    pub sentence_continuation_timeout: f32,
    pub min_dictation_words: usize,
    pub start_mode: StartMode,
    pub scroll_ticks: u32,
    pub dump_audio: Option<PathBuf>,
    pub output_method: OutputMethod,
    pub clipboard: SelectionConfig,
    pub primary: SelectionConfig,
    pub output_rules: HashMap<String, OutputMethod>,
    pub no_cmd: bool,
    pub cmd_model_dir: Option<PathBuf>,
    pub cmd_boost_first: f32,
    pub cmd_boost_phrase: f32,
    pub cmd_boost_vocab: f32,
    pub cmd_first_pass_ms: u32,
    pub cmd_prefill_ms: u32,
    pub cmd_silence_ms: u32,
    pub cmd_spec_silence_ms: u32,
    pub cmd_threads: i32,
    pub cmd_beam_width: i32,
    pub no_eou: bool,
    pub eou_model_dir: Option<PathBuf>,
    pub eou_threads: i32,
    pub eou_blank_penalty: f32,
    pub eou_no_int8: bool,
    pub aliases: ResolvedAliases,
    pub modifiers: Vec<(String, String)>,
}

/// Resolved voice command trigger aliases.
pub struct ResolvedAliases {
    pub click: Vec<String>,
    pub press: Vec<String>,
    pub scroll: Vec<String>,
    /// Each entry is a multi-word phrase split into individual words.
    pub slash_command: Vec<Vec<String>>,
    pub undo: Vec<String>,
    /// Phrases that wake the client from sleep mode.
    pub wake: Vec<String>,
    /// Phrases that put the client to sleep (stop listening until woken).
    pub sleep: Vec<String>,
}

// ── Default aliases (match the previously hardcoded consts) ──────────────

fn default_click() -> Vec<String> {
    vec!["click".into()]
}

fn default_press() -> Vec<String> {
    vec!["press".into()]
}

fn default_scroll() -> Vec<String> {
    vec!["scroll".into()]
}

fn default_slash_command_phrases() -> Vec<String> {
    vec![
        "slash command".into(),
        "flash command".into(),
        "splash command".into(),
    ]
}

fn default_undo() -> Vec<String> {
    vec!["undo".into()]
}

fn default_wake() -> Vec<String> {
    vec!["okay computer".into()]
}

fn default_sleep() -> Vec<String> {
    vec!["goodbye computer".into()]
}

fn default_modifiers() -> Vec<(String, String)> {
    vec![
        ("control".into(), "ctrl".into()),
        ("ctrl".into(), "ctrl".into()),
        ("shift".into(), "shift".into()),
        ("alt".into(), "alt".into()),
        ("super".into(), "super".into()),
        ("command".into(), "super".into()),
        ("meta".into(), "super".into()),
    ]
}

// ── Config loading ───────────────────────────────────────────────────────

/// Load config from the given explicit path, or fall back to the XDG default.
pub fn load_config(explicit_path: Option<&Path>) -> Result<(FileConfig, Option<PathBuf>)> {
    let path = match explicit_path {
        Some(p) => {
            if !p.exists() {
                anyhow::bail!("Config file not found: {}", p.display());
            }
            p.to_owned()
        }
        None => {
            let default = dirs_next::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("telemuze/listen.toml");
            if !default.exists() {
                return Ok((FileConfig::default(), None));
            }
            default
        }
    };

    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;
    let config: FileConfig = toml::from_str(&contents)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
    Ok((config, Some(path)))
}

// ── Dump config ──────────────────────────────────────────────────────────

/// Format a string list as a TOML inline array, e.g. `["a", "b", "c"]`.
fn toml_string_list(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("\"{s}\"")).collect();
    format!("[{}]", inner.join(", "))
}

/// Generate a complete commented TOML config from resolved values.
pub fn dump(cfg: &ResolvedConfig) -> String {
    let mut out = String::new();

    // Helper closures
    let mut line = |s: &str| {
        out.push_str(s);
        out.push('\n');
    };

    line("# ── Server ────────────────────────────────────────────────────────────────");
    line("# Telemuze server URL.");
    line(&format!("url = \"{}\"", cfg.url));
    line("");

    line("# ── Behavior ──────────────────────────────────────────────────────────────");
    line("# Use the /v1/dictate/smart endpoint (LLM post-processing, slower but cleaner).");
    line("# Options: true | false");
    line(&format!("smart = {}", cfg.smart));
    line("");

    line("# Type transcribed text into the focused window instead of printing to stdout.");
    line("# Options: true | false");
    line(&format!("type-text = {}", cfg.type_text));
    line("");

    line("# Default method for delivering text to the target window.");
    line("# Options: \"type\" (keystrokes) | \"paste\" (Ctrl+V) | \"paste-shift\" (Ctrl+Shift+V)");
    line("# Per-application overrides can be set in [output-rules] below.");
    line(&format!("output-method = \"{}\"", cfg.output_method));
    line("");

    line("# Show a system tray icon indicating recording/processing state.");
    line("# Uses XEmbed on X11 and StatusNotifierItem on Wayland.");
    line("# Options: true | false");
    line(&format!("tray = {}", cfg.tray));
    line("");

    line("# Show desktop notifications for recording/processing state.");
    line("# Options: true | false");
    line(&format!("notify = {}", cfg.notify));
    line("");

    line("# Play notification sounds on transcription success/error.");
    line("# Options: true | false");
    line(&format!("sound = {}", cfg.sound));
    line("");

    line("# Display server for text injection. Auto-detected if omitted.");
    line("# Options: \"wayland\" | \"x11\"");
    match &cfg.display_server {
        Some(ds) => line(&format!("display-server = \"{ds}\"")),
        None => line("# display-server = \"wayland\""),
    }
    line("");

    line("# Unix socket path for IPC between daemon and controller subcommands.");
    line(&format!("socket = \"{}\"", cfg.socket));
    line("");

    line("# Enable verbose logging (parsed actions, key injection details).");
    line("# Options: true | false");
    line(&format!("verbose = {}", cfg.verbose));
    line("");

    line("# Initial state when starting up.");
    line("# Options: \"active\" (listening), \"sleeping\" (wake words only), \"paused\" (fully off)");
    line(&format!("start-mode = \"{}\"", cfg.start_mode));
    line("");

    line("# ── VAD tuning ────────────────────────────────────────────────────────────");
    line("# Path to Silero VAD ONNX model file.");
    line("# Auto-downloads to ~/.local/share/telemuze/models/silero_vad.onnx if omitted.");
    match &cfg.vad_model_path {
        Some(p) => line(&format!("vad-model-path = \"{p}\"")),
        None => line("# vad-model-path = \"/path/to/silero_vad.onnx\""),
    }
    line("");

    line("# Speech detection threshold. Higher = harder to trigger.");
    line("# Range: 0.0–1.0");
    line(&format!("vad-threshold = {}", cfg.vad_threshold));
    line("");

    line("# Energy gate threshold (0.0–1.0). Frames quieter than this skip VAD inference,");
    line("# reducing idle CPU usage. 0 disables the gate. Higher = more aggressive gating.");
    line(&format!("vad-energy-gate = {}", cfg.vad_energy_gate));
    line("");

    line("# Fast silence threshold (seconds). VAD min silence duration — controls");
    line("# idle-mode command detection timing. In speculative dictation: triggers a decode.");
    line(&format!("fast-silence = {}", cfg.fast_silence));
    line("");

    line("# Slow silence threshold (seconds). In vad mode: flushes dictation.");
    line("# In speculative mode: emits cached text unconditionally after this silence.");
    line(&format!("slow-silence = {}", cfg.slow_silence));
    line("");

    line("# Dictation segmenting mode.");
    line("# Options: \"vad\" (silence-based flush) | \"speculative\" (server-side decode with silence-gated output)");
    line(&format!("segmenting = \"{}\"", cfg.segmenting));
    line("");

    line("# Minimum speech duration (seconds) before a segment is considered valid.");
    line(&format!("min-speech = {}", cfg.min_speech));
    line("");

    line("# Maximum VAD segment length (seconds) before force-flushing. Controls idle-mode");
    line("# command classification segment size; dictation uses dictation-max-speech instead.");
    line(&format!("max-speech = {}", cfg.max_speech));
    line("");

    line("# Maximum dictation utterance length (seconds) before force-flushing.");
    line(&format!("dictation-max-speech = {}", cfg.dictation_max_speech));
    line("");

    line("# Audio to include before speech onset, in milliseconds (captures breaths, soft starts).");
    line(&format!("prefill-ms = {}", cfg.prefill_ms));
    line("");

    line("# ── Hotwords ──────────────────────────────────────────────────────────────");
    line("# Comma-separated list of words to boost during dictation recognition.");
    match &cfg.hotwords {
        Some(hw) => line(&format!("hotwords = \"{hw}\"")),
        None => line("# hotwords = \"custom,words\""),
    }
    line("");

    line("# Score boost for user-provided dictation hotwords. Set 0.0 to disable.");
    line("# Range: 0.0 | 1.0–4.0");
    line(&format!("dictation-hotwords-score = {}", cfg.dictation_hotwords_score));
    line("");

    line("# ── Text processing ──────────────────────────────────────────────────────");

    line("# Enable sentence continuation: lowercase across segments and merge");
    line("# punctuation when the ASR signals a continuation.");
    line("# Options: true | false");
    line(&format!("sentence-continuation = {}", cfg.sentence_continuation));
    line("");

    line("# Seconds after last output before sentence continuation resets");
    line("# (treats the next segment as a fresh utterance).");
    line(&format!("sentence-continuation-timeout = {}", cfg.sentence_continuation_timeout));
    line("");

    line("# Minimum number of words for a dictation output to be kept.");
    line("# Shorter outputs are silently dropped. Set to 1 to keep everything.");
    line(&format!("min-dictation-words = {}", cfg.min_dictation_words));
    line("");

    line("# Number of scroll ticks per \"scroll up\" / \"scroll down\" voice command.");
    line(&format!("scroll-ticks = {}", cfg.scroll_ticks));
    line("");

    line("# ── Audio debugging ──────────────────────────────────────────────────────");
    line("# Dump each audio segment to a numbered WAV file in this directory.");
    match &cfg.dump_audio {
        Some(p) => line(&format!("dump-audio = \"{}\"", p.display())),
        None => line("# dump-audio = \"/tmp/telemuze-dumps\""),
    }
    line("");

    line("# ── Local command detection (110m transducer) ────────────────────────────");
    line("# Disable local command detection (fall back to server-side only).");
    line("# Options: true | false");
    line(&format!("no-cmd = {}", cfg.no_cmd));
    line("");

    line("# Path to Parakeet-TDT 110m model directory. Auto-downloads if omitted.");
    match &cfg.cmd_model_dir {
        Some(p) => line(&format!("cmd-model-dir = \"{}\"", p.display())),
        None => line("# cmd-model-dir = \"/path/to/parakeet-tdt-110m\""),
    }
    line("");

    line("# Hotword boost for trigger verbs in pass 1 (press, click, scroll, ...).");
    line("# Range: 1.0–5.0");
    line(&format!("cmd-boost-first = {}", cfg.cmd_boost_first));
    line("");

    line("# Boost for multi-word command phrases in pass 2 (e.g. \"press control\", \"scroll up\").");
    line(&format!("cmd-boost-phrase = {}", cfg.cmd_boost_phrase));
    line("");

    line("# Boost for supporting vocabulary in pass 2 (key names, modifiers, directions).");
    line(&format!("cmd-boost-vocab = {}", cfg.cmd_boost_vocab));
    line("");

    line("# Audio from onset for pass 1 decode (ms).");
    line(&format!("cmd-first-pass-ms = {}", cfg.cmd_first_pass_ms));
    line("");

    line("# Audio lookback before VAD trigger for command pipeline (ms).");
    line(&format!("cmd-prefill-ms = {}", cfg.cmd_prefill_ms));
    line("");

    line("# Silence duration (ms) after speech ends to trigger command decode.");
    line(&format!("cmd-silence-ms = {}", cfg.cmd_silence_ms));
    line("");

    line("# Speculative silence (ms) for early command decode; 0 disables.");
    line(&format!("cmd-spec-silence-ms = {}", cfg.cmd_spec_silence_ms));
    line("");

    line("# Number of threads for the command recognizer.");
    line(&format!("cmd-threads = {}", cfg.cmd_threads));
    line("");

    line("# Beam search width (max_active_paths). Higher = more accurate but slower.");
    line(&format!("cmd-beam-width = {}", cfg.cmd_beam_width));
    line("");

    line("# ── End-of-utterance model ─────────────────────────────────────────────");
    line("# Disable the streaming EOU model (semantic segmentation of dictation).");
    line("# Options: true | false");
    line(&format!("no-eou = {}", cfg.no_eou));
    line("");

    line("# Path to EOU model directory. Model must be pre-installed (not auto-downloaded).");
    match &cfg.eou_model_dir {
        Some(p) => line(&format!("eou-model-dir = \"{}\"", p.display())),
        None => line("# eou-model-dir = \"/path/to/parakeet-realtime-eou-120m-v1\""),
    }
    line("");

    line("# Number of threads for the EOU model.");
    line(&format!("eou-threads = {}", cfg.eou_threads));
    line("");

    line("# Blank token penalty (positive = fewer blanks = more token emissions).");
    line("# Range: 0.0–5.0");
    line(&format!("eou-blank-penalty = {}", cfg.eou_blank_penalty));
    line("");

    line("# Use full-precision EOU models instead of int8 quantized.");
    line("# Options: true | false");
    line(&format!("eou-no-int8 = {}", cfg.eou_no_int8));
    line("");

    line("# ── Selection buffer settings ─────────────────────────────────────────");
    line("# Each selection (clipboard, primary) can independently populate with dictation");
    line("# text and optionally restore original content after pasting.");
    line("# The selection required by the active paste method is always populated regardless.");
    line("[clipboard]");
    line(&format!("populate = {}", cfg.clipboard.populate));
    line(&format!("restore = {}", cfg.clipboard.restore));
    line("");
    line("[primary]");
    line(&format!("populate = {}", cfg.primary.populate));
    line(&format!("restore = {}", cfg.primary.restore));
    line("");

    line("# ── Per-application output method overrides ─────────────────────────────");
    line("# Map window class names to output methods. Falls back to the global output-method.");
    line("# To find a window's class name, run: xdotool getactivewindow getwindowclassname");
    line("# Keys are case-sensitive and must match the output exactly.");
    line("[output-rules]");
    if cfg.output_rules.is_empty() {
        line("# TelegramDesktop = \"paste\"");
        line("# kitty = \"paste-shift\"");
    } else {
        for (class, method) in &cfg.output_rules {
            line(&format!("{class} = \"{method}\""));
        }
    }
    line("");

    line("# ── Voice command trigger aliases ────────────────────────────────────────");
    line("# Each key maps a command type to alternate spoken words that trigger it.");
    line("# The STT model sometimes mishears these words, so add common misrecognitions.");
    line("[aliases]");
    line(&format!("click = {}", toml_string_list(&cfg.aliases.click)));
    line(&format!("press = {}", toml_string_list(&cfg.aliases.press)));
    line(&format!("scroll = {}", toml_string_list(&cfg.aliases.scroll)));
    line("# Slash command triggers are multi-word phrases (matched as complete phrases).");
    let slash_phrases: Vec<String> = cfg.aliases.slash_command.iter()
        .map(|words| words.join(" "))
        .collect();
    line(&format!("slash-command = {}", toml_string_list(&slash_phrases)));
    line("# Word(s) that trigger undo (erases last typed segment via backspace).");
    line(&format!("undo = {}", toml_string_list(&cfg.aliases.undo)));
    line("# Phrase(s) that wake the client from sleep mode.");
    line(&format!("wake = {}", toml_string_list(&cfg.aliases.wake)));
    line("# Word(s) that put the client to sleep (stops listening until woken).");
    line(&format!("sleep = {}", toml_string_list(&cfg.aliases.sleep)));
    line("");

    line("# ── Modifier key mappings ────────────────────────────────────────────────");
    line("# Maps spoken words to canonical modifier names used by xdotool/wtype.");
    line("# Canonical names: \"ctrl\", \"shift\", \"alt\", \"super\"");
    line("[modifiers]");
    for (alias, canonical) in &cfg.modifiers {
        if alias == "command" {
            line(&format!("{alias} = \"{canonical}\"   # macOS-style alias"));
        } else {
            line(&format!("{alias} = \"{canonical}\""));
        }
    }

    out
}

// ── Merging CLI + config ─────────────────────────────────────────────────

/// Returns true when clap reports the value came from the command line or an
/// environment variable (i.e. the user explicitly set it).
fn is_explicit(matches: &ArgMatches, id: &str) -> bool {
    // clap derive uses the field name (underscores) as the arg ID internally,
    // even though the CLI flag uses hyphens. value_source panics on unknown IDs,
    // so we convert hyphens to underscores to match the derive convention.
    let id = id.replace('-', "_");
    matches
        .value_source(&id)
        .is_some_and(|s| matches!(s, clap::parser::ValueSource::CommandLine | clap::parser::ValueSource::EnvVariable))
}

/// Merge CLI args + config file into a single `ResolvedConfig`.
///
/// Precedence: CLI arg (if explicitly provided) > config file > clap default.
pub fn resolve(cli: &Cli, matches: &ArgMatches) -> Result<(ResolvedConfig, Option<PathBuf>)> {
    let (file, config_path) = load_config(cli.config.as_deref())?;

    // Macro for scalar fields: if CLI was explicit, use cli value; else try
    // config file; else fall back to the clap default (which is in cli).
    macro_rules! r {
        ($field:ident, $id:expr) => {
            if is_explicit(matches, $id) {
                cli.$field.clone()
            } else {
                file.$field.unwrap_or_else(|| cli.$field.clone())
            }
        };
    }

    // Boolean flags: clap SetTrue action means default is false.  If the
    // config sets it to true and the user didn't pass the flag, use config.
    macro_rules! r_bool {
        ($field:ident, $id:expr) => {
            if is_explicit(matches, $id) {
                cli.$field
            } else {
                file.$field.unwrap_or(cli.$field)
            }
        };
    }

    // Option<T> fields: CLI explicit wins, then config, then None.
    macro_rules! r_opt {
        ($field:ident, $id:expr) => {
            if is_explicit(matches, $id) {
                cli.$field.clone()
            } else {
                file.$field.or_else(|| cli.$field.clone())
            }
        };
    }

    // Resolve aliases
    let aliases = ResolvedAliases {
        click: file.aliases.click.unwrap_or_else(default_click),
        press: file.aliases.press.unwrap_or_else(default_press),
        scroll: file.aliases.scroll.unwrap_or_else(default_scroll),
        slash_command: file
            .aliases
            .slash_command
            .unwrap_or_else(default_slash_command_phrases)
            .into_iter()
            .map(|phrase| phrase.split_whitespace().map(String::from).collect())
            .collect(),
        undo: file.aliases.undo.unwrap_or_else(default_undo),
        wake: file.aliases.wake.unwrap_or_else(default_wake),
        sleep: file.aliases.sleep.unwrap_or_else(default_sleep),
    };

    // Resolve modifiers
    let modifiers = if file.modifiers.is_empty() {
        default_modifiers()
    } else {
        file.modifiers.into_iter().collect()
    };

    Ok((ResolvedConfig {
        url: r!(url, "url"),
        vad_model_path: r_opt!(vad_model_path, "vad-model-path"),
        smart: r_bool!(smart, "smart"),
        type_text: r_bool!(type_text, "type-text"),
        tray: r_bool!(tray, "tray"),
        notify: r_bool!(notify, "notify"),
        sound: r_bool!(sound, "sound"),
        display_server: r_opt!(display_server, "display-server"),
        socket: r!(socket, "socket"),
        verbose: r_bool!(verbose, "verbose"),
        vad_threshold: r!(vad_threshold, "vad-threshold"),
        vad_energy_gate: r!(vad_energy_gate, "vad-energy-gate"),
        fast_silence: r!(fast_silence, "fast-silence"),
        slow_silence: r!(slow_silence, "slow-silence"),
        segmenting: if is_explicit(matches, "segmenting") {
            cli.segmenting
        } else if let Some(ref s) = file.segmenting {
            s.parse().context("Invalid segmenting mode in config file")?
        } else {
            cli.segmenting
        },
        min_speech: r!(min_speech, "min-speech"),
        max_speech: r!(max_speech, "max-speech"),
        dictation_max_speech: r!(dictation_max_speech, "dictation-max-speech"),
        prefill_ms: r!(prefill_ms, "prefill-ms"),
        hotwords: r_opt!(hotwords, "hotwords"),
        dictation_hotwords_score: r!(dictation_hotwords_score, "dictation-hotwords-score"),
        sentence_continuation: r_bool!(sentence_continuation, "sentence-continuation"),
        sentence_continuation_timeout: r!(sentence_continuation_timeout, "sentence-continuation-timeout"),
        min_dictation_words: r!(min_dictation_words, "min-dictation-words"),
        start_mode: if is_explicit(matches, "start-mode") {
            cli.start_mode
        } else if let Some(ref s) = file.start_mode {
            s.parse().context("Invalid start-mode in config file")?
        } else if file.paused == Some(true) {
            StartMode::Paused
        } else {
            cli.start_mode
        },
        scroll_ticks: r!(scroll_ticks, "scroll-ticks"),
        dump_audio: r_opt!(dump_audio, "dump-audio"),
        output_method: if is_explicit(matches, "output-method") {
            cli.output_method
        } else if let Some(ref s) = file.output_method {
            s.parse().context("Invalid output-method in config file")?
        } else {
            cli.output_method
        },
        clipboard: SelectionConfig {
            populate: file.clipboard.populate.unwrap_or(false),
            restore: file.clipboard.restore.unwrap_or(false),
        },
        primary: SelectionConfig {
            populate: file.primary.populate.unwrap_or(false),
            restore: file.primary.restore.unwrap_or(false),
        },
        output_rules: {
            let mut rules = HashMap::new();
            for (class, method_str) in &file.output_rules {
                let method: OutputMethod = method_str.parse()
                    .with_context(|| format!("Invalid output-method '{method_str}' for window class '{class}' in [output-rules]"))?;
                rules.insert(class.clone(), method);
            }
            rules
        },
        no_cmd: r_bool!(no_cmd, "no-cmd"),
        cmd_model_dir: r_opt!(cmd_model_dir, "cmd-model-dir"),
        cmd_boost_first: r!(cmd_boost_first, "cmd-boost-first"),
        cmd_boost_phrase: r!(cmd_boost_phrase, "cmd-boost-phrase"),
        cmd_boost_vocab: r!(cmd_boost_vocab, "cmd-boost-vocab"),
        cmd_first_pass_ms: r!(cmd_first_pass_ms, "cmd-first-pass-ms"),
        cmd_prefill_ms: r!(cmd_prefill_ms, "cmd-prefill-ms"),
        cmd_silence_ms: r!(cmd_silence_ms, "cmd-silence-ms"),
        cmd_spec_silence_ms: r!(cmd_spec_silence_ms, "cmd-spec-silence-ms"),
        cmd_threads: r!(cmd_threads, "cmd-threads"),
        cmd_beam_width: r!(cmd_beam_width, "cmd-beam-width"),
        no_eou: r_bool!(no_eou, "no-eou"),
        eou_model_dir: r_opt!(eou_model_dir, "eou-model-dir"),
        eou_threads: r!(eou_threads, "eou-threads"),
        eou_blank_penalty: r!(eou_blank_penalty, "eou-blank-penalty"),
        eou_no_int8: r_bool!(eou_no_int8, "eou-no-int8"),
        aliases,
        modifiers,
    }, config_path))
}
