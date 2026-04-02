//! Configuration file loading and CLI/config merging for telemuze-listen.

use anyhow::{Context, Result};
use clap::ArgMatches;
use serde::Deserialize;
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
    pub idle_silence: Option<f32>,
    pub dictation_silence: Option<f32>,
    pub min_speech: Option<f32>,
    pub max_speech: Option<f32>,
    pub dictation_max_speech: Option<f32>,
    pub prefill_ms: Option<u32>,
    pub hotwords: Option<String>,
    pub command_hotwords_score: Option<f32>,
    pub dictation_hotwords_score: Option<f32>,
    pub continuation_lowercase: Option<bool>,
    pub lowercase_timeout: Option<f32>,
    pub min_dictation_words: Option<usize>,
    pub paused: Option<bool>,
    pub scroll_ticks: Option<u32>,
    pub dump_audio: Option<PathBuf>,

    // ── Keyword spotting ──────────────────────────────────────────────────
    pub no_kws: Option<bool>,
    pub kws_model: Option<String>,
    pub kws_model_dir: Option<PathBuf>,
    pub kws_threshold: Option<f32>,
    pub kws_sleep_threshold: Option<f32>,
    pub kws_score: Option<f32>,
    pub kws_threads: Option<i32>,
    pub kws_timeout: Option<f32>,

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
    pub idle_silence: f32,
    pub dictation_silence: f32,
    pub min_speech: f32,
    pub max_speech: f32,
    pub dictation_max_speech: f32,
    pub prefill_ms: u32,
    pub hotwords: Option<String>,
    pub command_hotwords_score: f32,
    pub dictation_hotwords_score: f32,
    pub continuation_lowercase: bool,
    pub lowercase_timeout: f32,
    pub min_dictation_words: usize,
    pub paused: bool,
    pub scroll_ticks: u32,
    pub dump_audio: Option<PathBuf>,
    pub no_kws: bool,
    pub kws_model: crate::kws::KwsModel,
    pub kws_model_dir: Option<PathBuf>,
    pub kws_threshold: f32,
    pub kws_sleep_threshold: f32,
    pub kws_score: f32,
    pub kws_threads: i32,
    pub kws_timeout: f32,
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
    vec!["mouse".into(), "click".into(), "look".into(), "lick".into()]
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
    vec!["wake up".into()]
}

fn default_sleep() -> Vec<String> {
    vec!["sleep".into()]
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

    line("# Show a system tray icon indicating recording/processing state (X11 only).");
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

    line("# Start in paused state (model loaded but not listening). Use `toggle` to begin.");
    line("# Options: true | false");
    line(&format!("paused = {}", cfg.paused));
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

    line("# Silence duration (seconds) to end a segment in idle/command mode.");
    line(&format!("idle-silence = {}", cfg.idle_silence));
    line("");

    line("# Silence duration (seconds) to end a segment in dictation mode.");
    line(&format!("dictation-silence = {}", cfg.dictation_silence));
    line("");

    line("# Minimum speech duration (seconds) before a segment is considered valid.");
    line(&format!("min-speech = {}", cfg.min_speech));
    line("");

    line("# Maximum speech segment length (seconds) in idle mode before force-flushing.");
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

    line("# Score boost for built-in command hotwords in idle mode. Set 0.0 to disable.");
    line("# Range: 0.0 | 1.0–4.0");
    line(&format!("command-hotwords-score = {}", cfg.command_hotwords_score));
    line("");

    line("# Score boost for user-provided dictation hotwords. Set 0.0 to disable.");
    line("# Range: 0.0 | 1.0–4.0");
    line(&format!("dictation-hotwords-score = {}", cfg.dictation_hotwords_score));
    line("");

    line("# ── Text processing ──────────────────────────────────────────────────────");
    line("# Lowercase the first letter of a segment when the previous segment did not");
    line("# end with sentence-ending punctuation (. ! ?). Only applies in idle mode.");
    line("# Options: true | false");
    line(&format!("continuation-lowercase = {}", cfg.continuation_lowercase));
    line("");

    line("# Seconds after last output before continuation lowercasing resets");
    line("# (treats the next segment as a fresh utterance).");
    line(&format!("lowercase-timeout = {}", cfg.lowercase_timeout));
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

    line("# ── Keyword spotting ─────────────────────────────────────────────────────");
    line("# Disable keyword spotting and fall back to server-side command classification.");
    line("# Options: true | false");
    line(&format!("no-kws = {}", cfg.no_kws));
    line("");

    line("# KWS model variant.");
    line("# Options: \"gigaspeech\" (English, BPE) | \"zh-en\" (Chinese+English, phone)");
    line(&format!("kws-model = \"{}\"", cfg.kws_model));
    line("");

    line("# Path to KWS model directory. Auto-downloads if omitted.");
    match &cfg.kws_model_dir {
        Some(p) => line(&format!("kws-model-dir = \"{}\"", p.display())),
        None => line("# kws-model-dir = \"/path/to/kws-model\""),
    }
    line("");

    line("# Keyword detection threshold (lower = more sensitive, higher = fewer false positives).");
    line("# Range: 0.0–1.0");
    line(&format!("kws-threshold = {}", cfg.kws_threshold));
    line("");

    line("# Keyword detection threshold used in sleep mode (for wake-word detection).");
    line("# Higher than kws-threshold to reduce false wake-ups. Range: 0.0–1.0");
    line(&format!("kws-sleep-threshold = {}", cfg.kws_sleep_threshold));
    line("");

    line("# Keyword boost score.");
    line(&format!("kws-score = {}", cfg.kws_score));
    line("");

    line("# Number of threads for the KWS model.");
    line(&format!("kws-threads = {}", cfg.kws_threads));
    line("");

    line("# Seconds to wait for keyword detection after VAD speech onset");
    line("# before falling through to dictation mode.");
    line(&format!("kws-timeout = {}", cfg.kws_timeout));
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
        idle_silence: r!(idle_silence, "idle-silence"),
        dictation_silence: r!(dictation_silence, "dictation-silence"),
        min_speech: r!(min_speech, "min-speech"),
        max_speech: r!(max_speech, "max-speech"),
        dictation_max_speech: r!(dictation_max_speech, "dictation-max-speech"),
        prefill_ms: r!(prefill_ms, "prefill-ms"),
        hotwords: r_opt!(hotwords, "hotwords"),
        command_hotwords_score: r!(command_hotwords_score, "command-hotwords-score"),
        dictation_hotwords_score: r!(dictation_hotwords_score, "dictation-hotwords-score"),
        continuation_lowercase: r_bool!(continuation_lowercase, "continuation-lowercase"),
        lowercase_timeout: r!(lowercase_timeout, "lowercase-timeout"),
        min_dictation_words: r!(min_dictation_words, "min-dictation-words"),
        paused: r_bool!(paused, "paused"),
        scroll_ticks: r!(scroll_ticks, "scroll-ticks"),
        dump_audio: r_opt!(dump_audio, "dump-audio"),
        no_kws: r_bool!(no_kws, "no-kws"),
        kws_model: if is_explicit(matches, "kws-model") {
            cli.kws_model
        } else if let Some(ref s) = file.kws_model {
            s.parse().context("Invalid kws-model in config file")?
        } else {
            cli.kws_model
        },
        kws_model_dir: r_opt!(kws_model_dir, "kws-model-dir"),
        kws_threshold: r!(kws_threshold, "kws-threshold"),
        kws_sleep_threshold: r!(kws_sleep_threshold, "kws-sleep-threshold"),
        kws_score: r!(kws_score, "kws-score"),
        kws_threads: r!(kws_threads, "kws-threads"),
        kws_timeout: r!(kws_timeout, "kws-timeout"),
        aliases,
        modifiers,
    }, config_path))
}
