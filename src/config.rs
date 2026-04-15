use anyhow::{Context, Result};
use clap::{ArgMatches, Parser};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Telemuze: Self-hosted AI dictation and transcription server.
#[derive(Parser, Debug, Clone)]
#[command(name = "telemuze", version)]
pub struct Config {
    /// Path to TOML configuration file.
    /// Defaults to ~/.config/telemuze/server.toml when that file exists.
    #[arg(long, env = "TELEMUZE_CONFIG")]
    pub config: Option<PathBuf>,

    /// Print the full resolved configuration as TOML (with comments) and exit.
    /// Optionally accepts an output file path; if omitted, prints to stdout.
    #[arg(long, num_args = 0..=1, value_name = "FILE")]
    pub dump_config: Option<Option<PathBuf>>,

    /// Update the config file in place with the full resolved configuration and exit.
    /// Writes back to the file specified by --config (or the default location).
    #[arg(long)]
    pub update_config: bool,

    /// Host address to bind to. When unset, the server binds both
    /// IPv4 (0.0.0.0) and IPv6 ([::]) on the configured port.
    #[arg(long, env = "TELEMUZE_HOST")]
    pub host: Option<String>,

    /// Port to listen on
    #[arg(long, default_value_t = 7313, env = "TELEMUZE_PORT")]
    pub port: u16,

    /// Path to the Parakeet ONNX model directory.
    /// If omitted, models are auto-downloaded to --models-dir.
    #[arg(long, env = "TELEMUZE_STT_MODEL_PATH")]
    pub stt_model_path: Option<PathBuf>,

    /// STT model to auto-download when --stt-model-path is unset.
    /// Options: "v2" (parakeet-tdt-0.6b-v2), "v3" (parakeet-tdt-0.6b-v3),
    /// "unified" (parakeet-unified-en-0.6b — default).
    #[arg(long, env = "TELEMUZE_STT_MODEL", default_value = "unified",
           value_parser = clap::builder::PossibleValuesParser::new(["v2", "v3", "unified"]))]
    pub stt_model: String,

    /// Path to the Silero VAD ONNX model file.
    /// If omitted, models are auto-downloaded to --models-dir.
    #[arg(long, env = "TELEMUZE_VAD_MODEL_PATH")]
    pub vad_model_path: Option<PathBuf>,

    /// Path to the NVIDIA Sortformer v2.1 ONNX model file (4-speaker max).
    /// If omitted, looks for diar_streaming_sortformer_4spk-v2.1.onnx in
    /// --models-dir. Diarization is performed by spawning the
    /// `telemuze-diarize` subprocess once per long-form request.
    #[arg(long, env = "TELEMUZE_DIARIZATION_MODEL_PATH")]
    pub diarization_model_path: Option<PathBuf>,

    /// Path to the `telemuze-diarize` subprocess binary.
    /// If omitted, looks alongside the running telemuze binary, then on PATH.
    /// When the binary cannot be found, diarization is silently disabled.
    #[arg(long, env = "TELEMUZE_DIARIZE_BINARY")]
    pub diarize_binary: Option<PathBuf>,

    /// Path to the telemuze binary used to spawn long-form worker subprocesses.
    /// Defaults to the currently running telemuze binary.
    #[arg(long, env = "TELEMUZE_LONGFORM_BINARY")]
    pub longform_binary: Option<PathBuf>,

    /// Maximum number of concurrent long-form transcription jobs. Additional
    /// jobs queue in FIFO order. Default 1 — one long-form worker at a time.
    #[arg(long, env = "TELEMUZE_MAX_LONGFORM_CONCURRENCY", default_value_t = 1)]
    pub max_longform_concurrency: usize,

    /// Directory for storing downloaded models.
    /// Defaults to ~/.local/share/telemuze/models
    #[arg(long, env = "TELEMUZE_MODELS_DIR")]
    pub models_dir: Option<PathBuf>,

    /// Enable LLM dictation correction. When disabled (the default), the LLM
    /// model is NOT downloaded or loaded into RAM, and dictation relies
    /// solely on the phonetic/fuzzy dictionary pipeline.
    #[arg(long, env = "TELEMUZE_ENABLE_LLM_CORRECTION")]
    pub enable_llm_correction: bool,

    /// URL of an OpenAI-compatible chat completions API for LLM correction.
    /// Example: http://127.0.0.1:8081/v1/chat/completions
    /// When set, uses the HTTP backend instead of native inference.
    /// Only consulted when --enable-llm-correction is set.
    #[arg(long, env = "TELEMUZE_LLM_API_URL", default_value = "")]
    pub llm_api_url: String,

    /// Path to a GGUF model file for native LLM inference.
    /// If omitted (and no --llm-api-url), the model selected by
    /// --llm-model-size is auto-downloaded.
    /// Only consulted when --enable-llm-correction is set.
    #[arg(long, env = "TELEMUZE_LLM_MODEL_PATH")]
    pub llm_model_path: Option<PathBuf>,

    /// LLM model size for auto-download: "0.8b" or "2b".
    #[arg(long, env = "TELEMUZE_LLM_MODEL_SIZE", default_value = "2b",
           value_parser = clap::builder::PossibleValuesParser::new(["0.8b", "2b"]))]
    pub llm_model_size: String,

    /// Path to a terms file for smart dictation correction.
    /// One term per line — the LLM matches sound-alikes automatically.
    /// Defaults to ~/.config/telemuze/terms.txt
    #[arg(long, env = "TELEMUZE_TERMS_FILE")]
    pub terms_file: Option<PathBuf>,

    /// Temperature for LLM sampling (native backend).
    /// Qwen3.5 recommends 1.0 for non-thinking mode.
    #[arg(long, env = "TELEMUZE_LLM_TEMPERATURE", default_value_t = 1.0)]
    pub llm_temperature: f32,

    /// Disable phonetic matching (Double Metaphone) in the dictionary pipeline.
    #[arg(long, env = "TELEMUZE_DISABLE_PHONETIC_MATCH")]
    pub disable_phonetic_match: bool,

    /// Disable fuzzy string matching (Jaro-Winkler) in the dictionary pipeline.
    #[arg(long, env = "TELEMUZE_DISABLE_FUZZY_MATCH")]
    pub disable_fuzzy_match: bool,

    /// Jaro-Winkler similarity threshold for fuzzy matching (0.0–1.0).
    #[arg(long, env = "TELEMUZE_FUZZY_THRESHOLD", default_value_t = 0.85)]
    pub fuzzy_threshold: f64,

    /// Score boost applied to hotwords during recognition (0.0 = disabled).
    /// Requires modified_beam_search decoding. Typical range: 1.0–4.0.
    #[arg(long, env = "TELEMUZE_HOTWORDS_SCORE", default_value_t = 1.5)]
    pub hotwords_score: f32,

    /// Maximum number of active beam search paths (beam width).
    /// Lower values are faster but may reduce accuracy. Range: 1–10.
    #[arg(long, env = "TELEMUZE_MAX_ACTIVE_PATHS", default_value_t = 4)]
    pub max_active_paths: i32,

    /// Penalty applied to the blank token during beam search decoding.
    /// Positive values penalize blanks (slower frame advancement).
    /// Negative values boost blanks (faster advancement, helps prevent loops).
    #[arg(long, env = "TELEMUZE_BLANK_PENALTY", default_value_t = 0.0)]
    pub blank_penalty: f32,

    /// Telegram API ID (from https://my.telegram.org)
    #[arg(long, env = "TELEGRAM_API_ID", default_value_t = 0)]
    pub telegram_api_id: i32,

    /// Telegram API hash (from https://my.telegram.org)
    #[arg(long, env = "TELEGRAM_API_HASH", default_value = "")]
    pub telegram_api_hash: String,

    /// Telegram bot token (from @BotFather)
    #[arg(long, env = "TELEGRAM_BOT_TOKEN", default_value = "")]
    pub telegram_bot_token: String,

    /// Comma-separated list of Telegram usernames allowed to use the bot.
    /// If empty, all users are allowed.
    #[arg(long, env = "TELEGRAM_ALLOWED_USERS", default_value = "")]
    pub telegram_allowed_users: String,
}

impl Config {
    /// Resolved models directory, defaulting to
    /// `~/.local/share/telemuze/models`.
    pub fn resolved_models_dir(&self) -> PathBuf {
        if let Some(ref dir) = self.models_dir {
            dir.clone()
        } else {
            dirs_next::data_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("telemuze")
                .join("models")
        }
    }

    /// Map the user-facing `--stt-model` name ("v2" / "v3" / "unified") to
    /// the registry id used by `ModelManager`.
    pub fn stt_model_id(&self) -> &'static str {
        match self.stt_model.as_str() {
            "v2" => "parakeet-tdt-0.6b-v2",
            "v3" => "parakeet-tdt-0.6b-v3",
            _ => "parakeet-unified-en-0.6b-int8",
        }
    }

    /// Resolved terms file path, defaulting to
    /// `~/.config/telemuze/terms.txt`.
    pub fn resolved_terms_file(&self) -> PathBuf {
        if let Some(ref path) = self.terms_file {
            path.clone()
        } else {
            dirs_next::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("telemuze")
                .join("terms.txt")
        }
    }
}

// ── TOML file struct ─────────────────────────────────────────────────────

/// TOML file layout. Every field is optional so partial configs work; missing
/// fields fall back to the clap default.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "kebab-case", default)]
pub struct FileConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub stt_model_path: Option<PathBuf>,
    pub stt_model: Option<String>,
    pub vad_model_path: Option<PathBuf>,
    pub diarization_model_path: Option<PathBuf>,
    pub diarize_binary: Option<PathBuf>,
    pub longform_binary: Option<PathBuf>,
    pub max_longform_concurrency: Option<usize>,
    pub models_dir: Option<PathBuf>,
    pub enable_llm_correction: Option<bool>,
    pub llm_api_url: Option<String>,
    pub llm_model_path: Option<PathBuf>,
    pub llm_model_size: Option<String>,
    pub terms_file: Option<PathBuf>,
    pub llm_temperature: Option<f32>,
    pub disable_phonetic_match: Option<bool>,
    pub disable_fuzzy_match: Option<bool>,
    pub fuzzy_threshold: Option<f64>,
    pub hotwords_score: Option<f32>,
    pub max_active_paths: Option<i32>,
    pub blank_penalty: Option<f32>,
    pub telegram_api_id: Option<i32>,
    pub telegram_api_hash: Option<String>,
    pub telegram_bot_token: Option<String>,
    pub telegram_allowed_users: Option<String>,
}

// ── Config loading ───────────────────────────────────────────────────────

/// Default config file location: `~/.config/telemuze/server.toml`.
pub fn default_config_path() -> PathBuf {
    dirs_next::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("telemuze")
        .join("server.toml")
}

/// Load config from an explicit path, or the XDG default if present.
pub fn load_config(explicit_path: Option<&Path>) -> Result<(FileConfig, Option<PathBuf>)> {
    let path = match explicit_path {
        Some(p) => {
            if !p.exists() {
                anyhow::bail!("Config file not found: {}", p.display());
            }
            p.to_owned()
        }
        None => {
            let default = default_config_path();
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

// ── Merging CLI + config ─────────────────────────────────────────────────

/// True when clap reports the value came from the command line or an
/// environment variable (i.e. the user explicitly set it).
fn is_explicit(matches: &ArgMatches, id: &str) -> bool {
    let id = id.replace('-', "_");
    matches
        .value_source(&id)
        .is_some_and(|s| {
            matches!(
                s,
                clap::parser::ValueSource::CommandLine | clap::parser::ValueSource::EnvVariable
            )
        })
}

/// Merge a config file into the parsed CLI values. Precedence:
/// CLI arg / env (if explicit) > config file > clap default (already in cli).
pub fn resolve(mut cli: Config, matches: &ArgMatches) -> Result<(Config, Option<PathBuf>)> {
    let (file, config_path) = load_config(cli.config.as_deref())?;

    macro_rules! merge {
        ($field:ident, $id:expr) => {
            if !is_explicit(matches, $id) {
                if let Some(v) = file.$field {
                    cli.$field = v;
                }
            }
        };
    }
    macro_rules! merge_opt {
        ($field:ident, $id:expr) => {
            if !is_explicit(matches, $id) && cli.$field.is_none() {
                cli.$field = file.$field;
            }
        };
    }

    merge_opt!(host, "host");
    merge!(port, "port");
    merge_opt!(stt_model_path, "stt-model-path");
    merge!(stt_model, "stt-model");
    merge_opt!(vad_model_path, "vad-model-path");
    merge_opt!(diarization_model_path, "diarization-model-path");
    merge_opt!(diarize_binary, "diarize-binary");
    merge_opt!(longform_binary, "longform-binary");
    merge!(max_longform_concurrency, "max-longform-concurrency");
    merge_opt!(models_dir, "models-dir");
    merge!(enable_llm_correction, "enable-llm-correction");
    merge!(llm_api_url, "llm-api-url");
    merge_opt!(llm_model_path, "llm-model-path");
    merge!(llm_model_size, "llm-model-size");
    merge_opt!(terms_file, "terms-file");
    merge!(llm_temperature, "llm-temperature");
    merge!(disable_phonetic_match, "disable-phonetic-match");
    merge!(disable_fuzzy_match, "disable-fuzzy-match");
    merge!(fuzzy_threshold, "fuzzy-threshold");
    merge!(hotwords_score, "hotwords-score");
    merge!(max_active_paths, "max-active-paths");
    merge!(blank_penalty, "blank-penalty");
    merge!(telegram_api_id, "telegram-api-id");
    merge!(telegram_api_hash, "telegram-api-hash");
    merge!(telegram_bot_token, "telegram-bot-token");
    merge!(telegram_allowed_users, "telegram-allowed-users");

    Ok((cli, config_path))
}

// ── Dump config ──────────────────────────────────────────────────────────

/// Generate a complete commented TOML config from the resolved values.
pub fn dump(cfg: &Config) -> String {
    let mut out = String::new();
    let mut line = |s: &str| {
        out.push_str(s);
        out.push('\n');
    };

    let opt_path = |p: &Option<PathBuf>, placeholder: &str| -> String {
        match p {
            Some(p) => format!("{} = \"{}\"", placeholder, p.display()),
            None => format!("# {placeholder} = \"/path/to/file\""),
        }
    };

    line("# ── Networking ────────────────────────────────────────────────────────────");
    line("# Host to bind. Omit to listen on both IPv4 (0.0.0.0) and IPv6 ([::]).");
    match &cfg.host {
        Some(h) => line(&format!("host = \"{h}\"")),
        None => line("# host = \"0.0.0.0\""),
    }
    line("");
    line("# TCP port to listen on.");
    line(&format!("port = {}", cfg.port));
    line("");

    line("# ── Models ────────────────────────────────────────────────────────────────");
    line("# Directory for auto-downloaded models (STT, VAD, LLM, diarization).");
    line("# Defaults to ~/.local/share/telemuze/models when omitted.");
    match &cfg.models_dir {
        Some(p) => line(&format!("models-dir = \"{}\"", p.display())),
        None => line("# models-dir = \"/var/lib/telemuze/models\""),
    }
    line("");
    line("# STT model to auto-download: \"v2\" | \"v3\" | \"unified\".");
    line("# Ignored when stt-model-path is set.");
    line(&format!("stt-model = \"{}\"", cfg.stt_model));
    line("");
    line("# Explicit paths. If set, these override auto-download for that model.");
    line(&opt_path(&cfg.stt_model_path, "stt-model-path"));
    line(&opt_path(&cfg.vad_model_path, "vad-model-path"));
    line(&opt_path(&cfg.diarization_model_path, "diarization-model-path"));
    line("");

    line("# ── Long-form workers ─────────────────────────────────────────────────────");
    line("# Path to the `telemuze-diarize` binary (diarization subprocess).");
    line("# If omitted, looks alongside the telemuze binary, then on PATH.");
    line(&opt_path(&cfg.diarize_binary, "diarize-binary"));
    line("# Path to the telemuze binary used to spawn long-form workers.");
    line(&opt_path(&cfg.longform_binary, "longform-binary"));
    line("# Max concurrent long-form jobs. Additional jobs queue FIFO.");
    line(&format!("max-longform-concurrency = {}", cfg.max_longform_concurrency));
    line("");

    line("# ── LLM correction (opt-in) ───────────────────────────────────────────────");
    line("# When false (default), no LLM model is downloaded or loaded; dictation");
    line("# is corrected via the phonetic/fuzzy dictionary pipeline only.");
    line(&format!("enable-llm-correction = {}", cfg.enable_llm_correction));
    line("");
    line("# Optional OpenAI-compatible chat completions URL (external LLM).");
    line("# When set, uses the HTTP backend instead of native inference.");
    if cfg.llm_api_url.is_empty() {
        line("# llm-api-url = \"http://127.0.0.1:8081/v1/chat/completions\"");
    } else {
        line(&format!("llm-api-url = \"{}\"", cfg.llm_api_url));
    }
    line("");
    line("# Path to a GGUF model file for native inference.");
    line("# If omitted, auto-downloads the model selected by llm-model-size.");
    line(&opt_path(&cfg.llm_model_path, "llm-model-path"));
    line("# LLM model size for auto-download: \"0.8b\" | \"2b\".");
    line(&format!("llm-model-size = \"{}\"", cfg.llm_model_size));
    line("# Sampling temperature for the native backend (Qwen3.5 recommends 1.0).");
    line(&format!("llm-temperature = {}", cfg.llm_temperature));
    line("");

    line("# ── Dictionary pipeline ───────────────────────────────────────────────────");
    line("# Path to the dictation terms file (one term per line).");
    line("# Defaults to ~/.config/telemuze/terms.txt when omitted.");
    line(&opt_path(&cfg.terms_file, "terms-file"));
    line("# Disable phonetic (Double Metaphone) matching.");
    line(&format!("disable-phonetic-match = {}", cfg.disable_phonetic_match));
    line("# Disable fuzzy (Jaro-Winkler) matching.");
    line(&format!("disable-fuzzy-match = {}", cfg.disable_fuzzy_match));
    line("# Jaro-Winkler similarity threshold (0.0–1.0).");
    line(&format!("fuzzy-threshold = {}", cfg.fuzzy_threshold));
    line("");

    line("# ── STT decoding ──────────────────────────────────────────────────────────");
    line("# Hotword score boost during recognition (0.0 disables; typical 1.0–4.0).");
    line(&format!("hotwords-score = {}", cfg.hotwords_score));
    line("# Beam search width (1–10). Lower is faster, may reduce accuracy.");
    line(&format!("max-active-paths = {}", cfg.max_active_paths));
    line("# Blank-token penalty; positive slows frame advance, negative speeds it.");
    line(&format!("blank-penalty = {}", cfg.blank_penalty));
    line("");

    line("# ── Telegram bot (optional) ───────────────────────────────────────────────");
    line("# All three of telegram-api-id / telegram-api-hash / telegram-bot-token");
    line("# must be set to enable the bot.");
    line(&format!("telegram-api-id = {}", cfg.telegram_api_id));
    line(&format!("telegram-api-hash = \"{}\"", cfg.telegram_api_hash));
    line(&format!("telegram-bot-token = \"{}\"", cfg.telegram_bot_token));
    line("# Comma-separated allowed @usernames. Empty = allow all.");
    line(&format!("telegram-allowed-users = \"{}\"", cfg.telegram_allowed_users));

    out
}
