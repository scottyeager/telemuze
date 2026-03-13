use clap::Parser;
use std::path::PathBuf;

/// Telemuze: Self-hosted AI dictation and transcription server.
#[derive(Parser, Debug, Clone)]
#[command(name = "telemuze", version)]
pub struct Config {
    /// Host address to bind to
    #[arg(long, default_value = "0.0.0.0", env = "TELEMUZE_HOST")]
    pub host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 7313, env = "TELEMUZE_PORT")]
    pub port: u16,

    /// Path to the Parakeet ONNX model directory.
    /// If omitted, models are auto-downloaded to --models-dir.
    #[arg(long, env = "TELEMUZE_STT_MODEL_PATH")]
    pub stt_model_path: Option<PathBuf>,

    /// Path to the Silero VAD ONNX model file.
    /// If omitted, models are auto-downloaded to --models-dir.
    #[arg(long, env = "TELEMUZE_VAD_MODEL_PATH")]
    pub vad_model_path: Option<PathBuf>,

    /// Directory for storing downloaded models.
    /// Defaults to ~/.local/share/telemuze/models
    #[arg(long, env = "TELEMUZE_MODELS_DIR")]
    pub models_dir: Option<PathBuf>,

    /// URL of an OpenAI-compatible chat completions API for LLM correction.
    /// Example: http://127.0.0.1:8081/v1/chat/completions
    /// Leave empty to disable LLM correction (raw STT output returned).
    #[arg(long, env = "TELEMUZE_LLM_API_URL", default_value = "")]
    pub llm_api_url: String,

    /// Custom dictionary terms for smart dictation (comma-separated)
    #[arg(long, env = "TELEMUZE_CUSTOM_TERMS", default_value = "")]
    pub custom_terms: String,

    /// Telegram API ID (from https://my.telegram.org)
    #[arg(long, env = "TELEGRAM_API_ID", default_value_t = 0)]
    pub telegram_api_id: i32,

    /// Telegram API hash (from https://my.telegram.org)
    #[arg(long, env = "TELEGRAM_API_HASH", default_value = "")]
    pub telegram_api_hash: String,

    /// Telegram bot token (from @BotFather)
    #[arg(long, env = "TELEGRAM_BOT_TOKEN", default_value = "")]
    pub telegram_bot_token: String,
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
}
