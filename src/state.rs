use anyhow::Result;
use std::collections::HashSet;
use std::sync::Mutex;
use tracing::info;

use crate::config::Config;
use crate::engines::llm::LlmEngine;
use crate::engines::stt::SttEngine;
use crate::engines::vad::VadEngine;
use crate::models::ModelManager;

/// Shared application state holding all loaded AI models.
///
/// Models are loaded once at startup and shared across all request handlers
/// via `Arc<AppState>` in axum's state extractor.
pub struct AppState {
    pub stt_engine: Mutex<SttEngine>,
    pub llm_engine: LlmEngine,
    pub vad_engine: Mutex<VadEngine>,
    pub terms_content: String,
    pub telegram_allowed_users: HashSet<String>,
}

impl AppState {
    pub async fn new(config: &Config) -> Result<Self> {
        // Resolve model paths — use explicit paths if given, otherwise
        // auto-download to the models directory.
        let models_dir = config.resolved_models_dir();
        info!("Models directory: {}", models_dir.display());
        let mgr = ModelManager::new(models_dir)?;

        let (stt_path, vad_path) = if config.stt_model_path.is_some() && config.vad_model_path.is_some()
        {
            (
                config.stt_model_path.clone().unwrap(),
                config.vad_model_path.clone().unwrap(),
            )
        } else {
            if !mgr.all_models_available() {
                info!("Downloading missing STT/VAD models...");
                mgr.ensure_models().await?;
            }

            (
                config.stt_model_path.clone().unwrap_or_else(|| mgr.stt_model_path().unwrap()),
                config.vad_model_path.clone().unwrap_or_else(|| mgr.vad_model_path().unwrap()),
            )
        };

        info!("Loading STT model from {:?}...", stt_path);
        let stt_engine = SttEngine::new(&stt_path)?;
        info!("STT model loaded.");

        // Initialize LLM engine:
        // - If --llm-api-url is set, use HTTP backend
        // - If --llm-model-path is set, use native with that GGUF
        // - Otherwise, auto-download Qwen3.5-0.8B and use native
        info!("Initializing LLM engine...");
        let llm_engine = if !config.llm_api_url.is_empty() {
            LlmEngine::new_http(&config.llm_api_url)
        } else {
            let gguf_path = if let Some(ref path) = config.llm_model_path {
                path.clone()
            } else {
                info!("Downloading LLM model...");
                mgr.ensure_llm_model().await?;
                mgr.llm_model_path()?
            };
            LlmEngine::new_native(&gguf_path, config.llm_temperature)?
        };

        info!("Loading VAD model from {:?}...", vad_path);
        let vad_engine = VadEngine::new(&vad_path)?;
        info!("VAD model loaded.");

        // Load terms file
        let terms_file = config.resolved_terms_file();
        let terms_content = match std::fs::read_to_string(&terms_file) {
            Ok(content) => {
                info!("Loaded terms from {}", terms_file.display());
                content
            }
            Err(_) => {
                info!("No terms file at {} — LLM will correct without custom terms", terms_file.display());
                String::new()
            }
        };

        let telegram_allowed_users: HashSet<String> = config
            .telegram_allowed_users
            .split(',')
            .map(|s| s.trim().trim_start_matches('@').to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();

        if !telegram_allowed_users.is_empty() {
            info!("Telegram allowed users: {:?}", telegram_allowed_users);
        }

        Ok(Self {
            stt_engine: Mutex::new(stt_engine),
            llm_engine,
            vad_engine: Mutex::new(vad_engine),
            terms_content,
            telegram_allowed_users,
        })
    }
}
