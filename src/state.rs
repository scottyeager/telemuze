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
    pub custom_terms: Vec<String>,
    pub telegram_allowed_users: HashSet<String>,
}

impl AppState {
    pub async fn new(config: &Config) -> Result<Self> {
        // Resolve model paths — use explicit paths if given, otherwise
        // auto-download to the models directory.
        let (stt_path, vad_path) = if config.stt_model_path.is_some() && config.vad_model_path.is_some()
        {
            (
                config.stt_model_path.clone().unwrap(),
                config.vad_model_path.clone().unwrap(),
            )
        } else {
            let models_dir = config.resolved_models_dir();
            info!("Models directory: {}", models_dir.display());

            let mgr = ModelManager::new(models_dir)?;
            if !mgr.all_models_available() {
                info!("Downloading missing models...");
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

        info!("Initializing LLM engine...");
        let llm_engine = LlmEngine::new(&config.llm_api_url).await?;
        info!("LLM engine ready.");

        info!("Loading VAD model from {:?}...", vad_path);
        let vad_engine = VadEngine::new(&vad_path)?;
        info!("VAD model loaded.");

        let custom_terms: Vec<String> = config
            .custom_terms
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if !custom_terms.is_empty() {
            info!("Custom dictionary terms: {:?}", custom_terms);
        }

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
            custom_terms,
            telegram_allowed_users,
        })
    }
}
