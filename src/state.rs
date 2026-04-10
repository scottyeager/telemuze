use anyhow::Result;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;
use tracing::{error, info};

use crate::config::Config;
use crate::engines::diarization::DiarizationEngine;
use crate::engines::dictionary::{Dictionary, PipelineConfig};
use crate::engines::llm::LlmEngine;
use crate::engines::stt::SttEngine;
use crate::engines::vad::{SpeechSegment, VadEngine};
use crate::models::ModelManager;

/// A transcribed speech segment with timestamps and token-level timing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscribedSegment {
    pub start_secs: f64,
    pub end_secs: f64,
    pub text: String,
    /// Per-token strings from the recognizer (BPE subwords).
    pub tokens: Vec<String>,
    /// Per-token timestamps in seconds, relative to segment start.
    pub token_timestamps: Vec<f32>,
}

/// Shared application state holding all loaded AI models.
///
/// Models are loaded once at startup and shared across all request handlers
/// via `Arc<AppState>` in axum's state extractor.
pub struct AppState {
    pub stt_engine: Mutex<SttEngine>,
    pub llm_engine: LlmEngine,
    pub vad_engine: Mutex<VadEngine>,
    /// Present when both the diarize subprocess binary and the Sortformer
    /// model file are located at startup. The engine is internally
    /// stateless (each call spawns a fresh subprocess), so no Mutex.
    pub diarization_engine: Option<DiarizationEngine>,
    pub terms_content: String,
    pub dictionary: Dictionary,
    pub pipeline_config: PipelineConfig,
    pub disable_llm_correction: bool,
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
        let stt_engine = SttEngine::new(
            &stt_path,
            config.hotwords_score,
            config.max_active_paths,
            config.blank_penalty,
        )?;
        info!("STT model loaded.");

        // Initialize LLM engine:
        // - If --llm-api-url is set, use HTTP backend
        // - If --llm-model-path is set, use native with that GGUF
        // - Otherwise, auto-download based on --llm-model-size
        info!("Initializing LLM engine...");
        let llm_engine = if !config.llm_api_url.is_empty() {
            LlmEngine::new_http(&config.llm_api_url)
        } else {
            let gguf_path = if let Some(ref path) = config.llm_model_path {
                path.clone()
            } else {
                let model_id = format!("qwen3.5-{}", config.llm_model_size);
                info!("Ensuring LLM model {} is available...", model_id);
                mgr.ensure_llm_model(&model_id).await?;
                mgr.llm_model_path(&model_id)?
            };
            LlmEngine::new_native(&gguf_path, config.llm_temperature)?
        };

        info!("Loading VAD model from {:?}...", vad_path);
        let vad_engine = VadEngine::new(&vad_path)?;
        info!("VAD model loaded.");

        let diarization_engine = Self::load_diarization_engine(config, &mgr);

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

        // Build phonetic dictionary from terms
        let dictionary = Dictionary::from_terms_content(&terms_content);

        let pipeline_config = PipelineConfig {
            phonetic_enabled: !config.disable_phonetic_match,
            fuzzy_enabled: !config.disable_fuzzy_match,
            fuzzy_threshold: config.fuzzy_threshold,
        };
        info!(
            "Dictionary pipeline: phonetic={}, fuzzy={} (threshold={:.2}), llm={}",
            pipeline_config.phonetic_enabled,
            pipeline_config.fuzzy_enabled,
            pipeline_config.fuzzy_threshold,
            !config.disable_llm_correction,
        );

        if !telegram_allowed_users.is_empty() {
            info!("Telegram allowed users: {:?}", telegram_allowed_users);
        }

        Ok(Self {
            stt_engine: Mutex::new(stt_engine),
            llm_engine,
            vad_engine: Mutex::new(vad_engine),
            diarization_engine,
            terms_content,
            dictionary,
            pipeline_config,
            disable_llm_correction: config.disable_llm_correction,
            telegram_allowed_users,
        })
    }

    fn load_diarization_engine(config: &Config, mgr: &ModelManager) -> Option<DiarizationEngine> {
        let binary_path = match locate_diarize_binary(config) {
            Some(p) => p,
            None => {
                info!("telemuze-diarize binary not found — speaker labels disabled.");
                return None;
            }
        };

        let model_path = match config.diarization_model_path.as_ref() {
            Some(p) => p.clone(),
            None => {
                let p = mgr
                    .models_dir()
                    .join("diar_streaming_sortformer_4spk-v2.1.onnx");
                if !p.exists() {
                    info!(
                        "Diarization model not found at {} — speaker labels disabled.",
                        p.display()
                    );
                    return None;
                }
                p
            }
        };

        info!(
            "Diarization configured (binary={}, model={})",
            binary_path.display(),
            model_path.display()
        );
        Some(DiarizationEngine::new(binary_path, model_path))
    }

    /// Run VAD segmentation followed by per-segment STT transcription.
    ///
    /// Returns transcribed segments (skipping empty/failed ones).
    pub fn vad_transcribe(&self, pcm: &[f32]) -> Result<Vec<TranscribedSegment>> {
        self.vad_transcribe_with_hotwords(pcm, None)
    }

    /// Run VAD segmentation + per-segment STT with optional hotwords.
    pub fn vad_transcribe_with_hotwords(
        &self,
        pcm: &[f32],
        hotwords: Option<&str>,
    ) -> Result<Vec<TranscribedSegment>> {
        let segments = self.vad_engine.lock().unwrap().segment_audio(pcm)?;
        info!("VAD found {} speech segments", segments.len());

        let mut results = Vec::with_capacity(segments.len());
        for (i, seg) in segments.iter().enumerate() {
            match self.stt_engine.lock().unwrap().transcribe_with_hotwords(&seg.samples, hotwords) {
                Ok(stt_result) if !stt_result.text.trim().is_empty() => {
                    info!(
                        "Segment {}/{}: [{:.1}s - {:.1}s] '{}'",
                        i + 1,
                        segments.len(),
                        seg.start_secs,
                        seg.end_secs,
                        stt_result.text,
                    );
                    results.push(TranscribedSegment {
                        start_secs: seg.start_secs,
                        end_secs: seg.end_secs,
                        text: stt_result.text,
                        tokens: stt_result.tokens,
                        token_timestamps: stt_result.timestamps,
                    });
                }
                Ok(_) => {}
                Err(e) => {
                    error!("STT failed for segment {}: {e}", i + 1);
                }
            }
        }

        Ok(results)
    }

    /// Run VAD segmentation only, returning the speech segments for
    /// callers that want to drive STT themselves (e.g. with progress updates).
    pub fn vad_segment(&self, pcm: &[f32]) -> Result<Vec<SpeechSegment>> {
        let segments = self.vad_engine.lock().unwrap().segment_audio(pcm)?;
        info!("VAD found {} speech segments", segments.len());
        Ok(segments)
    }

    /// Transcribe a single speech segment, returning the text (or empty on failure).
    pub fn transcribe_segment(
        &self,
        seg: &SpeechSegment,
        index: usize,
        total: usize,
        hotwords: Option<&str>,
    ) -> Option<TranscribedSegment> {
        match self.stt_engine.lock().unwrap().transcribe_with_hotwords(&seg.samples, hotwords) {
            Ok(stt_result) if !stt_result.text.trim().is_empty() => {
                info!(
                    "Segment {}/{}: [{:.1}s - {:.1}s] '{}'",
                    index + 1,
                    total,
                    seg.start_secs,
                    seg.end_secs,
                    stt_result.text,
                );
                Some(TranscribedSegment {
                    start_secs: seg.start_secs,
                    end_secs: seg.end_secs,
                    text: stt_result.text,
                    tokens: stt_result.tokens,
                    token_timestamps: stt_result.timestamps,
                })
            }
            Ok(_) => None,
            Err(e) => {
                error!("STT failed for segment {}: {e}", index + 1);
                None
            }
        }
    }
}

/// Locate the `telemuze-diarize` subprocess binary.
///
/// Search order:
/// 1. Explicit `--diarize-binary` / `TELEMUZE_DIARIZE_BINARY` override.
/// 2. Sibling of the running `telemuze` binary (matches Cargo's
///    `target/{debug,release}/` layout and any reasonable install layout).
/// 3. PATH lookup.
fn locate_diarize_binary(config: &Config) -> Option<PathBuf> {
    if let Some(p) = &config.diarize_binary {
        return if p.exists() { Some(p.clone()) } else { None };
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("telemuze-diarize");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    which_in_path("telemuze-diarize")
}

/// Minimal PATH walker — looks for an executable file with the given
/// name in each `PATH` entry. Avoids pulling in the `which` crate.
fn which_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
