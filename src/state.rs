use anyhow::{Context, Result};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tempfile::NamedTempFile;
use tokio::sync::Semaphore;
use tracing::{error, info};

use crate::config::Config;
use crate::engines::dictionary::{Dictionary, PipelineConfig};
use crate::engines::llm::LlmEngine;
use crate::engines::long_form::LongFormEngine;
use crate::engines::stt::SttEngine;
use crate::engines::vad::{SpeechSegment, VadEngine};
use crate::long_form::LongFormOutcome;
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
    /// LLM correction engine — `None` when --enable-llm-correction is not set.
    /// No model is downloaded or loaded into RAM in that case.
    pub llm_engine: Option<LlmEngine>,
    pub vad_engine: Mutex<VadEngine>,
    /// Long-form worker launcher — spawns `telemuze transcribe` children.
    /// Carries the Sortformer model path (if configured); the worker loads
    /// it on demand and runs diarization alongside ASR in one subprocess.
    pub long_form_engine: LongFormEngine,
    /// FIFO queue for long-form jobs. Permit count defaults to 1 so only
    /// one worker runs at a time (bounded peak RAM).
    pub long_form_semaphore: Arc<Semaphore>,
    pub terms_content: String,
    pub dictionary: Dictionary,
    pub pipeline_config: PipelineConfig,
    pub telegram_allowed_users: HashSet<String>,
}

impl AppState {
    pub async fn new(config: &Config) -> Result<Self> {
        // Resolve model paths — use explicit paths if given, otherwise
        // auto-download to the models directory.
        let models_dir = config.resolved_models_dir();
        info!("Models directory: {}", models_dir.display());
        let stt_id = config.stt_model_id();
        info!("Selected STT model: {} ({})", config.stt_model, stt_id);
        let mgr = ModelManager::new(models_dir, stt_id)?;

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
            2,
        )?;
        info!("STT model loaded.");

        // Initialize LLM engine only when explicitly enabled. Otherwise the
        // GGUF is neither downloaded nor loaded into RAM.
        let llm_engine = if config.enable_llm_correction {
            info!("Initializing LLM engine (--enable-llm-correction set)...");
            let engine = if !config.llm_api_url.is_empty() {
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
            Some(engine)
        } else {
            info!("LLM correction disabled — model will not be downloaded or loaded.");
            None
        };

        info!("Loading VAD model from {:?}...", vad_path);
        let vad_engine = VadEngine::new(&vad_path)?;
        info!("VAD model loaded.");

        let diarize_model_path = Self::locate_diarize_model(config, &mgr);

        let long_form_binary = locate_long_form_binary(config);
        info!("Long-form worker binary: {}", long_form_binary.display());
        let long_form_engine = LongFormEngine::new(
            long_form_binary,
            stt_path.clone(),
            vad_path.clone(),
            diarize_model_path,
            config.hotwords_score,
            config.max_active_paths,
            config.blank_penalty,
            8,
        );
        let permits = config.max_longform_concurrency.max(1);
        info!("Long-form concurrency: {permits} permit(s)");
        let long_form_semaphore = Arc::new(Semaphore::new(permits));

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
            llm_engine.is_some(),
        );

        if !telegram_allowed_users.is_empty() {
            info!("Telegram allowed users: {:?}", telegram_allowed_users);
        }

        Ok(Self {
            stt_engine: Mutex::new(stt_engine),
            llm_engine,
            vad_engine: Mutex::new(vad_engine),
            long_form_engine,
            long_form_semaphore,
            terms_content,
            dictionary,
            pipeline_config,
            telegram_allowed_users,
        })
    }

    /// Run a long-form transcription job: acquires the long-form permit
    /// (FIFO queue), materializes the PCM to a single tempfile, spawns
    /// one `telemuze transcribe` worker that runs ASR and (if configured)
    /// diarization in parallel on the same PCM, and returns the merged
    /// outcome. The permit is held across the subprocess spawn so the
    /// peak RAM from worker children is bounded by the permit count.
    pub async fn long_form_transcribe(
        self: &Arc<Self>,
        pcm: &[f32],
        hotwords: Option<&str>,
    ) -> Result<LongFormOutcome> {
        let waited_start = std::time::Instant::now();
        let _permit = self
            .long_form_semaphore
            .clone()
            .acquire_owned()
            .await
            .context("Long-form semaphore closed")?;
        let waited = waited_start.elapsed();
        if waited > std::time::Duration::from_millis(50) {
            info!(
                "Long-form job waited {:.1}s in queue",
                waited.as_secs_f64()
            );
        }

        let pcm_file = write_pcm_tempfile(pcm)?;
        let pcm_path = pcm_file.path().to_path_buf();

        let long_form = self.long_form_engine.clone();
        let hw_owned = hotwords.map(str::to_owned);
        let result = tokio::task::spawn_blocking(move || {
            long_form.transcribe_and_diarize(&pcm_path, hw_owned.as_deref())
        })
        .await
        .context("Long-form worker task join failed")??;

        drop(pcm_file);

        if let Some(ref d) = result.diar {
            let n_spk = d.iter().map(|s| s.speaker).max().map(|m| m + 1).unwrap_or(0);
            info!("Diarization: {} segments, {} speakers", d.len(), n_spk);
        }

        Ok(LongFormOutcome {
            asr_segments: result.segments,
            diar_segments: result.diar,
        })
    }

    fn locate_diarize_model(config: &Config, mgr: &ModelManager) -> Option<PathBuf> {
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

        info!("Diarization configured (model={})", model_path.display());
        Some(model_path)
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

/// Write mono 16 kHz f32 PCM to a temp file as raw f32-LE bytes.
/// Returned `NamedTempFile` auto-deletes when dropped.
fn write_pcm_tempfile(pcm: &[f32]) -> Result<NamedTempFile> {
    let mut tmp = NamedTempFile::new().context("Failed to create PCM tempfile")?;
    let mut bytes = Vec::with_capacity(pcm.len() * 4);
    for sample in pcm {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    tmp.write_all(&bytes)
        .context("Failed to write PCM tempfile")?;
    tmp.flush().context("Failed to flush PCM tempfile")?;
    Ok(tmp)
}

/// Locate the telemuze binary used to spawn long-form workers.
/// Falls back to the currently running executable.
fn locate_long_form_binary(config: &Config) -> PathBuf {
    if let Some(p) = &config.longform_binary {
        if p.exists() {
            return p.clone();
        }
    }
    std::env::current_exe()
        .ok()
        .unwrap_or_else(|| PathBuf::from("telemuze"))
}

