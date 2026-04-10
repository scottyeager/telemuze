use anyhow::{Context, Result};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tempfile::NamedTempFile;
use tokio::sync::Semaphore;
use tracing::{error, info};

use crate::config::Config;
use crate::engines::diarization::DiarizationEngine;
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
    pub llm_engine: LlmEngine,
    pub vad_engine: Mutex<VadEngine>,
    /// Present when both the diarize subprocess binary and the Sortformer
    /// model file are located at startup. The engine is internally
    /// stateless (each call spawns a fresh subprocess), so no Mutex.
    pub diarization_engine: Option<DiarizationEngine>,
    /// Long-form worker launcher — spawns `telemuze transcribe` children.
    pub long_form_engine: LongFormEngine,
    /// FIFO queue for long-form jobs. Permit count defaults to 1 so only
    /// one worker runs at a time (bounded peak RAM).
    pub long_form_semaphore: Arc<Semaphore>,
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
            2,
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

        let long_form_binary = locate_long_form_binary(config);
        info!("Long-form worker binary: {}", long_form_binary.display());
        let long_form_engine = LongFormEngine::new(
            long_form_binary,
            stt_path.clone(),
            vad_path.clone(),
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
            long_form_engine,
            long_form_semaphore,
            terms_content,
            dictionary,
            pipeline_config,
            disable_llm_correction: config.disable_llm_correction,
            telegram_allowed_users,
        })
    }

    /// Run a long-form transcription job: acquires the long-form permit
    /// (FIFO queue), materializes the PCM to a single tempfile, spawns
    /// transcribe + diarize workers in parallel, and returns the merged
    /// outcome. The permit is held across both subprocess spawns so the
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
        let pcm_for_asr = pcm_path.clone();
        let asr_task = tokio::task::spawn_blocking(move || {
            long_form.transcribe(&pcm_for_asr, hw_owned.as_deref())
        });

        let diar_task = self.diarization_engine.clone().map(|diar| {
            let pcm_for_diar = pcm_path.clone();
            tokio::task::spawn_blocking(move || diar.diarize_from_path(&pcm_for_diar))
        });

        let asr_segments = asr_task.await.context("ASR task join failed")??;

        let diar_segments = match diar_task {
            Some(t) => match t.await.context("Diarization task join failed")? {
                Ok(d) => {
                    let n_spk = d.iter().map(|s| s.speaker).max().map(|m| m + 1).unwrap_or(0);
                    info!("Diarization: {} segments, {} speakers", d.len(), n_spk);
                    Some(d)
                }
                Err(e) => {
                    error!("Diarization failed: {e}");
                    None
                }
            },
            None => None,
        };

        drop(pcm_file);

        Ok(LongFormOutcome {
            asr_segments,
            diar_segments,
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
