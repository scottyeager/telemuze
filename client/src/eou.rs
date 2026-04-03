//! End-of-utterance (EOU) model integration using sherpa-onnx OnlineRecognizer.
//!
//! The parakeet_realtime_eou_120m-v1 model is a cache-aware streaming
//! FastConformer-RNNT that emits `<EOU>` tokens at semantic utterance
//! boundaries.  This module handles model discovery, initialization, and
//! a decode helper that returns structured events.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::debug;

// ── Defaults ──────────────────────────────────────────────────────────────

const DEFAULT_MODEL_NAME: &str = "parakeet-realtime-eou-120m-v1";
const DEFAULT_FEAT_DIM: i32 = 128;
const DEFAULT_THREADS: i32 = 2;
const DEFAULT_BLANK_PENALTY: f32 = 0.0;

pub const DEFAULT_EOU_THREADS: i32 = DEFAULT_THREADS;
pub const DEFAULT_EOU_BLANK_PENALTY: f32 = DEFAULT_BLANK_PENALTY;

// ── Config ────────────────────────────────────────────────────────────────

pub struct EouConfig {
    pub model_dir: PathBuf,
    pub num_threads: i32,
    pub blank_penalty: f32,
    pub use_int8: bool,
}

// ── Events ────────────────────────────────────────────────────────────────

/// Result of a decode cycle.
#[allow(dead_code)]
pub enum EouEvent {
    /// Partial text update (no boundary yet).
    Partial(String),
    /// `<EOU>` token detected — semantic utterance boundary.
    Eou(String),
}

// ── Model file discovery ──────────────────────────────────────────────────

pub fn default_model_dir() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(format!("telemuze/models/{DEFAULT_MODEL_NAME}"))
}

fn find_model_files(dir: &Path, int8: bool) -> Result<(String, String, String, String)> {
    let suffix = if int8 { ".int8.onnx" } else { ".onnx" };

    let encoder = dir.join(format!("encoder{suffix}"));
    let decoder = dir.join(format!("decoder{suffix}"));
    let joiner = dir.join(format!("joiner{suffix}"));
    let tokens = dir.join("tokens.txt");

    anyhow::ensure!(encoder.exists(), "EOU encoder not found: {}", encoder.display());
    anyhow::ensure!(decoder.exists(), "EOU decoder not found: {}", decoder.display());
    anyhow::ensure!(joiner.exists(), "EOU joiner not found: {}", joiner.display());
    anyhow::ensure!(tokens.exists(), "EOU tokens not found: {}", tokens.display());

    Ok((
        encoder.to_string_lossy().into_owned(),
        decoder.to_string_lossy().into_owned(),
        joiner.to_string_lossy().into_owned(),
        tokens.to_string_lossy().into_owned(),
    ))
}

// ── Initialization ────────────────────────────────────────────────────────

pub fn init_eou(
    cfg: &EouConfig,
) -> Result<(sherpa_onnx::OnlineRecognizer, sherpa_onnx::OnlineStream)> {
    let (encoder, decoder, joiner, tokens) = find_model_files(&cfg.model_dir, cfg.use_int8)?;

    let mut config = sherpa_onnx::OnlineRecognizerConfig::default();
    config.model_config.transducer.encoder = Some(encoder);
    config.model_config.transducer.decoder = Some(decoder);
    config.model_config.transducer.joiner = Some(joiner);
    config.model_config.tokens = Some(tokens);
    config.model_config.num_threads = cfg.num_threads;
    config.model_config.provider = Some("cpu".into());

    config.feat_config.feature_dim = DEFAULT_FEAT_DIM;

    config.decoding_method = Some("greedy_search".into());
    config.enable_endpoint = false; // VAD handles silence detection
    config.blank_penalty = cfg.blank_penalty;

    let recognizer = sherpa_onnx::OnlineRecognizer::create(&config)
        .context("Failed to create EOU recognizer")?;
    let stream = recognizer.create_stream();
    Ok((recognizer, stream))
}

// ── Decode helper ─────────────────────────────────────────────────────────

/// Run all available decode steps and return the most significant event.
///
/// Priority: Eou > Partial > None (no frames ready).
pub fn decode_all(
    recognizer: &sherpa_onnx::OnlineRecognizer,
    stream: &sherpa_onnx::OnlineStream,
) -> Option<EouEvent> {
    let mut last_text = String::new();
    let mut decoded_any = false;

    while recognizer.is_ready(stream) {
        recognizer.decode(stream);
        decoded_any = true;

        if let Some(result) = recognizer.get_result(stream) {
            let has_eou = result.tokens.iter().any(|t| t.contains("EOU"));

            if has_eou {
                // Strip special tokens from final text.
                let final_text: String = result
                    .tokens
                    .iter()
                    .filter(|t| !t.contains("EOU") && !t.contains("EOB") && !t.contains("blk"))
                    .cloned()
                    .collect();
                let final_text = final_text.trim().to_string();
                debug!(text = %final_text, "EOU token detected");
                return Some(EouEvent::Eou(final_text));
            }

            last_text = result.text.trim().to_string();
        }
    }

    if decoded_any && !last_text.is_empty() {
        Some(EouEvent::Partial(last_text))
    } else {
        None
    }
}
