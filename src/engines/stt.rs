//! Speech-to-Text engine wrapper around transcribe-rs.
//!
//! Uses the Parakeet ONNX model via transcribe-rs for fast,
//! accurate speech recognition. Supports both plain transcription
//! and timestamped segment output.

use anyhow::{Context, Result};
use std::path::Path;
use transcribe_rs::onnx::parakeet::ParakeetModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::{SpeechModel, TranscribeOptions};

/// Wraps the Parakeet STT model for thread-safe inference.
pub struct SttEngine {
    engine: ParakeetModel,
    options: TranscribeOptions,
}

// Safety: The underlying ONNX runtime session is thread-safe for inference.
unsafe impl Send for SttEngine {}
unsafe impl Sync for SttEngine {}

impl SttEngine {
    /// Load a Parakeet model from the given directory.
    ///
    /// The directory should contain:
    /// - encoder model (ONNX)
    /// - decoder_joint model (ONNX)
    /// - nemo128.onnx (audio preprocessor)
    /// - vocab.txt (vocabulary)
    pub fn new(model_path: &Path) -> Result<Self> {
        let engine = ParakeetModel::load(model_path, &Quantization::Int8)
            .context("Failed to load Parakeet STT model")?;

        Ok(Self { engine, options: TranscribeOptions::default() })
    }

    /// Transcribe mono 16kHz f32 PCM audio to text.
    pub fn transcribe(&mut self, pcm_16khz: &[f32]) -> Result<String> {
        let result = self
            .engine
            .transcribe(pcm_16khz, &self.options)
            .context("STT transcription failed")?;

        Ok(result.text)
    }

    /// Transcribe with timestamps, returning segments.
    #[allow(dead_code)]
    pub fn transcribe_with_timestamps(&mut self, pcm_16khz: &[f32]) -> Result<Vec<TranscriptSegment>> {
        let result = self
            .engine
            .transcribe(pcm_16khz, &self.options)
            .context("STT transcription with timestamps failed")?;

        let segments = if let Some(segs) = result.segments {
            segs.iter()
                .map(|s| TranscriptSegment {
                    start: s.start as f64,
                    end: s.end as f64,
                    text: s.text.clone(),
                })
                .collect()
        } else {
            // Fallback: return entire text as a single segment
            vec![TranscriptSegment {
                start: 0.0,
                end: pcm_16khz.len() as f64 / 16_000.0,
                text: result.text,
            }]
        };

        Ok(segments)
    }
}

/// A timestamped segment of transcribed text.
#[allow(dead_code)]
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}
