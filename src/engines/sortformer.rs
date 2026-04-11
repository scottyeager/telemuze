//! NVIDIA Sortformer diarization via sherpa-onnx.
//!
//! Replaces the separate `telemuze-diarize` subprocess (which wrapped the
//! `parakeet-rs` crate) with in-process diarization shared with the
//! long-form worker's ASR pass.

use anyhow::{anyhow, Context, Result};
use sherpa_onnx::{
    OfflineSortformerDiarization, OfflineSortformerDiarizationConfig,
    OfflineSortformerDiarizationModelConfig,
};
use std::path::Path;
use tracing::info;

use crate::engines::diarization::DiarSegment;

const SAMPLE_RATE: i32 = 16_000;

pub struct SortformerEngine {
    sd: OfflineSortformerDiarization,
}

// Safety: OfflineSortformerDiarization is already `Send`. The opaque C
// handle it wraps has its own internal synchronization for `process()`
// calls, matching how VadEngine treats its sherpa-onnx handle.
unsafe impl Sync for SortformerEngine {}

impl SortformerEngine {
    pub fn new(model_path: &Path, num_threads: i32) -> Result<Self> {
        let config = OfflineSortformerDiarizationConfig {
            model: OfflineSortformerDiarizationModelConfig {
                model: Some(model_path.to_string_lossy().into_owned()),
                num_threads,
                ..Default::default()
            },
            ..Default::default()
        };

        let sd = OfflineSortformerDiarization::create(&config).ok_or_else(|| {
            anyhow!(
                "Failed to load Sortformer model from {}",
                model_path.display()
            )
        })?;

        let sr = sd.sample_rate();
        anyhow::ensure!(
            sr == SAMPLE_RATE,
            "Sortformer model sample rate {} does not match expected {}",
            sr,
            SAMPLE_RATE
        );

        info!(
            "Sortformer loaded from {} ({} speakers max)",
            model_path.display(),
            sd.num_speakers()
        );

        Ok(Self { sd })
    }

    /// Run Sortformer over mono 16 kHz f32 PCM and return segments sorted
    /// by start time with 0-based speaker labels.
    pub fn diarize(&self, pcm: &[f32]) -> Result<Vec<DiarSegment>> {
        let result = self
            .sd
            .process(pcm)
            .context("Sortformer process() returned null")?;

        Ok(result
            .sort_by_start_time()
            .into_iter()
            .map(|s| DiarSegment {
                start: s.start,
                end: s.end,
                speaker: s.speaker,
            })
            .collect())
    }
}
