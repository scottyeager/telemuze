use anyhow::{Context, Result};
use sherpa_onnx::{
    OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig,
};
use std::path::Path;
use tracing::info;

/// A single diarization segment with a 0-based speaker label.
pub struct DiarSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: i32,
}

/// Wraps the sherpa-onnx offline speaker diarization pipeline
/// (pyannote segmentation → embedding → agglomerative clustering).
pub struct DiarizationEngine {
    diarizer: OfflineSpeakerDiarization,
    /// Baseline config stored so we can update clustering params between calls
    /// without touching the already-loaded model paths.
    config: OfflineSpeakerDiarizationConfig,
}

impl DiarizationEngine {
    pub fn new(segmentation_model: &Path, embedding_model: &Path) -> Result<Self> {
        let config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(segmentation_model.to_string_lossy().into_owned()),
                },
                ..Default::default()
            },
            embedding: SpeakerEmbeddingExtractorConfig {
                model: Some(embedding_model.to_string_lossy().into_owned()),
                ..Default::default()
            },
            // clustering defaults: num_clusters=-1 (auto), threshold=0.5
            ..Default::default()
        };

        let diarizer = OfflineSpeakerDiarization::create(&config)
            .context("Failed to create OfflineSpeakerDiarization")?;

        info!("Diarization engine ready (sample_rate={}Hz)", diarizer.sample_rate());

        Ok(Self { diarizer, config })
    }

    /// Diarize mono 16 kHz PCM. Pass `num_speakers` when known to skip
    /// threshold-based automatic estimation.
    pub fn diarize(&self, pcm: &[f32], num_speakers: Option<i32>) -> Result<Vec<DiarSegment>> {
        // Rebuild clustering config, swapping in num_clusters if provided.
        // set_config only touches clustering params on the already-loaded session.
        if let Some(n) = num_speakers {
            let mut config = self.config.clone();
            config.clustering.num_clusters = n;
            self.diarizer.set_config(&config);
        } else {
            // Reset to auto mode in case a prior call set num_clusters.
            self.diarizer.set_config(&self.config);
        }

        let result = self
            .diarizer
            .process(pcm)
            .context("Diarization returned no result")?;

        info!(
            "Diarization: {} speaker(s), {} segments",
            result.num_speakers(),
            result.num_segments()
        );

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

/// For each ASR segment `[asr_start, asr_end)`, return the speaker whose
/// diarization segment overlaps it the most. Returns `None` if there is no
/// overlap with any diarization segment.
pub fn assign_speakers(
    asr_starts: &[f64],
    asr_ends: &[f64],
    diar: &[DiarSegment],
) -> Vec<Option<i32>> {
    asr_starts
        .iter()
        .zip(asr_ends.iter())
        .map(|(&start, &end)| {
            let mut best_speaker: Option<i32> = None;
            let mut best_overlap = 0.0f64;
            for seg in diar {
                let overlap_start = start.max(seg.start as f64);
                let overlap_end = end.min(seg.end as f64);
                let overlap = (overlap_end - overlap_start).max(0.0);
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_speaker = Some(seg.speaker);
                }
            }
            best_speaker
        })
        .collect()
}
