//! Voice Activity Detection engine using sherpa-onnx (Silero VAD).
//!
//! Chunks audio into speech segments based on voice activity,
//! used by the long-form transcription endpoint to split large
//! files into manageable pieces with accurate timestamps.

use anyhow::{Context, Result};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use std::path::Path;
use tracing::info;

const SAMPLE_RATE: i32 = 16_000;
const WINDOW_SIZE: i32 = 512;

/// A detected speech segment with start/end times in seconds.
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Start time in seconds from beginning of audio.
    pub start_secs: f64,
    /// End time in seconds from beginning of audio.
    pub end_secs: f64,
    /// The PCM samples for this segment.
    pub samples: Vec<f32>,
}

/// Wraps the Silero VAD model for detecting speech regions.
pub struct VadEngine {
    config: VadModelConfig,
}

// Safety: VadModelConfig is just data. The VoiceActivityDetector is created
// fresh per call and never shared across threads.
unsafe impl Send for VadEngine {}
unsafe impl Sync for VadEngine {}

impl VadEngine {
    /// Load the Silero VAD ONNX model from `model_path`.
    pub fn new(model_path: &Path) -> Result<Self> {
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model_path.to_string_lossy().into_owned()),
                threshold: 0.5,
                min_silence_duration: 0.5,
                min_speech_duration: 0.1,
                max_speech_duration: 15.0,
                window_size: WINDOW_SIZE,
            },
            sample_rate: SAMPLE_RATE,
            num_threads: 1,
            debug: false,
            ..Default::default()
        };

        // Verify the model loads by creating a test detector.
        VoiceActivityDetector::create(&config, 1.0)
            .context("Failed to load Silero VAD model")?;

        info!("Silero VAD loaded from {:?}", model_path);
        Ok(Self { config })
    }

    /// Segment mono 16kHz audio into speech chunks.
    ///
    /// Creates a fresh VAD detector per call to avoid state accumulation.
    /// Feeds audio in 512-sample frames, then flushes trailing speech.
    pub fn segment_audio(&self, pcm_16khz: &[f32]) -> Result<Vec<SpeechSegment>> {
        let buffer_secs = pcm_16khz.len() as f32 / SAMPLE_RATE as f32 + 1.0;
        let vad = VoiceActivityDetector::create(&self.config, buffer_secs)
            .context("Failed to create VAD detector")?;

        // Feed audio in window-sized chunks.
        for chunk in pcm_16khz.chunks(WINDOW_SIZE as usize) {
            if chunk.len() == WINDOW_SIZE as usize {
                vad.accept_waveform(chunk);
            }
        }
        vad.flush();

        // Drain detected segments.
        let mut segments = Vec::new();
        while !vad.is_empty() {
            if let Some(seg) = vad.front() {
                let start_secs = seg.start() as f64 / SAMPLE_RATE as f64;
                let duration_secs = seg.n() as f64 / SAMPLE_RATE as f64;
                segments.push(SpeechSegment {
                    start_secs,
                    end_secs: start_secs + duration_secs,
                    samples: seg.samples().to_vec(),
                });
            }
            vad.pop();
        }

        // If VAD found no segments, treat entire audio as one.
        if segments.is_empty() && !pcm_16khz.is_empty() {
            segments.push(SpeechSegment {
                start_secs: 0.0,
                end_secs: pcm_16khz.len() as f64 / SAMPLE_RATE as f64,
                samples: pcm_16khz.to_vec(),
            });
        }

        Ok(segments)
    }
}
