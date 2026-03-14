//! Voice Activity Detection engine using the `vad-rs` crate (Silero VAD v4).
//!
//! Chunks audio into speech segments based on voice activity,
//! used by the long-form transcription endpoint to split large
//! files into manageable pieces with accurate timestamps.

use anyhow::{Context, Result};
use tracing::warn;
use vad_rs::Vad;

const SAMPLE_RATE: usize = 16_000;
const FRAME_SIZE: usize = 30 * SAMPLE_RATE / 1000; // 480 samples = 30ms

const POSITIVE_THRESHOLD: f32 = 0.5;
const NEGATIVE_THRESHOLD: f32 = 0.35;
const MIN_SPEECH_FRAMES: usize = 3;
const REDEMPTION_FRAMES: usize = 20;

/// Maximum frames per chunk before resetting the ORT session.
/// 5 minutes of audio = 10,000 frames at 30ms each.
/// Keeps us well under the ~22,500-frame ORT corruption threshold.
const MAX_CHUNK_FRAMES: usize = 10_000;

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
    vad: Vad,
}

impl VadEngine {
    /// Load the Silero VAD ONNX model from `model_path`.
    pub fn new(model_path: &std::path::Path) -> Result<Self> {
        let vad = Vad::new(model_path, 16_000)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Failed to load VAD model")?;
        Ok(Self { vad })
    }

    /// Segment mono 16kHz audio into speech chunks.
    ///
    /// Uses Silero VAD to detect speech/silence transitions, processing
    /// 30ms frames (480 samples at 16kHz). Returns speech segments
    /// typically 5-15 seconds long with accurate timestamps.
    ///
    /// Long audio is processed in chunks with VAD resets between them
    /// to avoid ORT session corruption on extended runs.
    pub fn segment_audio(&mut self, pcm_16khz: &[f32]) -> Result<Vec<SpeechSegment>> {
        self.vad.reset();

        let mut segments = Vec::new();
        let mut speech_start: Option<usize> = None;
        let mut speech_frames: usize = 0;
        let mut redemption_count: usize = 0;
        let mut in_speech = false;
        let mut pos = 0;
        let mut frames_since_reset: usize = 0;

        while pos + FRAME_SIZE <= pcm_16khz.len() {
            // Reset the VAD periodically to avoid ORT session corruption.
            if frames_since_reset >= MAX_CHUNK_FRAMES {
                // Flush any in-progress speech segment before resetting.
                if in_speech {
                    if let Some(start) = speech_start.take() {
                        segments.push(SpeechSegment {
                            start_secs: start as f64 / SAMPLE_RATE as f64,
                            end_secs: pos as f64 / SAMPLE_RATE as f64,
                            samples: pcm_16khz[start..pos].to_vec(),
                        });
                    }
                    in_speech = false;
                    speech_frames = 0;
                    redemption_count = 0;
                }
                self.vad.reset();
                frames_since_reset = 0;
            }

            let frame = &pcm_16khz[pos..pos + FRAME_SIZE];

            let prob = match self.vad.compute(frame) {
                Ok(result) => result.prob,
                Err(e) => {
                    warn!(
                        "VAD inference failed at {:.1}s, resetting: {e}",
                        pos as f64 / SAMPLE_RATE as f64
                    );
                    // Flush any in-progress segment and reset.
                    if in_speech {
                        if let Some(start) = speech_start.take() {
                            segments.push(SpeechSegment {
                                start_secs: start as f64 / SAMPLE_RATE as f64,
                                end_secs: pos as f64 / SAMPLE_RATE as f64,
                                samples: pcm_16khz[start..pos].to_vec(),
                            });
                        }
                        in_speech = false;
                        speech_frames = 0;
                        redemption_count = 0;
                    }
                    self.vad.reset();
                    frames_since_reset = 0;
                    pos += FRAME_SIZE;
                    continue;
                }
            };

            frames_since_reset += 1;

            if !in_speech {
                if prob > POSITIVE_THRESHOLD {
                    in_speech = true;
                    speech_start = Some(pos);
                    speech_frames = 1;
                    redemption_count = 0;
                }
            } else {
                speech_frames += 1;

                if prob < NEGATIVE_THRESHOLD {
                    redemption_count += 1;
                    if redemption_count > REDEMPTION_FRAMES {
                        if speech_frames >= MIN_SPEECH_FRAMES {
                            if let Some(start) = speech_start.take() {
                                let end = pos + FRAME_SIZE;
                                segments.push(SpeechSegment {
                                    start_secs: start as f64 / SAMPLE_RATE as f64,
                                    end_secs: end as f64 / SAMPLE_RATE as f64,
                                    samples: pcm_16khz[start..end].to_vec(),
                                });
                            }
                        }
                        in_speech = false;
                        speech_frames = 0;
                        redemption_count = 0;
                    }
                } else {
                    redemption_count = 0;
                }
            }

            pos += FRAME_SIZE;
        }

        // Flush any remaining speech segment
        if in_speech {
            if let Some(start) = speech_start {
                segments.push(SpeechSegment {
                    start_secs: start as f64 / SAMPLE_RATE as f64,
                    end_secs: pcm_16khz.len() as f64 / SAMPLE_RATE as f64,
                    samples: pcm_16khz[start..].to_vec(),
                });
            }
        }

        // If VAD found no segments, treat entire audio as one
        if segments.is_empty() && !pcm_16khz.is_empty() {
            segments.push(SpeechSegment {
                start_secs: 0.0,
                end_secs: pcm_16khz.len() as f64 / SAMPLE_RATE as f64,
                samples: pcm_16khz.to_vec(),
            });
        }

        // Split any segments longer than 15s into sub-segments
        let max_samples = 15 * SAMPLE_RATE;
        let mut final_segments = Vec::new();
        for seg in segments {
            if seg.samples.len() > max_samples {
                let mut offset = 0;
                let base_start = seg.start_secs;
                while offset < seg.samples.len() {
                    let end = (offset + max_samples).min(seg.samples.len());
                    final_segments.push(SpeechSegment {
                        start_secs: base_start + offset as f64 / SAMPLE_RATE as f64,
                        end_secs: base_start + end as f64 / SAMPLE_RATE as f64,
                        samples: seg.samples[offset..end].to_vec(),
                    });
                    offset = end;
                }
            } else {
                final_segments.push(seg);
            }
        }

        Ok(final_segments)
    }
}
