use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use tempfile::NamedTempFile;
use tracing::debug;

/// A single diarization segment with a 0-based speaker label.
pub struct DiarSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: i32,
}

/// Wire format for a single segment as emitted by the
/// `telemuze-diarize` subprocess on stdout.
#[derive(Deserialize)]
struct WireSegment {
    start: f64,
    end: f64,
    speaker: i32,
}

#[derive(Deserialize)]
struct WireOutput {
    segments: Vec<WireSegment>,
}

/// Diarization engine that delegates to a one-shot `telemuze-diarize`
/// subprocess. The subprocess loads the Sortformer ONNX model, runs
/// inference on a raw PCM tempfile, prints JSON to stdout, and exits —
/// keeping the 492 MB model out of the always-on server's address space.
pub struct DiarizationEngine {
    binary_path: PathBuf,
    model_path: PathBuf,
}

impl DiarizationEngine {
    pub fn new(binary_path: PathBuf, model_path: PathBuf) -> Self {
        Self {
            binary_path,
            model_path,
        }
    }

    /// Run diarization on mono 16 kHz f32 PCM. Spawns the diarize binary
    /// once and returns when it exits.
    pub fn diarize(&self, pcm: &[f32]) -> Result<Vec<DiarSegment>> {
        // 1. Materialize PCM as a tempfile of raw f32-LE bytes.
        let mut tmp = NamedTempFile::new().context("Failed to create PCM tempfile")?;
        let mut bytes = Vec::with_capacity(pcm.len() * 4);
        for sample in pcm {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        tmp.write_all(&bytes)
            .context("Failed to write PCM tempfile")?;
        tmp.flush().context("Failed to flush PCM tempfile")?;

        debug!(
            "Spawning {} for {} samples ({:.2}s)",
            self.binary_path.display(),
            pcm.len(),
            pcm.len() as f64 / 16_000.0
        );

        // 2. Spawn the subprocess and read it to completion.
        let output = Command::new(&self.binary_path)
            .arg("--model")
            .arg(&self.model_path)
            .arg("--pcm")
            .arg(tmp.path())
            .output()
            .with_context(|| {
                format!(
                    "Failed to spawn diarize binary: {}",
                    self.binary_path.display()
                )
            })?;

        // Forward subprocess stderr into our tracing pipeline so it lands
        // in journalctl alongside everything else.
        if !output.stderr.is_empty() {
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                debug!("diarize: {}", line);
            }
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "telemuze-diarize exited with status {}: {}",
                output.status,
                stderr.trim()
            );
        }

        // 3. Parse stdout as JSON.
        let stdout = std::str::from_utf8(&output.stdout)
            .context("telemuze-diarize stdout was not valid UTF-8")?;
        let parsed: WireOutput = serde_json::from_str(stdout.trim())
            .context("Failed to parse telemuze-diarize JSON output")?;

        Ok(parsed
            .segments
            .into_iter()
            .map(|s| DiarSegment {
                start: s.start as f32,
                end: s.end as f32,
                speaker: s.speaker,
            })
            .collect())
    }
}

/// A sub-segment produced by splitting an ASR segment at diarization
/// speaker-change boundaries using token timestamps.
pub struct SpeakerSubSegment {
    /// Absolute start time in seconds.
    pub start: f64,
    /// Absolute end time in seconds.
    pub end: f64,
    /// Concatenated token text for this sub-segment.
    pub text: String,
    /// Tokens belonging to this sub-segment.
    pub tokens: Vec<String>,
    /// Absolute per-token timestamps.
    pub token_timestamps: Vec<f64>,
    /// Speaker index (0-based), or `None` if no diarization segment overlaps.
    pub speaker: Option<i32>,
}

/// Find the dominant speaker over the time range `[start, end)`.
///
/// The pyannote diarization output contains overlapping segments — long
/// background "main turn" segments often contain shorter, more specific
/// interjection segments from other speakers nested inside. Naive
/// largest-overlap selection always picks the background, ignoring the
/// interjections. Instead we score each candidate as
/// `overlap / segment_duration`, which rewards specific (short) segments
/// that tightly cover the word and penalizes diffuse background ones.
/// Ties are broken by raw overlap amount.
///
/// If no segment overlaps, falls back to the nearest segment by time
/// distance to the word's midpoint.
fn dominant_speaker(start: f64, end: f64, diar: &[DiarSegment]) -> Option<i32> {
    if diar.is_empty() {
        return None;
    }
    let mut best_score = 0.0f64;
    let mut best_overlap = 0.0f64;
    let mut best_speaker: Option<i32> = None;
    for seg in diar {
        let s = seg.start as f64;
        let e = seg.end as f64;
        let ov = end.min(e) - start.max(s);
        if ov <= 0.0 {
            continue;
        }
        let seg_dur = (e - s).max(1e-6);
        let score = ov / seg_dur;
        if score > best_score || (score == best_score && ov > best_overlap) {
            best_score = score;
            best_overlap = ov;
            best_speaker = Some(seg.speaker);
        }
    }
    if best_speaker.is_some() {
        return best_speaker;
    }

    // Fallback: nearest segment to the midpoint.
    let mid = (start + end) * 0.5;
    let mut best_dist = f64::INFINITY;
    for seg in diar {
        let s = seg.start as f64;
        let e = seg.end as f64;
        let dist = if mid < s {
            s - mid
        } else if mid >= e {
            mid - e
        } else {
            0.0
        };
        if dist < best_dist {
            best_dist = dist;
            best_speaker = Some(seg.speaker);
        }
    }
    best_speaker
}

/// Group BPE tokens into words. Each returned `(start_idx, end_idx)` pair
/// is a half-open token range covering one word, where the first token
/// begins with the SentencePiece marker `▁` (or is the very first token).
fn group_tokens_into_words(tokens: &[String]) -> Vec<(usize, usize)> {
    let mut words = Vec::new();
    if tokens.is_empty() {
        return words;
    }
    let mut start = 0;
    for i in 1..tokens.len() {
        if tokens[i].starts_with('\u{2581}') {
            words.push((start, i));
            start = i;
        }
    }
    words.push((start, tokens.len()));
    words
}

/// Split ASR segments at diarization speaker-change boundaries, snapping
/// splits to whole-word boundaries derived from BPE token markers.
pub fn split_by_speakers(
    asr_segments: &[crate::state::TranscribedSegment],
    diar: &[DiarSegment],
) -> Vec<SpeakerSubSegment> {
    let mut out = Vec::new();

    for seg in asr_segments {
        if seg.tokens.is_empty() {
            out.push(SpeakerSubSegment {
                start: seg.start_secs,
                end: seg.end_secs,
                text: seg.text.clone(),
                tokens: vec![],
                token_timestamps: vec![],
                speaker: dominant_speaker(seg.start_secs, seg.end_secs, diar),
            });
            continue;
        }

        // Absolute per-token timestamps.
        let abs_ts: Vec<f64> = seg
            .token_timestamps
            .iter()
            .map(|&t| seg.start_secs + t as f64)
            .collect();

        // Group tokens into whole words.
        let words = group_tokens_into_words(&seg.tokens);

        // Compute (start, end, speaker) for each word.
        let mut word_info: Vec<(f64, f64, Option<i32>)> = Vec::with_capacity(words.len());
        for (wi, &(ws, _we)) in words.iter().enumerate() {
            let w_start = abs_ts[ws];
            let w_end = if wi + 1 < words.len() {
                abs_ts[words[wi + 1].0]
            } else {
                seg.end_secs
            };
            let spk = dominant_speaker(w_start, w_end, diar);
            word_info.push((w_start, w_end, spk));
        }

        // Group consecutive words with the same speaker into runs.
        let mut run_start_word = 0usize;
        let mut run_speaker = word_info[0].2;

        for wi in 1..word_info.len() {
            if word_info[wi].2 != run_speaker {
                // Emit run [run_start_word .. wi).
                let tok_start = words[run_start_word].0;
                let tok_end = words[wi].0;
                let sub_start = word_info[run_start_word].0;
                let sub_end = word_info[wi].0; // = next word's start
                let toks = seg.tokens[tok_start..tok_end].to_vec();
                let ts = abs_ts[tok_start..tok_end].to_vec();
                let text = join_tokens(&toks);
                out.push(SpeakerSubSegment {
                    start: sub_start,
                    end: sub_end,
                    text,
                    tokens: toks,
                    token_timestamps: ts,
                    speaker: run_speaker,
                });
                run_start_word = wi;
                run_speaker = word_info[wi].2;
            }
        }

        // Emit final run.
        let tok_start = words[run_start_word].0;
        let tok_end = seg.tokens.len();
        let sub_start = word_info[run_start_word].0;
        let sub_end = seg.end_secs;
        let toks = seg.tokens[tok_start..tok_end].to_vec();
        let ts = abs_ts[tok_start..tok_end].to_vec();
        let text = join_tokens(&toks);
        out.push(SpeakerSubSegment {
            start: sub_start,
            end: sub_end,
            text,
            tokens: toks,
            token_timestamps: ts,
            speaker: run_speaker,
        });
    }

    out
}

/// Join SentencePiece BPE tokens into readable text. The `▁` marker
/// indicates a word boundary and is rendered as a leading space.
fn join_tokens(tokens: &[String]) -> String {
    let mut s = String::new();
    for tok in tokens {
        s.push_str(&tok.replace('\u{2581}', " "));
    }
    s.trim().to_string()
}
