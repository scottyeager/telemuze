use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};
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
#[derive(Clone)]
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
        let mut tmp = NamedTempFile::new().context("Failed to create PCM tempfile")?;
        let mut bytes = Vec::with_capacity(pcm.len() * 4);
        for sample in pcm {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        tmp.write_all(&bytes)
            .context("Failed to write PCM tempfile")?;
        tmp.flush().context("Failed to flush PCM tempfile")?;
        self.diarize_from_path(tmp.path())
    }

    /// Run diarization using an already-materialized raw f32-LE PCM tempfile.
    /// The caller owns the tempfile lifetime.
    pub fn diarize_from_path(&self, pcm_path: &Path) -> Result<Vec<DiarSegment>> {
        debug!(
            "Spawning {} on pcm path {}",
            self.binary_path.display(),
            pcm_path.display()
        );

        let output = Command::new(&self.binary_path)
            .arg("--model")
            .arg(&self.model_path)
            .arg("--pcm")
            .arg(pcm_path)
            .output()
            .with_context(|| {
                format!(
                    "Failed to spawn diarize binary: {}",
                    self.binary_path.display()
                )
            })?;

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

/// Group BPE tokens into words. Each returned `(start_idx, end_idx)` pair
/// is a half-open token range covering one word.
///
/// A word starts at any token whose first character is a word-boundary
/// marker. Different STT models use different conventions:
///   - SentencePiece (`parakeet-tdt-0.6b-v2`): `▁` (U+2581).
///   - Raw-space BPE (`parakeet-unified-en-0.6b`): a literal leading space.
/// We accept either so the same post-processing works for both models.
fn group_tokens_into_words(tokens: &[String]) -> Vec<(usize, usize)> {
    let mut words = Vec::new();
    if tokens.is_empty() {
        return words;
    }
    let mut start = 0;
    for i in 1..tokens.len() {
        if is_word_start(&tokens[i]) {
            words.push((start, i));
            start = i;
        }
    }
    words.push((start, tokens.len()));
    words
}

/// True if `token` begins with any recognized word-start marker.
fn is_word_start(token: &str) -> bool {
    token.starts_with('\u{2581}') || token.starts_with(' ')
}

/// Split ASR segments at diarization speaker-change boundaries, snapping
/// splits to whole-word boundaries.
///
/// This is a port of NVIDIA NeMo's `get_word_level_json_list`: a single
/// monotonically-advancing pointer (`turn_idx`) over diarization segments
/// sorted by start time. For each word, advance while the word's anchor
/// timestamp is past the current segment's end, then assign the word to
/// the current segment's speaker.
///
/// Because the pointer never moves backward, brief later-starting
/// overlapping segments inside a longer run are simply skipped — you stay
/// with the speaker who got there first until that speaker's run ends.
/// This is the property that prevents spurious mid-monologue speaker
/// flips on Sortformer's overlapping per-speaker activity output.
pub fn split_by_speakers(
    asr_segments: &[crate::state::TranscribedSegment],
    diar: &[DiarSegment],
) -> Vec<SpeakerSubSegment> {
    // Sort diarization segments by start time. parakeet-rs's `binarize()`
    // emits segments per-speaker, so the input may not be globally sorted.
    let mut diar_sorted: Vec<&DiarSegment> = diar.iter().collect();
    diar_sorted.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = Vec::new();
    let mut turn_idx: usize = 0;

    // Look up the speaker at time `t`, advancing turn_idx forward as needed.
    // Returns None only when the diarization segment list is empty.
    let mut speaker_at = |t: f64| -> Option<i32> {
        if diar_sorted.is_empty() {
            return None;
        }
        while turn_idx + 1 < diar_sorted.len()
            && t > diar_sorted[turn_idx].end as f64
        {
            turn_idx += 1;
        }
        Some(diar_sorted[turn_idx].speaker)
    };

    for seg in asr_segments {
        if seg.tokens.is_empty() {
            let mid = (seg.start_secs + seg.end_secs) * 0.5;
            out.push(SpeakerSubSegment {
                start: seg.start_secs,
                end: seg.end_secs,
                text: seg.text.clone(),
                tokens: vec![],
                token_timestamps: vec![],
                speaker: speaker_at(mid),
            });
            continue;
        }

        // Absolute per-token timestamps.
        let abs_ts: Vec<f64> = seg
            .token_timestamps
            .iter()
            .map(|&t| seg.start_secs + t as f64)
            .collect();

        // Group tokens into whole words and label each via the monotonic
        // turn_idx walk. The word's first-token timestamp is the anchor.
        let words = group_tokens_into_words(&seg.tokens);
        let labels: Vec<Option<i32>> = words
            .iter()
            .map(|&(ws, _)| speaker_at(abs_ts[ws]))
            .collect();

        // Group consecutive words with the same speaker into runs.
        let mut run_start_word = 0usize;
        let mut run_speaker = labels[0];

        for wi in 1..labels.len() {
            if labels[wi] != run_speaker {
                let tok_start = words[run_start_word].0;
                let tok_end = words[wi].0;
                let sub_start = abs_ts[tok_start];
                let sub_end = abs_ts[words[wi].0];
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
                run_speaker = labels[wi];
            }
        }

        // Final run.
        let tok_start = words[run_start_word].0;
        let tok_end = seg.tokens.len();
        let sub_start = abs_ts[tok_start];
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

/// Join BPE tokens into readable text. Handles both word-boundary
/// conventions: SentencePiece's `▁` marker (translated to a leading
/// space) and raw-space BPE (passed through unchanged).
fn join_tokens(tokens: &[String]) -> String {
    let mut s = String::new();
    for tok in tokens {
        s.push_str(&tok.replace('\u{2581}', " "));
    }
    s.trim().to_string()
}
