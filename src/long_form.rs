//! Shared long-form finalize + formatting logic used by both the HTTP
//! long-form endpoint and the Telegram bot's >60 s path.

use crate::engines::diarization::{split_by_speakers, DiarSegment, SpeakerSubSegment};
use crate::state::TranscribedSegment;

/// Result bundle returned from `AppState::long_form_transcribe`.
pub struct LongFormOutcome {
    pub asr_segments: Vec<TranscribedSegment>,
    pub diar_segments: Option<Vec<DiarSegment>>,
}

/// Merge ASR segments with diarization (if present) into speaker-labeled
/// sub-segments. When diarization is absent each ASR segment becomes one
/// sub-segment with `speaker = None`.
pub fn finalize(outcome: LongFormOutcome) -> Vec<SpeakerSubSegment> {
    if let Some(diar) = &outcome.diar_segments {
        split_by_speakers(&outcome.asr_segments, diar)
    } else {
        outcome
            .asr_segments
            .into_iter()
            .map(|s| SpeakerSubSegment {
                start: s.start_secs,
                end: s.end_secs,
                text: s.text,
                tokens: s.tokens,
                token_timestamps: s
                    .token_timestamps
                    .iter()
                    .map(|&t| s.start_secs + t as f64)
                    .collect(),
                speaker: None,
            })
            .collect()
    }
}

/// Render speaker sub-segments as a readable transcript with timestamps
/// and (when available) speaker labels.
///
/// - Diarized output: one paragraph per speaker run, prefixed with
///   `[0:00 — Speaker 1]`.
/// - Non-diarized multi-segment output: prefixed with `[0:00]`.
/// - Single non-diarized segment: plain text, no prefix (keeps short
///   voice-message output clean).
pub fn format_as_text(subs: &[SpeakerSubSegment]) -> String {
    if subs.is_empty() {
        return String::new();
    }

    let any_speaker = subs.iter().any(|s| s.speaker.is_some());

    if !any_speaker && subs.len() == 1 {
        return subs[0].text.trim().to_string();
    }

    let mut out = String::new();
    for sub in subs {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        let ts = format_timestamp(sub.start);
        match sub.speaker {
            Some(spk) => out.push_str(&format!("[{ts} — Speaker {}]\n", spk + 1)),
            None => out.push_str(&format!("[{ts}]\n")),
        }
        out.push_str(sub.text.trim());
    }
    out
}

/// Format a timestamp in seconds as `H:MM:SS` (with hours) or `M:SS`.
fn format_timestamp(secs: f64) -> String {
    let total = secs as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// Human-readable audio duration used in status messages.
pub fn format_duration(secs: f64) -> String {
    let total = secs as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}
