//! Long-form transcription engine: spawns a one-shot `telemuze transcribe`
//! worker subprocess that loads VAD + STT (and optionally Sortformer
//! diarization), processes a raw PCM tempfile, and exits. The always-on
//! server never pays the worker's memory cost.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::NamedTempFile;
use tracing::debug;

use crate::engines::diarization::DiarSegment;
use crate::state::TranscribedSegment;

#[derive(Deserialize)]
struct WireSegment {
    start: f64,
    end: f64,
    text: String,
    tokens: Vec<String>,
    token_timestamps: Vec<f32>,
}

#[derive(Deserialize)]
struct WireDiarSegment {
    start: f64,
    end: f64,
    speaker: i32,
}

#[derive(Deserialize)]
struct WireOutput {
    segments: Vec<WireSegment>,
    #[serde(default)]
    diarization: Option<Vec<WireDiarSegment>>,
}

/// Result of a single long-form worker invocation — ASR segments plus
/// optional diarization from the same subprocess.
pub struct LongFormResult {
    pub segments: Vec<TranscribedSegment>,
    pub diar: Option<Vec<DiarSegment>>,
}

#[derive(Clone)]
pub struct LongFormEngine {
    binary_path: PathBuf,
    stt_model_dir: PathBuf,
    vad_model_path: PathBuf,
    diarize_model_path: Option<PathBuf>,
    hotwords_score: f32,
    max_active_paths: i32,
    blank_penalty: f32,
    num_threads: i32,
}

impl LongFormEngine {
    pub fn new(
        binary_path: PathBuf,
        stt_model_dir: PathBuf,
        vad_model_path: PathBuf,
        diarize_model_path: Option<PathBuf>,
        hotwords_score: f32,
        max_active_paths: i32,
        blank_penalty: f32,
        num_threads: i32,
    ) -> Self {
        Self {
            binary_path,
            stt_model_dir,
            vad_model_path,
            diarize_model_path,
            hotwords_score,
            max_active_paths,
            blank_penalty,
            num_threads,
        }
    }

    /// Run the long-form worker on an already-materialized raw f32-LE PCM
    /// tempfile. Returns ASR segments and, when a diarization model is
    /// configured, the matching speaker segments.
    pub fn transcribe_and_diarize(
        &self,
        pcm_path: &Path,
        hotwords: Option<&str>,
    ) -> Result<LongFormResult> {
        let hotwords_tmp = match hotwords {
            Some(hw) if !hw.is_empty() => {
                let mut tmp = NamedTempFile::new().context("Failed to create hotwords tempfile")?;
                tmp.write_all(hw.as_bytes())
                    .context("Failed to write hotwords tempfile")?;
                tmp.flush().context("Failed to flush hotwords tempfile")?;
                Some(tmp)
            }
            _ => None,
        };

        debug!(
            "Spawning long-form worker {} on pcm path {}",
            self.binary_path.display(),
            pcm_path.display()
        );

        let mut cmd = Command::new(&self.binary_path);
        cmd.arg("transcribe")
            .arg("--stt-model")
            .arg(&self.stt_model_dir)
            .arg("--vad-model")
            .arg(&self.vad_model_path)
            .arg("--pcm")
            .arg(pcm_path)
            .arg("--hotwords-score")
            .arg(self.hotwords_score.to_string())
            .arg("--max-active-paths")
            .arg(self.max_active_paths.to_string())
            .arg("--blank-penalty")
            .arg(self.blank_penalty.to_string())
            .arg("--num-threads")
            .arg(self.num_threads.to_string());

        if let Some(ref path) = self.diarize_model_path {
            cmd.arg("--diarize-model").arg(path);
        }

        if let Some(ref tmp) = hotwords_tmp {
            cmd.arg("--hotwords-file").arg(tmp.path());
        }

        let output = cmd.output().with_context(|| {
            format!(
                "Failed to spawn long-form worker: {}",
                self.binary_path.display()
            )
        })?;

        if !output.stderr.is_empty() {
            for line in String::from_utf8_lossy(&output.stderr).lines() {
                debug!("transcribe: {}", line);
            }
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "telemuze transcribe exited with status {}: {}",
                output.status,
                stderr.trim()
            );
        }

        let stdout = std::str::from_utf8(&output.stdout)
            .context("telemuze transcribe stdout was not valid UTF-8")?;
        let parsed: WireOutput = serde_json::from_str(stdout.trim())
            .context("Failed to parse telemuze transcribe JSON output")?;

        drop(hotwords_tmp);

        let segments = parsed
            .segments
            .into_iter()
            .map(|s| TranscribedSegment {
                start_secs: s.start,
                end_secs: s.end,
                text: s.text,
                tokens: s.tokens,
                token_timestamps: s.token_timestamps,
            })
            .collect();

        let diar = parsed.diarization.map(|segs| {
            segs.into_iter()
                .map(|s| DiarSegment {
                    start: s.start as f32,
                    end: s.end as f32,
                    speaker: s.speaker,
                })
                .collect()
        });

        Ok(LongFormResult { segments, diar })
    }
}
