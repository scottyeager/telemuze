//! `telemuze transcribe` subcommand — one-shot long-form worker.
//!
//! Loads VAD + STT (and optionally Sortformer diarization), reads raw
//! f32-LE mono 16 kHz PCM from a tempfile, runs ASR and diarization
//! concurrently on the shared buffer, and prints a JSON
//! `{"segments":[...], "diarization":[...]}` payload on stdout. Exits
//! with code 0 on success.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use tracing::{error, info};

use crate::engines::sortformer::SortformerEngine;
use crate::engines::stt::SttEngine;
use crate::engines::vad::VadEngine;

const SAMPLE_RATE: f64 = 16_000.0;

#[derive(Parser, Debug)]
#[command(name = "telemuze transcribe")]
struct WorkerArgs {
    #[arg(long)]
    stt_model: PathBuf,

    #[arg(long)]
    vad_model: PathBuf,

    #[arg(long)]
    pcm: PathBuf,

    #[arg(long)]
    hotwords_file: Option<PathBuf>,

    #[arg(long, default_value_t = 1.5)]
    hotwords_score: f32,

    #[arg(long, default_value_t = 4)]
    max_active_paths: i32,

    #[arg(long, default_value_t = 0.0)]
    blank_penalty: f32,

    #[arg(long, default_value_t = 8)]
    num_threads: i32,

    /// Path to the NVIDIA Sortformer ONNX model. When set, the worker
    /// also runs diarization on the same PCM in parallel with ASR.
    #[arg(long)]
    diarize_model: Option<PathBuf>,

    #[arg(long, default_value_t = 2)]
    diarize_num_threads: i32,
}

#[derive(Serialize)]
struct OutSegment {
    start: f64,
    end: f64,
    text: String,
    tokens: Vec<String>,
    token_timestamps: Vec<f32>,
}

#[derive(Serialize)]
struct OutDiarSegment {
    start: f64,
    end: f64,
    speaker: i32,
}

#[derive(Serialize)]
struct Output {
    segments: Vec<OutSegment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diarization: Option<Vec<OutDiarSegment>>,
}

fn read_pcm(path: &std::path::Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)
        .with_context(|| format!("Failed to read PCM file: {}", path.display()))?;
    anyhow::ensure!(
        bytes.len() % 4 == 0,
        "PCM byte length {} is not a multiple of 4",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn run_asr(
    vad: &VadEngine,
    stt: &SttEngine,
    pcm: &[f32],
    hotwords: Option<&str>,
) -> Result<Vec<OutSegment>> {
    let segments = vad.segment_audio(pcm).context("VAD segmentation failed")?;
    info!("VAD found {} speech segments", segments.len());

    let mut out_segments = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        match stt.transcribe_with_hotwords(&seg.samples, hotwords) {
            Ok(result) if !result.text.trim().is_empty() => {
                info!(
                    "Segment {}/{}: [{:.1}s - {:.1}s] '{}'",
                    i + 1,
                    segments.len(),
                    seg.start_secs,
                    seg.end_secs,
                    result.text,
                );
                out_segments.push(OutSegment {
                    start: seg.start_secs,
                    end: seg.end_secs,
                    text: result.text,
                    tokens: result.tokens,
                    token_timestamps: result.timestamps,
                });
            }
            Ok(_) => {}
            Err(e) => {
                error!("STT failed for segment {}: {e}", i + 1);
            }
        }
    }
    Ok(out_segments)
}

pub fn run(argv: &[String]) -> Result<()> {
    let args = WorkerArgs::parse_from(
        std::iter::once("telemuze-transcribe".to_string()).chain(argv.iter().cloned()),
    );

    if !args.stt_model.is_dir() {
        anyhow::bail!("STT model directory not found: {}", args.stt_model.display());
    }
    if !args.vad_model.is_file() {
        anyhow::bail!("VAD model file not found: {}", args.vad_model.display());
    }
    if !args.pcm.is_file() {
        anyhow::bail!("PCM file not found: {}", args.pcm.display());
    }

    let pcm = read_pcm(&args.pcm)?;
    info!(
        "telemuze transcribe: loaded {} samples ({:.1}s)",
        pcm.len(),
        pcm.len() as f64 / SAMPLE_RATE
    );

    info!("Loading VAD model from {:?}...", args.vad_model);
    let vad = VadEngine::new(&args.vad_model).context("Failed to load VAD model")?;

    info!("Loading STT model from {:?}...", args.stt_model);
    let stt = SttEngine::new(
        &args.stt_model,
        args.hotwords_score,
        args.max_active_paths,
        args.blank_penalty,
        args.num_threads,
    )
    .context("Failed to load STT model")?;

    let sortformer = match args.diarize_model.as_ref() {
        Some(path) => {
            if !path.is_file() {
                anyhow::bail!("Sortformer model file not found: {}", path.display());
            }
            info!("Loading Sortformer diarization model from {:?}...", path);
            Some(
                SortformerEngine::new(path, args.diarize_num_threads)
                    .context("Failed to load Sortformer model")?,
            )
        }
        None => None,
    };

    let hotwords = if let Some(path) = &args.hotwords_file {
        let s = fs::read_to_string(path)
            .with_context(|| format!("Failed to read hotwords file: {}", path.display()))?;
        Some(s)
    } else {
        None
    };

    let (asr_result, diar_result) = std::thread::scope(|scope| {
        let pcm_ref = &pcm;
        let vad_ref = &vad;
        let stt_ref = &stt;
        let hotwords_ref = hotwords.as_deref();

        let asr_handle =
            scope.spawn(move || run_asr(vad_ref, stt_ref, pcm_ref, hotwords_ref));

        let diar_handle = sortformer
            .as_ref()
            .map(|sf| scope.spawn(move || sf.diarize(pcm_ref)));

        let asr = asr_handle
            .join()
            .map_err(|_| anyhow::anyhow!("ASR thread panicked"))
            .and_then(|r| r);

        let diar = diar_handle.map(|h| {
            h.join()
                .map_err(|_| anyhow::anyhow!("Diarization thread panicked"))
                .and_then(|r| r)
        });

        (asr, diar)
    });

    let out_segments = asr_result?;

    let diarization = match diar_result {
        Some(Ok(segs)) => {
            let n_spk = segs.iter().map(|s| s.speaker).max().map(|m| m + 1).unwrap_or(0);
            info!("Diarization: {} segments, {} speakers", segs.len(), n_spk);
            Some(
                segs.into_iter()
                    .map(|s| OutDiarSegment {
                        start: s.start as f64,
                        end: s.end as f64,
                        speaker: s.speaker,
                    })
                    .collect(),
            )
        }
        Some(Err(e)) => {
            error!("Diarization failed: {e}");
            None
        }
        None => None,
    };

    let out = Output {
        segments: out_segments,
        diarization,
    };
    let json = serde_json::to_string(&out).context("Failed to serialize output")?;
    println!("{json}");
    Ok(())
}
