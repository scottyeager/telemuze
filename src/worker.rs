//! `telemuze transcribe` subcommand — one-shot long-form worker.
//!
//! Loads VAD + STT, reads raw f32-LE mono 16 kHz PCM from a tempfile,
//! segments via VAD, transcribes each segment, and prints a JSON
//! `{"segments":[...]}` payload on stdout. Exits with code 0 on success.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use tracing::{error, info};

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
struct Output {
    segments: Vec<OutSegment>,
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

    let hotwords = if let Some(path) = &args.hotwords_file {
        let s = fs::read_to_string(path)
            .with_context(|| format!("Failed to read hotwords file: {}", path.display()))?;
        Some(s)
    } else {
        None
    };

    let segments = vad.segment_audio(&pcm).context("VAD segmentation failed")?;
    info!("VAD found {} speech segments", segments.len());

    let mut out_segments = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        match stt.transcribe_with_hotwords(&seg.samples, hotwords.as_deref()) {
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

    let out = Output {
        segments: out_segments,
    };
    let json = serde_json::to_string(&out).context("Failed to serialize output")?;
    println!("{json}");
    Ok(())
}
