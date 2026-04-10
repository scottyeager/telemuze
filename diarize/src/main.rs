//! `telemuze-diarize` — on-demand speaker diarization subprocess.
//!
//! Loads NVIDIA's Sortformer v2 ONNX model, runs offline diarization on a
//! mono 16 kHz f32-LE PCM file, prints `{"segments": [...]}` to stdout, and
//! exits. Designed to be invoked once per long-form transcription request
//! by the main `telemuze` server, so the 492 MB model isn't permanently
//! resident.

use anyhow::{Context, Result};
use clap::Parser;
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

const SAMPLE_RATE: u32 = 16_000;

#[derive(Parser, Debug)]
#[command(name = "telemuze-diarize", version, about)]
struct Args {
    /// Path to the Sortformer ONNX model file.
    #[arg(long)]
    model: PathBuf,

    /// Path to a raw PCM file: mono 16 kHz f32 little-endian samples,
    /// no header.
    #[arg(long)]
    pcm: PathBuf,
}

#[derive(Serialize)]
struct OutSegment {
    start: f64,
    end: f64,
    speaker: i32,
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

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.model.is_file() {
        anyhow::bail!("Model file not found: {}", args.model.display());
    }
    if !args.pcm.is_file() {
        anyhow::bail!("PCM file not found: {}", args.pcm.display());
    }

    let pcm = read_pcm(&args.pcm)?;
    eprintln!(
        "telemuze-diarize: loaded {} samples ({:.1}s at {} Hz)",
        pcm.len(),
        pcm.len() as f32 / SAMPLE_RATE as f32,
        SAMPLE_RATE
    );

    let mut sortformer = Sortformer::with_config(&args.model, None, DiarizationConfig::callhome())
        .with_context(|| format!("Failed to load Sortformer model: {}", args.model.display()))?;

    let segs = sortformer
        .diarize(pcm, SAMPLE_RATE, 1)
        .context("Sortformer inference failed")?;

    eprintln!("telemuze-diarize: produced {} segments", segs.len());

    let out = Output {
        segments: segs
            .into_iter()
            .map(|s| OutSegment {
                start: s.start as f64 / SAMPLE_RATE as f64,
                end: s.end as f64 / SAMPLE_RATE as f64,
                speaker: s.speaker_id as i32,
            })
            .collect(),
    };

    let json = serde_json::to_string(&out).context("Failed to serialize output")?;
    println!("{json}");
    Ok(())
}
