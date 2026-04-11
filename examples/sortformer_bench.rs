//! Sortformer diarization benchmark via sherpa-onnx.
//!
//! Runs the same `OfflineSortformerDiarization` sherpa-onnx engine the
//! long-form worker uses, on a mono 16 kHz f32-LE PCM file, and prints
//! segments as JSON to stdout (matching the format of `telemuze-diarize`
//! and `nemo_diarize.py` so the existing `der_compare.py` tool can score
//! it).
//!
//! Usage:
//!   cargo run --release --example sortformer_bench -- \
//!       --model ~/.local/share/telemuze/models/diar_streaming_sortformer_4spk-v2.1.onnx \
//!       --pcm ~/media/vid-chat-demo.pcm > segments.json

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Serialize;
use sherpa_onnx::{
    OfflineSortformerDiarization, OfflineSortformerDiarizationConfig,
    OfflineSortformerDiarizationModelConfig,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const SAMPLE_RATE: u32 = 16_000;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the Sortformer ONNX model file.
    #[arg(long)]
    model: PathBuf,

    /// Path to a raw PCM file: mono 16 kHz f32 little-endian samples, no header.
    #[arg(long)]
    pcm: PathBuf,

    /// Number of ONNX threads.
    #[arg(long, default_value_t = 4)]
    threads: i32,
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

    let pcm = read_pcm(&args.pcm)?;
    eprintln!(
        "sortformer_bench: loaded {} samples ({:.1}s at {} Hz)",
        pcm.len(),
        pcm.len() as f32 / SAMPLE_RATE as f32,
        SAMPLE_RATE
    );

    let load_start = Instant::now();
    let config = OfflineSortformerDiarizationConfig {
        model: OfflineSortformerDiarizationModelConfig {
            model: Some(args.model.to_string_lossy().into_owned()),
            num_threads: args.threads,
            ..Default::default()
        },
        ..Default::default()
    };
    let sd = OfflineSortformerDiarization::create(&config)
        .ok_or_else(|| anyhow!("Failed to load Sortformer from {}", args.model.display()))?;
    eprintln!(
        "sortformer_bench: model loaded in {:.2}s ({} speakers max)",
        load_start.elapsed().as_secs_f32(),
        sd.num_speakers()
    );

    let t0 = Instant::now();
    let result = sd
        .process(&pcm)
        .ok_or_else(|| anyhow!("Sortformer process() returned null"))?;
    let segs = result.sort_by_start_time();
    let inference_s = t0.elapsed().as_secs_f32();

    eprintln!(
        "sortformer_bench: produced {} segments in {:.2}s (RTF {:.4})",
        segs.len(),
        inference_s,
        inference_s / (pcm.len() as f32 / SAMPLE_RATE as f32),
    );

    let out = Output {
        segments: segs
            .into_iter()
            .map(|s| OutSegment {
                start: s.start as f64,
                end: s.end as f64,
                speaker: s.speaker,
            })
            .collect(),
    };

    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
