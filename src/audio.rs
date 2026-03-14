//! Audio/video decoding via FFmpeg.
//!
//! Shells out to `ffmpeg` to decode any audio or video file into
//! mono 16kHz f32 PCM suitable for speech-to-text models.

use anyhow::{Context, Result};
use std::io::Write;
use std::process::{Command, Stdio};
use tracing::debug;

/// Decode any audio/video file into mono f32 PCM at 16kHz.
///
/// Pipes raw bytes into `ffmpeg` which handles all demuxing, decoding,
/// resampling, and channel mixdown in a single pass.
pub fn decode_to_pcm(data: &[u8]) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-i", "pipe:0",       // read from stdin
            "-f", "f32le",        // output raw 32-bit float little-endian
            "-acodec", "pcm_f32le",
            "-ar", "16000",       // resample to 16kHz
            "-ac", "1",           // mixdown to mono
            "-v", "error",        // suppress banner noise
            "pipe:1",             // write to stdout
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn ffmpeg — is it installed?")?;

    // Write input data to ffmpeg's stdin
    child
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(data)
        .context("Failed to write to ffmpeg stdin")?;

    let output = child
        .wait_with_output()
        .context("Failed to wait for ffmpeg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg failed: {stderr}");
    }

    let raw = &output.stdout;
    if raw.is_empty() {
        anyhow::bail!("ffmpeg produced no audio output — file may contain no audio track");
    }

    debug!(
        "ffmpeg decoded {} bytes of f32le PCM ({:.1}s at 16kHz)",
        raw.len(),
        raw.len() as f64 / (16_000.0 * 4.0)
    );

    // Reinterpret the raw bytes as f32 samples (little-endian)
    let samples: Vec<f32> = raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(samples)
}
