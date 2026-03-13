//! Pure-Rust audio pipeline.
//!
//! Handles demuxing, decoding, resampling, and mono mixdown for any
//! audio or video file. Uses symphonia for demux/decode and rubato
//! for sample-rate conversion. No FFmpeg required.

use anyhow::{Context, Result};
use rubato::{FftFixedIn, Resampler};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::debug;

/// Target sample rate for all STT models.
const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Decode any audio/video file into mono f32 PCM at 16kHz.
///
/// Pipeline:
/// 1. Symphonia probes the byte stream and selects the first audio track
/// 2. Decodes compressed audio to raw PCM frames
/// 3. Rubato resamples to exactly 16kHz
/// 4. Multi-channel audio is mixed down to mono
pub fn decode_to_pcm(data: &[u8]) -> Result<Vec<f32>> {
    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .context("Failed to probe audio format (unsupported container?)")?;

    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .context("No audio track found in file")?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let source_sample_rate = codec_params
        .sample_rate
        .context("Audio track has no sample rate")?;
    let channels = codec_params.channels.map(|c| c.count()).unwrap_or(1);

    debug!(
        "Audio track: {}Hz, {} channel(s)",
        source_sample_rate, channels
    );

    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &decoder_opts)
        .context("Failed to create audio decoder")?;

    // Decode all packets into interleaved f32 samples
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };

        let spec = *decoded.spec();
        let num_frames = decoded.capacity();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(sample_buf.samples());
    }

    if all_samples.is_empty() {
        anyhow::bail!("No audio samples decoded from file");
    }

    // Mixdown to mono if multi-channel
    let mono_samples = if channels > 1 {
        mixdown_to_mono(&all_samples, channels)
    } else {
        all_samples
    };

    // Resample to 16kHz if needed
    if source_sample_rate == TARGET_SAMPLE_RATE {
        debug!("Audio already at 16kHz, skipping resample");
        Ok(mono_samples)
    } else {
        debug!(
            "Resampling from {}Hz to {}Hz",
            source_sample_rate, TARGET_SAMPLE_RATE
        );
        resample(&mono_samples, source_sample_rate, TARGET_SAMPLE_RATE)
    }
}

/// Average multi-channel interleaved samples down to mono.
fn mixdown_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    let inv = 1.0 / channels as f32;
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * inv)
        .collect()
}

/// Resample a mono f32 signal using rubato's FFT-based resampler.
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    let chunk_size = 1024;

    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        1, // sub-chunks
        1, // mono
    )
    .context("Failed to create resampler")?;

    let mut output = Vec::new();

    // Process in chunks
    let mut pos = 0;
    while pos + chunk_size <= input.len() {
        let chunk = &input[pos..pos + chunk_size];
        let result = resampler
            .process(&[chunk], None)
            .context("Resampling failed")?;
        output.extend_from_slice(&result[0]);
        pos += chunk_size;
    }

    // Handle remaining samples by zero-padding
    if pos < input.len() {
        let remaining = &input[pos..];
        let mut padded = remaining.to_vec();
        padded.resize(chunk_size, 0.0);
        let result = resampler
            .process(&[&padded], None)
            .context("Resampling tail failed")?;
        // Only take proportional output
        let expected = ((input.len() - pos) as f64 * to_rate as f64 / from_rate as f64) as usize;
        let take = expected.min(result[0].len());
        output.extend_from_slice(&result[0][..take]);
    }

    Ok(output)
}
