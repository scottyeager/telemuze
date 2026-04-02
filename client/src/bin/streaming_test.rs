// Test binary for sherpa-onnx streaming (online) recognition from live mic.
//
// Tests the Nemotron streaming transducer model with endpoint detection.
// The streaming latency (chunk size) is baked into the model at export time.
// Nemotron supports 80ms, 160ms, 560ms, and 1120ms chunk sizes — our current
// export uses 1120ms (chunk_shift=112 frames at 10ms/frame).
//
// Usage:
//   cargo run --bin streaming-test
//   cargo run --bin streaming-test -- --model-dir /path/to/model --no-int8
//
// Model dir defaults to ~/.local/share/telemuze/models/nemotron-speech-streaming-en-0.6b

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;
use tracing::{debug, info};

const SAMPLE_RATE: u32 = 16000;

#[derive(Parser)]
#[command(
    name = "streaming-test",
    about = "Test sherpa-onnx streaming recognition from live mic"
)]
struct Args {
    /// Path to model directory containing encoder, decoder, joiner, tokens.
    /// Defaults to ~/.local/share/telemuze/models/nemotron-speech-streaming-en-0.6b
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Use full-precision models instead of int8 quantized
    #[arg(long)]
    no_int8: bool,

    /// Feature dimension (Nemotron uses 128, most others use 80)
    #[arg(long, default_value_t = 128)]
    feat_dim: i32,

    /// Number of model threads
    #[arg(long, default_value_t = 2)]
    num_threads: i32,

    /// Min trailing silence (seconds) to trigger endpoint without speech
    #[arg(long, default_value_t = 2.4)]
    rule1_silence: f32,

    /// Min trailing silence (seconds) to trigger endpoint after speech
    #[arg(long, default_value_t = 1.2)]
    rule2_silence: f32,

    /// Max utterance length (seconds) before forced endpoint
    #[arg(long, default_value_t = 30.0)]
    rule3_max_length: f32,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn find_model_files(dir: &PathBuf, int8: bool) -> Result<(String, String, String, String)> {
    let suffix = if int8 { ".int8.onnx" } else { ".onnx" };

    let encoder = dir.join(format!("encoder{suffix}"));
    let decoder = dir.join(format!("decoder{suffix}"));
    let joiner = dir.join(format!("joiner{suffix}"));
    let tokens = dir.join("tokens.txt");

    anyhow::ensure!(encoder.exists(), "encoder not found: {}", encoder.display());
    anyhow::ensure!(decoder.exists(), "decoder not found: {}", decoder.display());
    anyhow::ensure!(joiner.exists(), "joiner not found: {}", joiner.display());
    anyhow::ensure!(tokens.exists(), "tokens not found: {}", tokens.display());

    Ok((
        encoder.to_string_lossy().into_owned(),
        decoder.to_string_lossy().into_owned(),
        joiner.to_string_lossy().into_owned(),
        tokens.to_string_lossy().into_owned(),
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let log_level = if args.verbose {
        "streaming_test=debug"
    } else {
        "streaming_test=info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .init();

    let int8 = !args.no_int8;
    let model_dir = args.model_dir.unwrap_or_else(|| {
        dirs_next::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("telemuze/models/nemotron-speech-streaming-en-0.6b")
    });
    let (encoder, decoder, joiner, tokens) = find_model_files(&model_dir, int8)?;
    info!("Encoder: {encoder}");
    info!("Decoder: {decoder}");
    info!("Joiner:  {joiner}");
    info!("Quantized: {int8}");

    // Set up audio capture BEFORE model init (onnxruntime can interfere with ALSA).
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("No input audio device available")?;
    debug!("Audio device: {}", device.name().unwrap_or_default());

    let audio_config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let (audio_tx, audio_rx) = mpsc::sync_channel::<Vec<f32>>(200);

    let audio_stream = device
        .build_input_stream(
            &audio_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let _ = audio_tx.try_send(data.to_vec());
            },
            |err| eprintln!("Audio stream error: {err}"),
            None,
        )
        .context("Failed to build audio input stream")?;

    audio_stream.play().context("Failed to start audio stream")?;

    // Build recognizer config.
    let mut config = sherpa_onnx::OnlineRecognizerConfig::default();
    config.model_config.transducer.encoder = Some(encoder);
    config.model_config.transducer.decoder = Some(decoder);
    config.model_config.transducer.joiner = Some(joiner);
    config.model_config.tokens = Some(tokens);
    config.model_config.num_threads = args.num_threads;
    config.model_config.provider = Some("cpu".into());
    config.model_config.debug = args.verbose;

    config.feat_config.feature_dim = args.feat_dim;

    config.decoding_method = Some("greedy_search".into());
    config.enable_endpoint = true;
    config.rule1_min_trailing_silence = args.rule1_silence;
    config.rule2_min_trailing_silence = args.rule2_silence;
    config.rule3_min_utterance_length = args.rule3_max_length;

    info!("Creating online recognizer (this may take a moment)...");
    let recognizer =
        sherpa_onnx::OnlineRecognizer::create(&config).context("Failed to create recognizer")?;

    let stream = recognizer.create_stream();

    eprintln!("Listening (feat_dim={}, greedy_search)... Ctrl+C to stop", args.feat_dim);

    // Ctrl+C handler.
    let running = Arc::new(AtomicBool::new(true));
    signal_hook::flag::register_conditional_default(signal_hook::consts::SIGINT, running.clone())?;
    signal_hook::flag::register_conditional_default(signal_hook::consts::SIGTERM, running.clone())?;

    let mut total_samples = 0u64;
    let mut last_text = String::new();
    let mut segment_idx = 0u32;

    while running.load(Ordering::Relaxed) {
        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                total_samples += samples.len() as u64;
                stream.accept_waveform(SAMPLE_RATE as i32, &samples);

                while recognizer.is_ready(&stream) {
                    recognizer.decode(&stream);

                    if let Some(result) = recognizer.get_result(&stream) {
                        let text = result.text.trim().to_string();
                        if !text.is_empty() && text != last_text {
                            let secs = total_samples as f64 / SAMPLE_RATE as f64;
                            eprint!("\r\x1b[K[{secs:.1}s] #{segment_idx}: {text}");
                            last_text = text;
                        }
                    }

                    if recognizer.is_endpoint(&stream) {
                        if !last_text.is_empty() {
                            let secs = total_samples as f64 / SAMPLE_RATE as f64;
                            eprintln!("\r\x1b[K[{secs:.1}s] #{segment_idx}: {last_text}");
                            segment_idx += 1;
                            last_text.clear();
                        }
                        recognizer.reset(&stream);
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Flush any remaining text.
    if !last_text.is_empty() {
        let secs = total_samples as f64 / SAMPLE_RATE as f64;
        eprintln!("\r\x1b[K[{secs:.1}s] #{segment_idx}: {last_text}");
    }

    drop(audio_stream);
    info!("Shutting down");
    std::process::exit(0);
}
