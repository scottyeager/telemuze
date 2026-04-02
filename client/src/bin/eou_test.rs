// Test binary for parakeet_realtime_eou_120m-v1 end-of-utterance detection.
//
// This model is a cache-aware streaming FastConformer-RNNT that emits an <EOU>
// token when it detects the speaker has finished their utterance. Sherpa-onnx
// handles encoder cache management internally (window_size=25, chunk_shift=16,
// 70-frame attention context), so we just feed audio and decode continuously.
//
// Usage:
//   cargo run --bin eou-test
//   cargo run --bin eou-test -- -v
//   cargo run --bin eou-test -- --no-endpoint  # EOU-only, no silence fallback
//
// Model dir defaults to ~/.local/share/telemuze/models/parakeet-realtime-eou-120m-v1

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use tracing::{debug, info};

const SAMPLE_RATE: u32 = 16000;

#[derive(Parser)]
#[command(
    name = "eou-test",
    about = "Test parakeet realtime EOU detection from live mic"
)]
struct Args {
    /// Path to model directory containing encoder, decoder, joiner, tokens.
    /// Defaults to ~/.local/share/telemuze/models/parakeet-realtime-eou-120m-v1
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Use full-precision models instead of int8 quantized
    #[arg(long)]
    no_int8: bool,

    /// Feature dimension (parakeet EOU uses 128)
    #[arg(long, default_value_t = 128)]
    feat_dim: i32,

    /// Number of model threads
    #[arg(long, default_value_t = 2)]
    num_threads: i32,

    /// Min trailing silence (seconds) to trigger endpoint without speech.
    /// Acts as a fallback when no <EOU> token is emitted.
    #[arg(long, default_value_t = 3.0)]
    rule1_silence: f32,

    /// Min trailing silence (seconds) to trigger endpoint after speech.
    /// Acts as a fallback when no <EOU> token is emitted.
    #[arg(long, default_value_t = 2.0)]
    rule2_silence: f32,

    /// Max utterance length (seconds) before forced endpoint
    #[arg(long, default_value_t = 30.0)]
    rule3_max_length: f32,

    /// Disable silence-based endpoint detection (rely solely on <EOU> token)
    #[arg(long)]
    no_endpoint: bool,

    /// Penalty for blank token (positive = fewer blanks = more token emissions)
    #[arg(long, default_value_t = 0.0)]
    blank_penalty: f32,

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
        "eou_test=debug"
    } else {
        "eou_test=info"
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
            .join("telemuze/models/parakeet-realtime-eou-120m-v1")
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
    config.enable_endpoint = !args.no_endpoint;
    config.rule1_min_trailing_silence = args.rule1_silence;
    config.rule2_min_trailing_silence = args.rule2_silence;
    config.rule3_min_utterance_length = args.rule3_max_length;
    config.blank_penalty = args.blank_penalty;

    info!("Creating online recognizer (this may take a moment)...");
    let recognizer =
        sherpa_onnx::OnlineRecognizer::create(&config).context("Failed to create recognizer")?;

    let stream = recognizer.create_stream();

    eprintln!(
        "Listening (feat_dim={})... Ctrl+C to stop",
        args.feat_dim
    );
    eprintln!("Watching for <EOU> tokens in decoded output.\n");

    // Ctrl+C handler.
    let running = Arc::new(AtomicBool::new(true));
    signal_hook::flag::register_conditional_default(signal_hook::consts::SIGINT, running.clone())?;
    signal_hook::flag::register_conditional_default(signal_hook::consts::SIGTERM, running.clone())?;

    let mut total_samples = 0u64;
    let mut last_text = String::new();
    let mut segment_idx = 0u32;
    let mut eou_count = 0u32;
    let mut decode_calls = 0u64;

    while running.load(Ordering::Relaxed) {
        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                total_samples += samples.len() as u64;
                let audio_secs = total_samples as f64 / SAMPLE_RATE as f64;
                stream.accept_waveform(SAMPLE_RATE as i32, &samples);

                let chunk_start = Instant::now();
                let mut chunk_decodes = 0u32;

                while recognizer.is_ready(&stream) {
                    let t0 = Instant::now();
                    recognizer.decode(&stream);
                    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    decode_calls += 1;
                    chunk_decodes += 1;

                    if decode_ms > 50.0 {
                        debug!(
                            "Slow decode: {decode_ms:.1}ms (call #{decode_calls}, \
                             audio={audio_secs:.1}s)"
                        );
                    }

                    if let Some(result) = recognizer.get_result(&stream) {
                        let has_eou = result.tokens.iter().any(|t| t.contains("EOU"));
                        let text = result.text.trim().to_string();

                        if has_eou {
                            let secs = total_samples as f64 / SAMPLE_RATE as f64;
                            debug!("tokens: {:?}", result.tokens);

                            // Strip special tokens from final text.
                            let final_text: String = result
                                .tokens
                                .iter()
                                .filter(|t| {
                                    !t.contains("EOU") && !t.contains("EOB") && !t.contains("blk")
                                })
                                .cloned()
                                .collect::<String>();
                            let final_text = final_text.trim();

                            eou_count += 1;
                            if !final_text.is_empty() {
                                eprintln!(
                                    "\r\x1b[K[{secs:.1}s] #{segment_idx}: {final_text} \x1b[32m[EOU]\x1b[0m"
                                );
                            } else {
                                eprintln!(
                                    "\r\x1b[K[{secs:.1}s] #{segment_idx}: \x1b[32m[EOU detected]\x1b[0m"
                                );
                            }
                            segment_idx += 1;
                            last_text.clear();
                            recognizer.reset(&stream);
                            break;
                        } else if !text.is_empty() && text != last_text {
                            let secs = total_samples as f64 / SAMPLE_RATE as f64;
                            debug!("tokens: {:?}", result.tokens);
                            eprint!("\r\x1b[K[{secs:.1}s] #{segment_idx}: {text}");
                            last_text = text;
                        }
                    }

                    // Fallback: silence-based endpoint detection.
                    if recognizer.is_endpoint(&stream) {
                        let secs = total_samples as f64 / SAMPLE_RATE as f64;
                        let text = last_text.trim().to_string();
                        if !text.is_empty() {
                            eprintln!(
                                "\r\x1b[K[{secs:.1}s] #{segment_idx}: {text} \x1b[33m[silence]\x1b[0m"
                            );
                            segment_idx += 1;
                        }
                        last_text.clear();
                        recognizer.reset(&stream);
                    }
                }

                let chunk_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;
                if chunk_decodes > 0 {
                    debug!(
                        "Chunk: {chunk_decodes} decodes in {chunk_ms:.1}ms \
                         ({:.1}ms/decode), audio={audio_secs:.1}s",
                        chunk_ms / chunk_decodes as f64
                    );
                }
                if chunk_ms > 500.0 {
                    debug!(
                        "Decode loop took {chunk_ms:.0}ms for {chunk_decodes} decodes — \
                         falling behind real-time!"
                    );
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Flush any remaining text.
    let text = last_text.trim();
    if !text.is_empty() {
        let secs = total_samples as f64 / SAMPLE_RATE as f64;
        eprintln!("\r\x1b[K[{secs:.1}s] #{segment_idx}: {text}");
    }

    eprintln!("\n--- Stats: {segment_idx} segments, {eou_count} EOU ---");

    drop(audio_stream);
    info!("Shutting down");
    std::process::exit(0);
}
