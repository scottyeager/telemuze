// Test binary for sherpa-onnx keyword spotting from live microphone input.
//
// Downloads the selected KWS model on first run, tokenizes plain-text keywords
// (BPE for gigaspeech, phone-based for zh-en), and streams audio from the
// default input device.
//
// Usage:
//   cargo run --bin keyword-test -- --keywords "hello,stop,hey computer"
//   cargo run --bin keyword-test -- --model zh-en --keywords "light up,hello"

use anyhow::{bail, Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;
use telemuze_listen::kws;
use tracing::{debug, info};

const SAMPLE_RATE: u32 = kws::SAMPLE_RATE;
/// Reset the keyword spotter stream after this many seconds of audio without a
/// detection, preventing unbounded internal-state growth and rising CPU usage.
const RESET_INTERVAL_SECS: u32 = 60;

#[derive(Parser)]
#[command(name = "keyword-test", about = "Test sherpa-onnx keyword spotting from live mic")]
struct Args {
    /// Path to model directory (auto-downloads the default zh-en model if omitted).
    /// Use a gigaspeech model dir for BPE-based English-only keyword spotting.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Comma-separated plain-text keywords (e.g., "hello,stop,hey computer")
    #[arg(long, group = "kw")]
    keywords: Option<String>,

    /// Path to pre-tokenized keywords file
    #[arg(long, group = "kw")]
    keywords_file: Option<PathBuf>,

    /// Keyword boost score
    #[arg(long, default_value_t = 1.0)]
    keywords_score: f32,

    /// Keyword detection threshold (lower = more sensitive)
    #[arg(long, default_value_t = 0.25)]
    keywords_threshold: f32,

    /// Number of model threads
    #[arg(long, default_value_t = 2)]
    num_threads: i32,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let log_level = if args.verbose { "keyword_test=debug" } else { "keyword_test=warn" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .init();

    if args.keywords.is_none() && args.keywords_file.is_none() {
        bail!("Provide either --keywords or --keywords-file");
    }

    let model_dir = args.model_dir.unwrap_or_else(kws::default_model_dir);
    let model = kws::detect_model(&model_dir).unwrap_or(kws::DEFAULT_MODEL);

    // Set up audio capture FIRST — before loading the model, because
    // onnxruntime initialization can interfere with ALSA thread setup.
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

    // Now load model and create keyword spotter.
    // If a pre-tokenized keywords file was given, use the lower-level API
    // directly; otherwise go through the convenience initializer.
    let spotter = if let Some(ref kw_file) = args.keywords_file {
        use sherpa_onnx::KeywordSpotterConfig;

        kws::ensure_model(&model_dir, model)?;
        let (encoder, decoder, joiner, tokens) =
            kws::find_model_files(&model_dir, model.chunk_hint())?;

        let mut config = KeywordSpotterConfig::default();
        config.model_config.transducer.encoder = Some(encoder);
        config.model_config.transducer.decoder = Some(decoder);
        config.model_config.transducer.joiner = Some(joiner);
        config.model_config.tokens = Some(tokens);
        config.model_config.num_threads = args.num_threads;
        config.model_config.provider = Some("cpu".into());
        config.keywords_score = args.keywords_score;
        config.keywords_threshold = args.keywords_threshold;
        config.keywords_file = Some(kw_file.to_string_lossy().into_owned());

        sherpa_onnx::KeywordSpotter::create(&config)
            .context("Failed to create KeywordSpotter")?
    } else {
        kws::init_keyword_spotter(&kws::KwsConfig {
            model,
            model_dir: model_dir.clone(),
            keywords: args.keywords.unwrap(),
            keywords_score: args.keywords_score,
            keywords_threshold: args.keywords_threshold,
            num_threads: args.num_threads,
        })?
    };

    let kws_stream = spotter.create_stream();

    eprintln!("Listening... (Ctrl+C to stop)");

    // Set up Ctrl+C handler — first Ctrl+C sets the flag, second one kills the process
    let running = Arc::new(AtomicBool::new(true));
    signal_hook::flag::register_conditional_default(signal_hook::consts::SIGINT, running.clone())?;
    signal_hook::flag::register_conditional_default(signal_hook::consts::SIGTERM, running.clone())?;

    let mut total_samples = 0u64;
    let mut samples_since_reset = 0u64;
    let reset_threshold = RESET_INTERVAL_SECS as u64 * SAMPLE_RATE as u64;
    while running.load(Ordering::Relaxed) {
        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                total_samples += samples.len() as u64;
                samples_since_reset += samples.len() as u64;
                kws_stream.accept_waveform(SAMPLE_RATE as i32, &samples);

                while spotter.is_ready(&kws_stream) {
                    spotter.decode(&kws_stream);
                    if let Some(result) = spotter.get_result(&kws_stream) {
                        if !result.keyword.is_empty() {
                            let secs = total_samples as f64 / SAMPLE_RATE as f64;
                            println!("[{secs:.1}s] {}", result.keyword);
                            spotter.reset(&kws_stream);
                            samples_since_reset = 0;
                        }
                    }
                }

                if samples_since_reset >= reset_threshold {
                    debug!("Periodic stream reset after {RESET_INTERVAL_SECS}s without detection");
                    spotter.reset(&kws_stream);
                    samples_since_reset = 0;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    drop(audio_stream);
    info!("Shutting down");
    std::process::exit(0);
}
