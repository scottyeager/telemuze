// Test binary for sherpa-onnx keyword spotting from live microphone input.
//
// Downloads the English KWS model on first run, tokenizes plain-text keywords
// via sentencepiece BPE, and streams audio from the default input device.
//
// Usage:
//   cargo run --bin keyword-test -- --keywords "hello,stop,hey computer"

use anyhow::{bail, Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sentencepiece_model::SentencePieceModel;
use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;
use tracing::{debug, info};

const SAMPLE_RATE: u32 = 16_000;

const MODEL_NAME: &str = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01";
const MODEL_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2";

#[derive(Parser)]
#[command(name = "keyword-test", about = "Test sherpa-onnx keyword spotting from live mic")]
struct Args {
    /// Path to model directory (auto-downloads English GigaSpeech model if omitted)
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

// ── BPE tokenizer ───────────────────────────────────────────────────────────

/// A simple BPE encoder using a sentencepiece model file.
struct BpeTokenizer {
    /// token string → (piece_id, score)
    vocab: HashMap<String, (u32, f32)>,
    /// Whether the model vocabulary is uppercase (auto-detected)
    uppercase: bool,
}

impl BpeTokenizer {
    fn load(model_path: &Path) -> Result<Self> {
        let model = SentencePieceModel::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load BPE model: {e}"))?;

        let mut vocab = HashMap::new();
        let mut upper_count = 0u32;
        let mut lower_count = 0u32;
        for (id, piece) in model.pieces().iter().enumerate() {
            if let Some(ref s) = piece.piece {
                vocab.insert(s.clone(), (id as u32, piece.score.unwrap_or(0.0)));
                // Sample alphabetic tokens to detect case
                let alpha: String = s.chars().filter(|c| c.is_alphabetic()).collect();
                if !alpha.is_empty() {
                    if alpha == alpha.to_uppercase() {
                        upper_count += 1;
                    } else {
                        lower_count += 1;
                    }
                }
            }
        }
        let uppercase = upper_count > lower_count;
        Ok(Self { vocab, uppercase })
    }

    /// Encode text into BPE token strings, matching sentencepiece behavior.
    /// Prepends ▁ at word boundaries and iteratively merges the highest-scored pair.
    fn encode(&self, text: &str) -> Vec<String> {
        // Sentencepiece normalizes by prepending ▁ and replacing spaces with ▁
        let text = if self.uppercase { text.to_uppercase() } else { text.to_lowercase() };
        let normalized = format!("▁{}", text.replace(' ', "▁"));

        // Start with individual characters (as strings)
        let mut symbols: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();

        // Iteratively merge the adjacent pair that forms the highest-scored vocab token
        loop {
            if symbols.len() < 2 {
                break;
            }

            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = None;

            for i in 0..symbols.len() - 1 {
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                if let Some(&(_, score)) = self.vocab.get(&merged) {
                    if score > best_score {
                        best_score = score;
                        best_idx = Some(i);
                    }
                }
            }

            match best_idx {
                Some(i) => {
                    symbols[i] = format!("{}{}", symbols[i], symbols[i + 1]);
                    symbols.remove(i + 1);
                }
                None => break,
            }
        }

        symbols
    }
}

// ── Model management ────────────────────────────────────────────────────────

/// Find the encoder/decoder/joiner files in model_dir, preferring int8 variants.
fn find_model_files(model_dir: &Path) -> Result<(String, String, String, String)> {
    let entries: Vec<String> = fs::read_dir(model_dir)
        .context("Cannot read model directory")?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();

    let find = |prefix: &str| -> Result<String> {
        // Prefer int8 variant
        if let Some(f) = entries.iter().find(|f| f.starts_with(prefix) && f.contains("int8") && f.ends_with(".onnx")) {
            return Ok(model_dir.join(f).to_string_lossy().into_owned());
        }
        if let Some(f) = entries.iter().find(|f| f.starts_with(prefix) && f.ends_with(".onnx")) {
            return Ok(model_dir.join(f).to_string_lossy().into_owned());
        }
        bail!("No {prefix}*.onnx found in {}", model_dir.display());
    };

    let encoder = find("encoder")?;
    let decoder = find("decoder")?;
    let joiner = find("joiner")?;

    let tokens = model_dir.join("tokens.txt");
    if !tokens.exists() {
        bail!("tokens.txt not found in {}", model_dir.display());
    }

    Ok((encoder, decoder, joiner, tokens.to_string_lossy().into_owned()))
}

/// Tokenize plain-text keywords into the format expected by KeywordSpotter.
fn tokenize_keywords(keywords: &str, model_dir: &Path) -> Result<String> {
    let bpe_model_path = model_dir.join("bpe.model");
    if !bpe_model_path.exists() {
        bail!("bpe.model not found in {} — cannot tokenize plain-text keywords", model_dir.display());
    }

    let tokenizer = BpeTokenizer::load(&bpe_model_path)?;

    let mut lines = Vec::new();
    for kw in keywords.split(',') {
        let kw = kw.trim();
        if kw.is_empty() {
            continue;
        }
        let tokens = tokenizer.encode(kw);
        let label = kw.replace(' ', "_");
        let line = format!("{} @{}", tokens.join(" "), label);
        debug!("Keyword '{kw}' → {line}");
        lines.push(line);
    }

    if lines.is_empty() {
        bail!("No keywords provided");
    }

    Ok(lines.join("\n"))
}

/// Download and extract the KWS model if not present.
fn ensure_model(model_dir: &Path) -> Result<()> {
    let has_encoder = fs::read_dir(model_dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.starts_with("encoder") && name.ends_with(".onnx")
                })
        })
        .unwrap_or(false);

    if has_encoder && model_dir.join("tokens.txt").exists() {
        return Ok(());
    }

    let parent = model_dir.parent().context("Model dir has no parent")?;
    fs::create_dir_all(parent)?;

    eprintln!("Downloading KWS model...");
    let response = reqwest::blocking::get(MODEL_URL).context("Failed to download model")?;
    if !response.status().is_success() {
        bail!("Download failed with status {}", response.status());
    }

    let bytes = response.bytes().context("Failed to read response body")?;
    eprintln!("Downloaded {} MB, extracting...", bytes.len() / 1_000_000);

    let decoder = bzip2::read::BzDecoder::new(bytes.as_ref());
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(parent).context("Failed to extract model archive")?;

    eprintln!("Model ready.");
    Ok(())
}

fn default_model_dir() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("telemuze")
        .join("models")
        .join(MODEL_NAME)
}

// ── Main ────────────────────────────────────────────────────────────────────

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

    let model_dir = args.model_dir.unwrap_or_else(default_model_dir);

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

    // Now load model and create keyword spotter
    ensure_model(&model_dir)?;

    let (encoder, decoder, joiner, tokens) = find_model_files(&model_dir)?;
    debug!("Encoder: {encoder}");
    debug!("Decoder: {decoder}");
    debug!("Joiner:  {joiner}");
    debug!("Tokens:  {tokens}");

    let mut config = KeywordSpotterConfig::default();
    config.model_config.transducer.encoder = Some(encoder);
    config.model_config.transducer.decoder = Some(decoder);
    config.model_config.transducer.joiner = Some(joiner);
    config.model_config.tokens = Some(tokens);
    config.model_config.num_threads = args.num_threads;
    config.model_config.provider = Some("cpu".into());
    config.keywords_score = args.keywords_score;
    config.keywords_threshold = args.keywords_threshold;

    if let Some(ref kw_file) = args.keywords_file {
        config.keywords_file = Some(kw_file.to_string_lossy().into_owned());
    } else if let Some(ref kw_text) = args.keywords {
        let keywords_buf = tokenize_keywords(kw_text, &model_dir)?;
        config.keywords_buf = Some(keywords_buf);
    }

    let kws = KeywordSpotter::create(&config)
        .context("Failed to create KeywordSpotter — check model paths and keywords format")?;
    let kws_stream = kws.create_stream();

    eprintln!("Listening... (Ctrl+C to stop)");

    // Set up Ctrl+C handler — first Ctrl+C sets the flag, second one kills the process
    let running = Arc::new(AtomicBool::new(true));
    signal_hook::flag::register_conditional_default(signal_hook::consts::SIGINT, running.clone())?;
    signal_hook::flag::register_conditional_default(signal_hook::consts::SIGTERM, running.clone())?;

    let mut total_samples = 0u64;
    while running.load(Ordering::Relaxed) {
        match audio_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(samples) => {
                total_samples += samples.len() as u64;
                kws_stream.accept_waveform(SAMPLE_RATE as i32, &samples);

                while kws.is_ready(&kws_stream) {
                    kws.decode(&kws_stream);
                    if let Some(result) = kws.get_result(&kws_stream) {
                        if !result.keyword.is_empty() {
                            let secs = total_samples as f64 / SAMPLE_RATE as f64;
                            println!("[{secs:.1}s] {}", result.keyword);
                            kws.reset(&kws_stream);
                        }
                    }
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
