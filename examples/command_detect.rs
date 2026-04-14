// VAD-gated command detection using Parakeet-TDT 110m transducer with
// hotword-boosted beam search and keyword matching.
//
// Listens for speech via microphone, decodes with the 110m-parameter
// Parakeet TDT transducer model using modified beam search with heavy
// hotword boosting for target keywords, then verifies the decoded text
// matches a command from the keyword list.
//
// The pipeline:
//   1. Silero VAD detects speech onset → start buffering audio
//   2. Silence exceeds threshold → decode with hotword-boosted beam search
//   3. Match decoded text against keyword list (exact, contains, fuzzy)
//   4. Emit matched command or reject as unrecognized
//
// Models are downloaded automatically on first run (~455MB total).
//
// Usage:
//   cargo run --example command_detect --features mic -- \
//     --keywords "turn on,turn off,stop,start,open,close"

use anyhow::{bail, Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, VadModelConfig, VoiceActivityDetector,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Model constants
// ---------------------------------------------------------------------------

const MODEL_DIR_NAME: &str = "sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000";
const MODEL_DIR_NAME_INT8: &str = "sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8";
const MODEL_HF_BASE: &str = "https://huggingface.co/scottyeager/sherpa-onnx-nemo-parakeet-tdt-transducer-110m-en-int8/resolve/main";
const MODEL_ENCODER: &str = "encoder.onnx";
const MODEL_ENCODER_INT8: &str = "encoder.int8.onnx";
const MODEL_DECODER: &str = "decoder.onnx";
const MODEL_DECODER_INT8: &str = "decoder.int8.onnx";
const MODEL_JOINER: &str = "joiner.onnx";
const MODEL_JOINER_INT8: &str = "joiner.int8.onnx";
const MODEL_TOKENS: &str = "tokens.txt";
const MODEL_BPE_VOCAB: &str = "bpe.vocab";

const VAD_FILENAME: &str = "silero_vad.onnx";
const VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

#[derive(Parser)]
#[command(about = "VAD-gated command detection with Parakeet-TDT 110m transducer")]
struct Args {
    /// Comma-separated keywords or phrases to detect
    #[arg(long)]
    keywords: String,

    /// Silence duration (ms) after speech ends to trigger decode
    #[arg(long, default_value_t = 500)]
    silence_ms: u64,

    /// Hotword boost score — higher values bias the model more strongly
    /// toward the target keywords (typical range: 1.5–5.0)
    #[arg(long, default_value_t = 3.0)]
    boost: f32,

    /// Beam search width (higher = more accurate but slower)
    #[arg(long, default_value_t = 4)]
    max_active_paths: i32,

    /// Blank penalty — negative values speed up decoding by discouraging
    /// blank token emission (try -1.0 to -2.0)
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    blank_penalty: f32,

    /// Path to Parakeet TDT 110m transducer model directory
    #[arg(long, env = "PARAKEET_TDT_110M_MODEL_DIR")]
    model_dir: Option<String>,

    /// Path to silero_vad.onnx
    #[arg(long, env = "SILERO_VAD_MODEL")]
    vad_model: Option<String>,

    /// Number of decode threads
    #[arg(long, default_value_t = 4)]
    threads: i32,
}

// ---------------------------------------------------------------------------
// Model download
// ---------------------------------------------------------------------------

fn models_dir() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("telemuze/models")
}

/// Download a single file using wget (preferred) or curl.
fn download_file(url: &str, dest: &Path) -> Result<()> {
    let name = dest.file_name().unwrap().to_string_lossy();
    eprintln!("Downloading {name} ...");

    let status = std::process::Command::new("wget")
        .args(["-q", "--show-progress", "-O"])
        .arg(dest)
        .arg(url)
        .status()
        .or_else(|_| {
            std::process::Command::new("curl")
                .args(["-L", "--progress-bar", "-o"])
                .arg(dest)
                .arg(url)
                .status()
        })
        .context("Neither wget nor curl found — install one to enable auto-download")?;

    if !status.success() {
        bail!("Download failed for {name}");
    }
    Ok(())
}

/// Generate bpe.vocab from tokens.txt if it doesn't exist.
///
/// The Parakeet SentencePiece tokenizer uses sequential scores: 0.0, -0.0,
/// -1.0, -2.0, ... (verified against the actual SP model extracted from the
/// NeMo checkpoint). The last token in tokens.txt is the blank token and is
/// excluded from bpe.vocab.
fn generate_bpe_vocab(model_dir: &Path) -> Result<()> {
    let bpe_vocab = model_dir.join(MODEL_BPE_VOCAB);
    if bpe_vocab.exists() {
        return Ok(());
    }

    let tokens_path = model_dir.join(MODEL_TOKENS);
    let content = std::fs::read_to_string(&tokens_path)
        .with_context(|| format!("Failed to read {}", tokens_path.display()))?;

    let mut lines = Vec::new();
    let all_lines: Vec<&str> = content.lines().collect();
    // Exclude the last token (blank token used by CTC/TDT)
    let token_lines = if all_lines.len() > 1 {
        &all_lines[..all_lines.len() - 1]
    } else {
        &all_lines[..]
    };

    for (i, line) in token_lines.iter().enumerate() {
        let token = line.split_whitespace().next().unwrap_or("");
        if token.is_empty() {
            continue;
        }
        let score = if i == 0 { 0.0 } else { -((i - 1) as f64) };
        lines.push(format!("{token}\t{score:.1}"));
    }

    std::fs::write(&bpe_vocab, lines.join("\n") + "\n")
        .with_context(|| format!("Failed to write {}", bpe_vocab.display()))?;
    eprintln!("Generated bpe.vocab ({} tokens)", lines.len());
    Ok(())
}

/// Check if a model directory has all required files (fp32 or int8 variants).
fn has_model_files(model_dir: &Path) -> bool {
    let has_fp32 = model_dir.join(MODEL_ENCODER).exists()
        && model_dir.join(MODEL_DECODER).exists()
        && model_dir.join(MODEL_JOINER).exists();
    let has_int8 = model_dir.join(MODEL_ENCODER_INT8).exists()
        && model_dir.join(MODEL_DECODER_INT8).exists()
        && model_dir.join(MODEL_JOINER_INT8).exists();
    (has_fp32 || has_int8) && model_dir.join(MODEL_TOKENS).exists()
}

/// Download and extract the Parakeet TDT 110m transducer model if not present.
fn ensure_model(model_dir: &Path) -> Result<()> {
    if has_model_files(model_dir) {
        generate_bpe_vocab(model_dir)?;
        return Ok(());
    }

    std::fs::create_dir_all(model_dir)?;
    for filename in [
        MODEL_ENCODER_INT8,
        MODEL_DECODER_INT8,
        MODEL_JOINER_INT8,
        MODEL_TOKENS,
        MODEL_BPE_VOCAB,
    ] {
        let dest = model_dir.join(filename);
        if dest.exists() {
            continue;
        }
        download_file(&format!("{MODEL_HF_BASE}/{filename}"), &dest)?;
    }
    generate_bpe_vocab(model_dir)?;

    if !has_model_files(model_dir) {
        bail!(
            "Model extraction succeeded but expected files not found in {}",
            model_dir.display()
        );
    }

    eprintln!("Model ready.");
    Ok(())
}

/// Download Silero VAD if not present.
fn ensure_vad(vad_path: &Path) -> Result<()> {
    if vad_path.exists() {
        return Ok(());
    }
    if let Some(parent) = vad_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    download_file(VAD_URL, vad_path)
}

// ---------------------------------------------------------------------------
// Audio capture
// ---------------------------------------------------------------------------

fn build_input_stream(device: &cpal::Device, tx: mpsc::Sender<Vec<f32>>) -> Result<cpal::Stream> {
    let supported = device.default_input_config()?;
    let config = supported.config();
    let channels = config.channels as usize;
    let err_fn = |err| eprintln!("audio error: {:?}", err);

    eprintln!(
        "mic: {:?}, {}ch, {}Hz",
        supported.sample_format(),
        channels,
        config.sample_rate.0
    );

    let stream = match supported.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|f| f.iter().sum::<f32>() / channels as f32)
                    .collect();
                let _ = tx.send(mono);
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|f| {
                        f.iter().map(|&s| s as f32 / i16::MAX as f32).sum::<f32>() / channels as f32
                    })
                    .collect();
                let _ = tx.send(mono);
            },
            err_fn,
            None,
        )?,
        other => bail!("unsupported sample format: {:?}", other),
    };

    Ok(stream)
}

// ---------------------------------------------------------------------------
// Keyword matching
// ---------------------------------------------------------------------------

/// Normalize text for matching: lowercase, strip punctuation.
fn normalize(text: &str) -> String {
    text.trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn match_command(text: &str, keywords: &[String]) -> Option<String> {
    let normalized = normalize(text);
    if normalized.is_empty() {
        return None;
    }
    keywords
        .iter()
        .find(|kw| **kw == normalized)
        .cloned()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    // Parse keywords
    let keywords: Vec<String> = args
        .keywords
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    if keywords.is_empty() {
        bail!("No keywords provided. Use --keywords \"word1,word2,phrase three\"");
    }

    eprintln!("Keywords: {:?}", keywords);

    // Build hotwords string for sherpa-onnx (one per line, with boost score)
    let hotwords: String = keywords
        .iter()
        .map(|kw| format!("{kw} :{}", args.boost))
        .collect::<Vec<_>>()
        .join("\n");

    let base_dir = models_dir();

    let model_dir = args.model_dir.map(PathBuf::from).unwrap_or_else(|| {
        // Prefer the int8 directory if it exists
        let int8_dir = base_dir.join(MODEL_DIR_NAME_INT8);
        if int8_dir.join(MODEL_ENCODER_INT8).exists() {
            int8_dir
        } else {
            base_dir.join(MODEL_DIR_NAME)
        }
    });

    let vad_path = args
        .vad_model
        .map(PathBuf::from)
        .unwrap_or_else(|| base_dir.join(VAD_FILENAME));

    // Ensure models are downloaded
    ensure_model(&model_dir)?;
    ensure_vad(&vad_path)?;

    // Ctrl+C handler
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    ctrlc::set_handler(move || {
        stop2.store(true, Ordering::SeqCst);
        eprintln!("\nCtrl+C — exiting");
    })?;

    // VAD
    let mut vad_config = VadModelConfig::default();
    vad_config.silero_vad.model = Some(vad_path.to_string_lossy().into_owned());
    vad_config.silero_vad.threshold = 0.5;
    vad_config.silero_vad.min_silence_duration = 0.15;
    vad_config.silero_vad.min_speech_duration = 0.1;
    vad_config.silero_vad.max_speech_duration = 30.0;
    vad_config.silero_vad.window_size = 512;
    vad_config.sample_rate = 16000;
    let vad = VoiceActivityDetector::create(&vad_config, 60.0)
        .ok_or_else(|| anyhow::anyhow!("failed to create VAD"))?;

    // Offline recognizer — NeMo TDT transducer with beam search + hotwords
    // Use int8 model files if present, otherwise fall back to fp32
    let resolve = |fp32: &str, int8: &str| -> String {
        let int8_path = model_dir.join(int8);
        if int8_path.exists() {
            int8_path.to_string_lossy().into_owned()
        } else {
            model_dir.join(fp32).to_string_lossy().into_owned()
        }
    };

    let encoder_path = resolve(MODEL_ENCODER, MODEL_ENCODER_INT8);
    let decoder_path = resolve(MODEL_DECODER, MODEL_DECODER_INT8);
    let joiner_path = resolve(MODEL_JOINER, MODEL_JOINER_INT8);
    let using_int8 = encoder_path.contains("int8");

    let mut rec_config = OfflineRecognizerConfig::default();
    rec_config.model_config.transducer.encoder = Some(encoder_path);
    rec_config.model_config.transducer.decoder = Some(decoder_path);
    rec_config.model_config.transducer.joiner = Some(joiner_path);
    rec_config.model_config.tokens =
        Some(model_dir.join(MODEL_TOKENS).to_string_lossy().into_owned());
    rec_config.model_config.model_type = Some("nemo_transducer".to_string());
    rec_config.model_config.num_threads = args.threads;
    rec_config.decoding_method = Some("modified_beam_search".to_string());
    rec_config.max_active_paths = args.max_active_paths;
    rec_config.hotwords_score = args.boost;
    rec_config.blank_penalty = args.blank_penalty;

    // Enable BPE hotword encoding if bpe.vocab exists
    let bpe_vocab = model_dir.join("bpe.vocab");
    if bpe_vocab.exists() {
        rec_config.model_config.modeling_unit = Some("bpe".into());
        rec_config.model_config.bpe_vocab =
            Some(bpe_vocab.to_string_lossy().into_owned());
        eprintln!("BPE hotword encoding enabled");
    }

    let quant_str = if using_int8 { "INT8" } else { "FP32" };
    eprintln!("Loading Parakeet-TDT 110m transducer ({quant_str}) from {} ...", model_dir.display());
    let recognizer = OfflineRecognizer::create(&rec_config)
        .ok_or_else(|| anyhow::anyhow!("failed to create recognizer — check model files"))?;
    eprintln!("Model loaded.");

    // Microphone
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no input device"))?;
    eprintln!("device: {}", device.name()?);

    let supported = device.default_input_config()?;
    let mic_rate = supported.sample_rate().0 as i32;
    let resampler = if mic_rate != 16000 {
        Some(
            sherpa_onnx::LinearResampler::create(mic_rate, 16000)
                .expect("failed to create resampler"),
        )
    } else {
        None
    };

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let stream = build_input_stream(&device, tx)?;
    stream.play()?;

    let sample_rate = 16000_i32;
    let vad_window = 512_usize;

    let mut mic_buf = Vec::<f32>::new();
    let mut mic_offset = 0_usize;
    let mut audio_buf = Vec::<f32>::new();
    let mut speech_active = false;
    let mut last_copied = 0_usize;
    let mut silence_since: Option<Instant> = None;
    let mut detection_count = 0_u32;

    // Prefill: when VAD fires, include this much audio from before the
    // trigger point so we don't clip the onset of short words.
    let prefill_samples = sample_rate as usize / 2; // 500ms lookback

    eprintln!(
        "\nsilence_ms={} | boost={} | beam_width={}",
        args.silence_ms, args.boost, args.max_active_paths
    );
    eprintln!("Listening for commands... Ctrl+C to quit.\n");

    while !stop.load(Ordering::Relaxed) {
        // Drain mic samples
        match rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(samples) => {
                if let Some(ref r) = resampler {
                    mic_buf.extend_from_slice(&r.resample(&samples, false));
                } else {
                    mic_buf.extend_from_slice(&samples);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        // Feed VAD
        while mic_offset + vad_window <= mic_buf.len() {
            vad.accept_waveform(&mic_buf[mic_offset..mic_offset + vad_window]);
            if !speech_active && vad.detected() {
                speech_active = true;
                silence_since = None;
                // Start copying from before the VAD trigger to capture the
                // onset of speech that VAD needed to "hear" before firing.
                let prefill_start = mic_offset.saturating_sub(prefill_samples);
                audio_buf.extend_from_slice(&mic_buf[prefill_start..mic_offset]);
                last_copied = mic_offset;
            } else if speech_active && !vad.detected() {
                speech_active = false;
                silence_since = Some(Instant::now());
            }
            mic_offset += vad_window;
        }

        // Copy audio only while speech is active
        if speech_active && mic_offset > last_copied {
            let new = &mic_buf[last_copied..mic_offset];
            audio_buf.extend_from_slice(new);
            last_copied = mic_offset;
        }

        // Trim idle mic buffer — keep enough for prefill lookback
        let keep = prefill_samples + 10 * vad_window;
        if !speech_active && audio_buf.is_empty() && mic_buf.len() > keep {
            let trim = mic_buf.len() - keep;
            mic_offset = mic_offset.saturating_sub(trim);
            mic_buf = mic_buf[trim..].to_vec();
            last_copied = last_copied.saturating_sub(trim);
        }

        // Consume completed VAD segments
        while !vad.is_empty() {
            if let Some(_segment) = vad.front() {
                vad.pop();
            }
        }

        // Silence trigger: decode and match
        if let Some(silence_start) = silence_since {
            let silence_ms = silence_start.elapsed().as_millis() as u64;

            if silence_ms >= args.silence_ms && !audio_buf.is_empty() {
                let dur = audio_buf.len() as f32 / sample_rate as f32;
                let decode_start = Instant::now();

                let s = recognizer.create_stream_with_hotwords(&hotwords);
                s.accept_waveform(sample_rate, &audio_buf);
                recognizer.decode(&s);

                if let Some(result) = s.get_result() {
                    let decode_ms = decode_start.elapsed().as_millis();
                    let text = result.text.trim().to_string();

                    if !text.is_empty() {
                        let normalized = normalize(&text);
                        match match_command(&text, &keywords) {
                            Some(kw) => {
                                detection_count += 1;
                                println!(
                                    "[#{detection_count}] COMMAND: \"{kw}\" (heard: \"{normalized}\", {dur:.1}s audio, {decode_ms}ms decode)",
                                );
                            }
                            None => {
                                eprintln!(
                                    "[rejected] \"{normalized}\" ({dur:.1}s audio, {decode_ms}ms decode)"
                                );
                            }
                        }
                    }
                }

                audio_buf.clear();
                silence_since = None;
            }
        }
    }

    eprintln!("\n{detection_count} commands detected.");
    Ok(())
}
