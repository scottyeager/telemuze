// Two-pass command detection experiment using Parakeet-TDT 110m transducer.
//
// Runs two decode passes on the same audio with different hotword configs:
//   Pass 1 (early, short audio): heavily boosts action verbs to nail the first word
//   Pass 2 (full audio after silence): boosts continuation vocabulary (modifiers, keys)
//   Combined: first word from pass 1 + remaining words from pass 2
//
// Usage:
//   cargo run --example two_pass_detect --features mic -- \
//     --first-words "press,click,scroll,slash,undo" \
//     --continuation-words "shift,control,alt,super,enter,tab,escape,space,backspace,delete,up,down,left,right,upper,lower" \
//     --boost-first 5.0 --boost-continuation 5.0

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
#[command(about = "Two-pass VAD-gated command detection with Parakeet-TDT 110m transducer")]
struct Args {
    /// Comma-separated first-position command words (e.g., "press,click,scroll")
    #[arg(long)]
    first_words: String,

    /// Comma-separated continuation vocabulary (e.g., "shift,control,alt,enter")
    #[arg(long)]
    continuation_words: String,

    /// Hotword boost score for first-word pass
    #[arg(long, default_value_t = 5.0)]
    boost_first: f32,

    /// Hotword boost score for continuation pass
    #[arg(long, default_value_t = 5.0)]
    boost_continuation: f32,

    /// Audio lookback before VAD trigger (ms) to capture speech onset
    #[arg(long, default_value_t = 500)]
    prefill_ms: u64,

    /// Audio from VAD onset to send to pass 1 (ms)
    #[arg(long, default_value_t = 800)]
    first_pass_ms: u64,

    /// Silence duration after speech ends before triggering pass 2 (ms)
    #[arg(long, default_value_t = 500)]
    silence_ms: u64,

    /// Beam search width
    #[arg(long, default_value_t = 4)]
    max_active_paths: i32,

    /// Blank penalty (negative values speed up decoding)
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
// State machine
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum State {
    Idle,
    Capturing { onset_sample: usize },
    WaitingSilence { pass1_text: String },
}

// ---------------------------------------------------------------------------
// Model download (same as command_detect)
// ---------------------------------------------------------------------------

fn models_dir() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("telemuze/models")
}

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

fn has_model_files(model_dir: &Path) -> bool {
    let has_fp32 = model_dir.join(MODEL_ENCODER).exists()
        && model_dir.join(MODEL_DECODER).exists()
        && model_dir.join(MODEL_JOINER).exists();
    let has_int8 = model_dir.join(MODEL_ENCODER_INT8).exists()
        && model_dir.join(MODEL_DECODER_INT8).exists()
        && model_dir.join(MODEL_JOINER_INT8).exists();
    (has_fp32 || has_int8) && model_dir.join(MODEL_TOKENS).exists()
}

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
// Helpers
// ---------------------------------------------------------------------------

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

/// Build hotwords string for sherpa-onnx (one word per line with boost score).
fn build_hotwords(words: &[String], boost: f32) -> String {
    words
        .iter()
        .map(|w| format!("{w} :{boost}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Create an OfflineRecognizer with the given hotword boost score.
fn create_recognizer(
    model_dir: &Path,
    args: &Args,
    hotwords_score: f32,
) -> Result<OfflineRecognizer> {
    let resolve = |fp32: &str, int8: &str| -> String {
        let int8_path = model_dir.join(int8);
        if int8_path.exists() {
            int8_path.to_string_lossy().into_owned()
        } else {
            model_dir.join(fp32).to_string_lossy().into_owned()
        }
    };

    let mut rec_config = OfflineRecognizerConfig::default();
    rec_config.model_config.transducer.encoder =
        Some(resolve(MODEL_ENCODER, MODEL_ENCODER_INT8));
    rec_config.model_config.transducer.decoder =
        Some(resolve(MODEL_DECODER, MODEL_DECODER_INT8));
    rec_config.model_config.transducer.joiner =
        Some(resolve(MODEL_JOINER, MODEL_JOINER_INT8));
    rec_config.model_config.tokens =
        Some(model_dir.join(MODEL_TOKENS).to_string_lossy().into_owned());
    rec_config.model_config.model_type = Some("nemo_transducer".to_string());
    rec_config.model_config.num_threads = args.threads;
    rec_config.decoding_method = Some("modified_beam_search".to_string());
    rec_config.max_active_paths = args.max_active_paths;
    rec_config.hotwords_score = hotwords_score;
    rec_config.blank_penalty = args.blank_penalty;

    let bpe_vocab = model_dir.join("bpe.vocab");
    if bpe_vocab.exists() {
        rec_config.model_config.modeling_unit = Some("bpe".into());
        rec_config.model_config.bpe_vocab = Some(bpe_vocab.to_string_lossy().into_owned());
    }

    OfflineRecognizer::create(&rec_config)
        .ok_or_else(|| anyhow::anyhow!("failed to create recognizer — check model files"))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    // Parse word lists
    let first_words: Vec<String> = args
        .first_words
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    let continuation_words: Vec<String> = args
        .continuation_words
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    if first_words.is_empty() {
        bail!("No first words provided. Use --first-words \"press,click,scroll\"");
    }
    if continuation_words.is_empty() {
        bail!("No continuation words provided. Use --continuation-words \"shift,control,alt\"");
    }

    eprintln!("First words:        {:?}", first_words);
    eprintln!("Continuation words: {:?}", continuation_words);

    let hotwords_pass1 = build_hotwords(&first_words, args.boost_first);
    let hotwords_pass2 = build_hotwords(&continuation_words, args.boost_continuation);

    let base_dir = models_dir();

    let model_dir = args.model_dir.as_ref().map(PathBuf::from).unwrap_or_else(|| {
        let int8_dir = base_dir.join(MODEL_DIR_NAME_INT8);
        if int8_dir.join(MODEL_ENCODER_INT8).exists() {
            int8_dir
        } else {
            base_dir.join(MODEL_DIR_NAME)
        }
    });

    let vad_path = args
        .vad_model
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| base_dir.join(VAD_FILENAME));

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

    // Two recognizer instances with different hotword configs
    let using_int8 = model_dir.join(MODEL_ENCODER_INT8).exists();
    let quant_str = if using_int8 { "INT8" } else { "FP32" };
    eprintln!(
        "Loading Parakeet-TDT 110m transducer ({quant_str}) from {} ...",
        model_dir.display()
    );

    let recognizer_pass1 = create_recognizer(&model_dir, &args, args.boost_first)?;
    let recognizer_pass2 = create_recognizer(&model_dir, &args, args.boost_continuation)?;

    let bpe_vocab = model_dir.join("bpe.vocab");
    if bpe_vocab.exists() {
        eprintln!("BPE hotword encoding enabled");
    }
    eprintln!("Both recognizers loaded.");

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
    let prefill_samples = (args.prefill_ms as usize * sample_rate as usize) / 1000;
    let first_pass_samples = (args.first_pass_ms as usize * sample_rate as usize) / 1000;

    let mut mic_buf = Vec::<f32>::new();
    let mut mic_offset = 0_usize;
    let mut audio_buf = Vec::<f32>::new();
    let mut speech_active = false;
    let mut last_copied = 0_usize;
    let mut silence_since: Option<Instant> = None;
    let mut state = State::Idle;
    let mut detection_count = 0_u32;
    // Track the sample index in audio_buf where VAD onset was captured
    // (i.e., after the prefill portion).
    let mut onset_audio_idx: usize;

    eprintln!(
        "\nprefill_ms={} | first_pass_ms={} | silence_ms={} | boost_first={} | boost_continuation={} | beam_width={}",
        args.prefill_ms, args.first_pass_ms, args.silence_ms,
        args.boost_first, args.boost_continuation, args.max_active_paths
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
                let prefill_start = mic_offset.saturating_sub(prefill_samples);
                audio_buf.extend_from_slice(&mic_buf[prefill_start..mic_offset]);
                onset_audio_idx = audio_buf.len();
                last_copied = mic_offset;
                if state == State::Idle {
                    state = State::Capturing {
                        onset_sample: onset_audio_idx,
                    };
                }
            } else if speech_active && !vad.detected() {
                speech_active = false;
                silence_since = Some(Instant::now());
            }
            mic_offset += vad_window;
        }

        // Copy audio while speech is active or we're in a non-idle state
        if (speech_active || !matches!(state, State::Idle)) && mic_offset > last_copied {
            let new = &mic_buf[last_copied..mic_offset];
            audio_buf.extend_from_slice(new);
            last_copied = mic_offset;
        }

        // Trim idle mic buffer
        let keep = prefill_samples + 10 * vad_window;
        if !speech_active && matches!(state, State::Idle) && mic_buf.len() > keep {
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

        // State machine transitions
        match &state {
            State::Idle => {}

            State::Capturing { onset_sample } => {
                let samples_since_onset = audio_buf.len().saturating_sub(*onset_sample);
                if samples_since_onset >= first_pass_samples {
                    // Run pass 1 on prefill + first_pass_ms of audio
                    let pass1_end = onset_sample + first_pass_samples;
                    let pass1_audio = &audio_buf[..pass1_end.min(audio_buf.len())];
                    let dur = pass1_audio.len() as f32 / sample_rate as f32;

                    let decode_start = Instant::now();
                    let s = recognizer_pass1.create_stream_with_hotwords(&hotwords_pass1);
                    s.accept_waveform(sample_rate, pass1_audio);
                    recognizer_pass1.decode(&s);

                    let pass1_text = if let Some(result) = s.get_result() {
                        normalize(&result.text)
                    } else {
                        String::new()
                    };
                    let decode_ms = decode_start.elapsed().as_millis();

                    println!("--- pass 1: {dur:.1}s audio ---");
                    println!(
                        "  Pass 1 (first-word boost):   {:<24} [{decode_ms}ms]",
                        format!("\"{}\"", pass1_text)
                    );

                    state = State::WaitingSilence { pass1_text };
                }
            }

            State::WaitingSilence { pass1_text } => {
                if let Some(silence_start) = silence_since {
                    let silence_ms = silence_start.elapsed().as_millis() as u64;
                    if silence_ms >= args.silence_ms && !audio_buf.is_empty() {
                        let dur = audio_buf.len() as f32 / sample_rate as f32;

                        let decode_start = Instant::now();
                        let s =
                            recognizer_pass2.create_stream_with_hotwords(&hotwords_pass2);
                        s.accept_waveform(sample_rate, &audio_buf);
                        recognizer_pass2.decode(&s);

                        let pass2_text = if let Some(result) = s.get_result() {
                            normalize(&result.text)
                        } else {
                            String::new()
                        };
                        let decode_ms = decode_start.elapsed().as_millis();

                        // Combine: first word of pass 1 + remaining words of pass 2
                        let first_word = pass1_text.split_whitespace().next().unwrap_or("");
                        let pass2_rest: String = pass2_text
                            .split_whitespace()
                            .skip(1)
                            .collect::<Vec<_>>()
                            .join(" ");
                        let combined = if pass2_rest.is_empty() {
                            first_word.to_string()
                        } else {
                            format!("{first_word} {pass2_rest}")
                        };

                        detection_count += 1;
                        println!("--- pass 2: {dur:.1}s audio ---");
                        println!(
                            "  Pass 2 (continuation boost): {:<24} [{decode_ms}ms]",
                            format!("\"{}\"", pass2_text)
                        );
                        println!(
                            "  Combined:                    \"{}\"",
                            combined
                        );
                        println!();

                        audio_buf.clear();
                        silence_since = None;
                        state = State::Idle;
                    }
                }
            }
        }
    }

    eprintln!("\n{detection_count} commands detected.");
    Ok(())
}
