// Speculative offline decoding from microphone with VAD-gated triggers.
//
// Single Silero VAD with three silence-duration thresholds:
//   - FAST (200ms): triggers speculative decode, caches result
//   - SLOW (1000ms): emits text if cached result ends with sentence punct
//   - FINAL (3000ms): emits text unconditionally
//
// Press Enter for full decode comparison.  Ctrl+C to quit.
//
// Usage:
//   cargo run --example speculative_decode --features mic -- [OPTIONS]

use anyhow::Result;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, VadModelConfig, VoiceActivityDetector,
};
use std::io::{self, Read as _};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Speculative offline decoding with VAD-gated triggers")]
struct Args {
    /// Silence duration (ms) to trigger speculative decode
    #[arg(long, default_value_t = 200)]
    fast: u64,

    /// Silence duration (ms) to emit if sentence ends with punctuation
    #[arg(long, default_value_t = 1000)]
    slow: u64,

    /// Silence duration (ms) to emit unconditionally
    #[arg(long, default_value_t = 3000, alias = "timeout")]
    r#final: u64,

    /// Path to Parakeet model directory
    #[arg(long, env = "PARAKEET_MODEL_DIR")]
    model_dir: Option<String>,

    /// Path to silero_vad.onnx
    #[arg(long, env = "SILERO_VAD_MODEL")]
    vad_model: Option<String>,

    /// Number of decode threads
    #[arg(long, default_value_t = 4)]
    threads: i32,
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
        other => anyhow::bail!("unsupported sample format: {:?}", other),
    };

    Ok(stream)
}

// ---------------------------------------------------------------------------
// Punctuation helpers
// ---------------------------------------------------------------------------

/// Check if text ends with a sentence-ending punctuation, excluding "..."
fn ends_with_sentence_punct(tokens: &[String]) -> bool {
    // Walk backwards past empty tokens
    let mut i = tokens.len();
    while i > 0 {
        i -= 1;
        if !tokens[i].is_empty() {
            break;
        }
    }
    if i >= tokens.len() {
        return false;
    }

    let last = &tokens[i];
    if last != "." && last != "?" && last != "!" {
        return false;
    }

    // Exclude "..." — check if the last 3 tokens are all "."
    if last == "." {
        let mut dot_count = 0;
        let mut j = i + 1; // include current
        while j > 0 && dot_count < 3 {
            j -= 1;
            if tokens[j] == "." {
                dot_count += 1;
            } else {
                break;
            }
        }
        if dot_count >= 3 {
            return false; // it's an ellipsis
        }
    }

    true
}

/// Decode audio buffer.
fn decode_buf(
    recognizer: &OfflineRecognizer,
    sample_rate: i32,
    buf: &[f32],
) -> Option<sherpa_onnx::OfflineRecognizerResult> {
    if buf.is_empty() {
        return None;
    }
    let s = recognizer.create_stream();
    s.accept_waveform(sample_rate, buf);
    recognizer.decode(&s);
    s.get_result()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    let data_dir = dirs_next::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("telemuze/models");

    let model_dir = args.model_dir.unwrap_or_else(|| {
        data_dir
            .join("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
            .to_string_lossy()
            .into_owned()
    });

    let vad_model = args.vad_model.unwrap_or_else(|| {
        data_dir
            .join("silero_vad.onnx")
            .to_string_lossy()
            .into_owned()
    });

    let fast_silence_ms = args.fast;
    let slow_silence_ms = args.slow;
    let final_silence_ms = args.r#final;

    // Ctrl+C handler
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    ctrlc::set_handler(move || {
        stop2.store(true, Ordering::SeqCst);
        eprintln!("\nCtrl+C — exiting");
    })?;

    // VAD — short min_silence so we get responsive speech/silence transitions
    let mut vad_config = VadModelConfig::default();
    vad_config.silero_vad.model = Some(vad_model);
    vad_config.silero_vad.threshold = 0.5;
    vad_config.silero_vad.min_silence_duration = 0.15; // responsive transitions
    vad_config.silero_vad.min_speech_duration = 0.1;
    vad_config.silero_vad.max_speech_duration = 60.0;
    vad_config.silero_vad.window_size = 512;
    vad_config.sample_rate = 16000;
    let vad = VoiceActivityDetector::create(&vad_config, 120.0)
        .ok_or_else(|| anyhow::anyhow!("failed to create VAD"))?;

    // Offline recognizer
    let mut rec_config = OfflineRecognizerConfig::default();
    rec_config.model_config.transducer.encoder =
        Some(format!("{}/encoder.int8.onnx", model_dir));
    rec_config.model_config.transducer.decoder =
        Some(format!("{}/decoder.int8.onnx", model_dir));
    rec_config.model_config.transducer.joiner =
        Some(format!("{}/joiner.int8.onnx", model_dir));
    rec_config.model_config.tokens = Some(format!("{}/tokens.txt", model_dir));
    rec_config.model_config.model_type = Some("nemo_transducer".to_string());
    rec_config.model_config.num_threads = args.threads;

    eprintln!("loading model from {} ...", model_dir);
    let recognizer = OfflineRecognizer::create(&rec_config)
        .ok_or_else(|| anyhow::anyhow!("failed to create recognizer"))?;
    eprintln!("model loaded.");

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

    // Set stdin to non-blocking raw mode
    let mut old_termios = None::<libc::termios>;
    unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::tcgetattr(libc::STDIN_FILENO, &mut t) == 0 {
            old_termios = Some(t);
            let mut raw = t;
            raw.c_lflag &= !(libc::ICANON | libc::ECHO);
            raw.c_cc[libc::VMIN] = 0;
            raw.c_cc[libc::VTIME] = 0;
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw);
        }
    }

    let sample_rate = 16000_i32;
    let vad_window = 512_usize;

    let mut mic_buf = Vec::<f32>::new();
    let mut mic_offset = 0_usize;

    // audio_buf: current segment being decoded (cleared on emit)
    // full_buf:  all audio since last Enter (for ground-truth comparison)
    let mut audio_buf = Vec::<f32>::new();
    let mut full_buf = Vec::<f32>::new();

    let mut confirmed_text = String::new();
    let mut speech_active = false;
    let mut last_copied = 0_usize;

    // Silence tracking
    let mut silence_since: Option<Instant> = None;
    // Cached decode result from fast-silence trigger
    let mut cached_text = String::new();
    let mut cached_tokens: Vec<String> = Vec::new();
    let mut cached_has_punct = false;
    // Which silence thresholds have fired for this silence period
    let mut fast_fired = false;
    let mut slow_fired = false;

    eprintln!(
        "\nfast {}ms | slow {}ms | final {}ms",
        fast_silence_ms, slow_silence_ms, final_silence_ms
    );
    eprintln!("press Enter for full-decode comparison.  Ctrl+C to quit.\n");

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
                fast_fired = false;
                slow_fired = false;
                last_copied = mic_offset;
            } else if speech_active && !vad.detected() {
                speech_active = false;
                silence_since = Some(Instant::now());
                fast_fired = false;
                slow_fired = false;
            }
            mic_offset += vad_window;
        }

        // Trim idle mic buffer (no speech, nothing pending)
        if !speech_active && audio_buf.is_empty() && mic_buf.len() > 10 * vad_window {
            let trim = mic_buf.len() - 10 * vad_window;
            mic_offset = mic_offset.saturating_sub(trim);
            mic_buf = mic_buf[trim..].to_vec();
            last_copied = last_copied.saturating_sub(trim);
        }

        // Copy new audio into buffers while speech is active
        if speech_active && mic_offset > last_copied {
            let new = &mic_buf[last_copied..mic_offset];
            audio_buf.extend_from_slice(new);
            full_buf.extend_from_slice(new);
            last_copied = mic_offset;
        }

        // Also copy during silence (VAD may still be delivering trailing audio)
        if !speech_active && silence_since.is_some() && mic_offset > last_copied {
            let new = &mic_buf[last_copied..mic_offset];
            audio_buf.extend_from_slice(new);
            full_buf.extend_from_slice(new);
            last_copied = mic_offset;
        }

        // Consume completed VAD segments (don't double-count audio)
        while !vad.is_empty() {
            if let Some(_segment) = vad.front() {
                vad.pop();
            }
        }

        // Check for Enter key
        {
            let mut byte = [0u8; 1];
            if let Ok(1) = io::stdin().read(&mut byte) {
                if byte[0] == b'\n' || byte[0] == b'\r' {
                    if full_buf.is_empty() {
                        eprintln!("\n(no audio accumulated)\n");
                    } else {
                        let dur = full_buf.len() as f32 / sample_rate as f32;
                        eprintln!("\n--- FULL DECODE ({:.2}s) ---", dur);

                        let s = recognizer.create_stream();
                        s.accept_waveform(sample_rate, &full_buf);
                        recognizer.decode(&s);
                        let full_text = s
                            .get_result()
                            .map(|r| r.text.clone())
                            .unwrap_or_default();

                        eprintln!("{}", full_text);
                        eprintln!("---\n");

                        full_buf.clear();
                        audio_buf.clear();
                        confirmed_text.clear();
                        cached_text.clear();
                        cached_tokens.clear();
                        cached_has_punct = false;
                        silence_since = None;
                        fast_fired = false;
                        slow_fired = false;
                    }
                    continue;
                }
            }
        }

        // --- Silence-based triggers ---
        if let Some(silence_start) = silence_since {
            let silence_ms = silence_start.elapsed().as_millis() as u64;

            // FAST: trigger speculative decode
            if !fast_fired && silence_ms >= fast_silence_ms && !audio_buf.is_empty() {
                fast_fired = true;

                if let Some(result) = decode_buf(&recognizer, sample_rate, &audio_buf) {
                    let dur = audio_buf.len() as f32 / sample_rate as f32;
                    cached_has_punct = ends_with_sentence_punct(&result.tokens);
                    cached_text = result.text.clone();
                    cached_tokens = result.tokens.clone();
                    eprintln!("[decode] {:.1}s: {}", dur, cached_text);
                }
            }

            // SLOW: emit if ends with sentence punct
            if !slow_fired && fast_fired && silence_ms >= slow_silence_ms {
                slow_fired = true;

                if cached_has_punct && !cached_text.is_empty() {
                    eprintln!("[emit] {}", cached_text);
                    if !confirmed_text.is_empty() {
                        confirmed_text.push(' ');
                    }
                    confirmed_text.push_str(&cached_text);
                    audio_buf.clear();
                    cached_text.clear();
                    cached_tokens.clear();
                    cached_has_punct = false;
                } else if !cached_text.is_empty() {
                    eprintln!("[skip] no sentence ending: {}", cached_text);
                }
            }

            // FINAL: emit unconditionally
            if slow_fired && !cached_text.is_empty() && silence_ms >= final_silence_ms {
                // Re-decode in case more audio arrived since fast trigger
                if let Some(result) = decode_buf(&recognizer, sample_rate, &audio_buf) {
                    cached_text = result.text.clone();
                }

                if !cached_text.is_empty() {
                    eprintln!("[timeout] {}", cached_text);
                    if !confirmed_text.is_empty() {
                        confirmed_text.push(' ');
                    }
                    confirmed_text.push_str(&cached_text);
                }
                audio_buf.clear();
                cached_text.clear();
                cached_tokens.clear();
                cached_has_punct = false;
                silence_since = None;
            }
        }
    }

    // Final full decode if audio remains
    if !full_buf.is_empty() {
        eprintln!(
            "\n--- FULL DECODE ({:.2}s) ---",
            full_buf.len() as f32 / sample_rate as f32
        );
        let s = recognizer.create_stream();
        s.accept_waveform(sample_rate, &full_buf);
        recognizer.decode(&s);
        let full_text = s
            .get_result()
            .map(|r| r.text.clone())
            .unwrap_or_default();
        eprintln!("{}", full_text);
        eprintln!("---");
    }

    // Restore terminal
    if let Some(t) = old_termios {
        unsafe {
            libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t);
        }
    }

    eprintln!("\ndone.");
    Ok(())
}
