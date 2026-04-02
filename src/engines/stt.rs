//! Speech-to-Text engine using sherpa-onnx.
//!
//! Uses the Parakeet TDT ONNX model via sherpa-onnx's OfflineRecognizer
//! for fast, accurate speech recognition.

use anyhow::{Context, Result};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Maximum number of zombie decode threads before we skip beam search entirely.
const MAX_ZOMBIES: u32 = 3;

/// Wraps the sherpa-onnx OfflineRecognizer for Parakeet TDT inference.
pub struct SttEngine {
    /// Primary recognizer using modified_beam_search (supports hotwords).
    recognizer: OfflineRecognizer,
    /// Fallback recognizer using greedy_search (no hotwords, but reliable).
    greedy_recognizer: OfflineRecognizer,
    decode_timeout: Option<Duration>,
    /// Number of zombie beam search threads still running after timeout.
    zombie_count: AtomicU32,
}

// Safety: The underlying ONNX runtime session is thread-safe for inference.
// We guard access with a Mutex in AppState.
unsafe impl Send for SttEngine {}
unsafe impl Sync for SttEngine {}

impl SttEngine {
    /// Load a Parakeet TDT model from the given directory.
    ///
    /// The directory should contain:
    /// - encoder.int8.onnx
    /// - decoder.int8.onnx
    /// - joiner.int8.onnx
    /// - tokens.txt
    /// - bpe.vocab (for hotword support)
    pub fn new(
        model_dir: &Path,
        hotwords_score: f32,
        decode_timeout_secs: u64,
        max_active_paths: i32,
        blank_penalty: f32,
    ) -> Result<Self> {
        let decode_timeout = if decode_timeout_secs > 0 {
            Some(Duration::from_secs(decode_timeout_secs))
        } else {
            None
        };
        let mut config = OfflineRecognizerConfig::default();
        config.model_config.transducer.encoder =
            Some(model_dir.join("encoder.int8.onnx").to_string_lossy().into_owned());
        config.model_config.transducer.decoder =
            Some(model_dir.join("decoder.int8.onnx").to_string_lossy().into_owned());
        config.model_config.transducer.joiner =
            Some(model_dir.join("joiner.int8.onnx").to_string_lossy().into_owned());
        config.model_config.tokens =
            Some(model_dir.join("tokens.txt").to_string_lossy().into_owned());
        config.model_config.model_type = Some("nemo_transducer".into());
        // Use fewer threads for beam search so zombie threads have less CPU impact.
        config.model_config.num_threads = 2;
        config.decoding_method = Some("modified_beam_search".into());
        config.max_active_paths = max_active_paths;
        config.blank_penalty = blank_penalty;
        config.hotwords_score = hotwords_score;

        // Enable BPE hotword encoding so sherpa-onnx tokenizes hotwords
        // using the SentencePiece vocabulary rather than the default
        // "cjkchar" mode which breaks ▁-prefixed BPE tokens.
        let bpe_vocab_path = model_dir.join("bpe.vocab");
        if bpe_vocab_path.exists() {
            config.model_config.modeling_unit = Some("bpe".into());
            config.model_config.bpe_vocab =
                Some(bpe_vocab_path.to_string_lossy().into_owned());
            info!("BPE hotword encoding enabled via {}", bpe_vocab_path.display());
        }

        info!(
            "Creating beam_search recognizer (hotwords_score={hotwords_score}, \
             max_active_paths={max_active_paths}, blank_penalty={blank_penalty}, \
             num_threads=2, decode_timeout={decode_timeout_secs}s)"
        );
        let recognizer = OfflineRecognizer::create(&config)
            .context("Failed to create sherpa-onnx OfflineRecognizer (beam search)")?;

        // Create a second recognizer with greedy_search for fallback.
        // Greedy search doesn't support hotwords but is immune to the
        // beam search pathologies that can cause hangs.
        // Give the greedy fallback full thread count since it's our safety net.
        let mut greedy_config = config.clone();
        greedy_config.decoding_method = Some("greedy_search".into());
        greedy_config.hotwords_score = 0.0;
        greedy_config.model_config.num_threads = 4;
        info!("Creating greedy_search fallback recognizer (num_threads=4)");
        let greedy_recognizer = OfflineRecognizer::create(&greedy_config)
            .context("Failed to create sherpa-onnx OfflineRecognizer (greedy search)")?;

        Ok(Self {
            recognizer,
            greedy_recognizer,
            decode_timeout,
            zombie_count: AtomicU32::new(0),
        })
    }

    /// Transcribe mono 16kHz f32 PCM audio to text.
    pub fn transcribe(&self, pcm_16khz: &[f32]) -> Result<String> {
        self.transcribe_with_hotwords(pcm_16khz, None)
    }

    /// Transcribe with optional per-request hotwords.
    ///
    /// `hotwords` should be one word/phrase per line in sherpa-onnx format.
    /// Sherpa-onnx handles BPE tokenization internally when `bpe.vocab` is
    /// configured, so hotwords can be plain words (e.g. "PRESS ENTER").
    pub fn transcribe_with_hotwords(
        &self,
        pcm_16khz: &[f32],
        hotwords: Option<&str>,
    ) -> Result<String> {
        let audio_duration_secs = pcm_16khz.len() as f64 / 16_000.0;
        // If too many zombie beam threads have accumulated, skip beam search
        // entirely to avoid further CPU waste.
        let zombies = self.zombie_count.load(Ordering::Relaxed);
        if zombies >= MAX_ZOMBIES {
            error!(
                "Skipping beam search: {zombies} zombie threads active — \
                 using greedy search directly. Restart server to reclaim CPU."
            );
            return self.greedy_decode(pcm_16khz, audio_duration_secs);
        }

        // Phase 1: Try decoding with hotwords (if any) using beam search.
        let stream = match hotwords {
            Some(hw) if !hw.is_empty() => {
                debug!(hotwords = %hw, "Passing hotwords to sherpa-onnx");
                self.recognizer.create_stream_with_hotwords(hw)
            }
            _ => self.recognizer.create_stream(),
        };
        stream.accept_waveform(16000, pcm_16khz);

        match self.decode_with_timeout(&stream, audio_duration_secs) {
            Ok(()) => {
                let result = stream
                    .get_result()
                    .context("sherpa-onnx returned no recognition result")?;
                debug!(tokens = ?result.tokens, "Recognition tokens");
                Ok(result.text)
            }
            Err(_) => {
                // Leak the stream so the background thread (which
                // is still running decode) doesn't hit freed memory.
                std::mem::forget(stream);
                let new_zombies = self.zombie_count.fetch_add(1, Ordering::Relaxed) + 1;
                warn!(
                    "Beam search decode timed out on {audio_duration_secs:.1}s audio \
                     — falling back to greedy search (zombie threads: {new_zombies}). \
                     Hotwords were: {:?}",
                    hotwords.unwrap_or("(none)")
                );

                // Phase 2: Fall back to greedy_search which is immune
                // to the beam search pathologies that cause hangs.
                self.greedy_decode(pcm_16khz, audio_duration_secs)
            }
        }
    }

    /// Decode audio using the greedy fallback recognizer.
    fn greedy_decode(&self, pcm_16khz: &[f32], audio_duration_secs: f64) -> Result<String> {
        let fallback_stream = self.greedy_recognizer.create_stream();
        fallback_stream.accept_waveform(16000, pcm_16khz);
        let fallback_start = Instant::now();
        self.greedy_recognizer.decode(&fallback_stream);
        info!(
            "Greedy fallback decode completed in {:?} ({audio_duration_secs:.1}s audio)",
            fallback_start.elapsed()
        );

        let result = fallback_stream
            .get_result()
            .context("sherpa-onnx returned no recognition result")?;
        debug!(tokens = ?result.tokens, "Greedy fallback recognition tokens");
        Ok(result.text)
    }

    /// Run `recognizer.decode()` on a background thread with a timeout.
    ///
    /// Returns `Ok(())` if decode completed, `Err(())` if it timed out.
    /// On timeout the stream is NOT freed here — the caller must leak it
    /// to keep the background thread's pointers valid.
    fn decode_with_timeout(
        &self,
        stream: &sherpa_onnx::OfflineStream,
        audio_duration_secs: f64,
    ) -> std::result::Result<(), ()> {
        let timeout = match self.decode_timeout {
            Some(t) => t,
            None => {
                let start = Instant::now();
                self.recognizer.decode(stream);
                info!(
                    "Beam search decode completed in {:?} ({audio_duration_secs:.1}s audio)",
                    start.elapsed()
                );
                return Ok(());
            }
        };

        let rec_ptr = &self.recognizer as *const OfflineRecognizer as usize;
        let stream_ptr = stream as *const sherpa_onnx::OfflineStream as usize;

        let (tx, rx) = std::sync::mpsc::channel();
        let start = Instant::now();

        std::thread::spawn(move || {
            unsafe {
                let rec = &*(rec_ptr as *const OfflineRecognizer);
                let s = &*(stream_ptr as *const sherpa_onnx::OfflineStream);
                rec.decode(s);
            }
            let _ = tx.send(());
        });

        if rx.recv_timeout(timeout).is_err() {
            warn!(
                "STT decode timed out after {:?} on {audio_duration_secs:.1}s audio",
                start.elapsed()
            );
            return Err(());
        }

        info!(
            "Beam search decode completed in {:?} ({audio_duration_secs:.1}s audio)",
            start.elapsed()
        );
        Ok(())
    }
}
