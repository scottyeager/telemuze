//! Speech-to-Text engine using sherpa-onnx.
//!
//! Uses the Parakeet TDT ONNX model via sherpa-onnx's OfflineRecognizer
//! for fast, accurate speech recognition.

use anyhow::{Context, Result};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};
use std::path::Path;
use std::time::Duration;
use tracing::{debug, info, warn};

/// Wraps the sherpa-onnx OfflineRecognizer for Parakeet TDT inference.
pub struct SttEngine {
    recognizer: OfflineRecognizer,
    decode_timeout: Option<Duration>,
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
    pub fn new(model_dir: &Path, hotwords_score: f32, decode_timeout_secs: u64) -> Result<Self> {
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
        config.model_config.num_threads = 4;
        config.decoding_method = Some("modified_beam_search".into());
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
            "Creating sherpa-onnx recognizer from {:?} (decoding=modified_beam_search, hotwords_score={hotwords_score})",
            model_dir
        );
        let recognizer = OfflineRecognizer::create(&config)
            .context("Failed to create sherpa-onnx OfflineRecognizer")?;

        if let Some(t) = decode_timeout {
            info!("STT decode timeout: {t:?}");
        }

        Ok(Self { recognizer, decode_timeout })
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
        let stream = match hotwords {
            Some(hw) if !hw.is_empty() => {
                debug!(hotwords = %hw, "Passing hotwords to sherpa-onnx");
                self.recognizer.create_stream_with_hotwords(hw)
            }
            _ => self.recognizer.create_stream(),
        };
        stream.accept_waveform(16000, pcm_16khz);

        match self.decode_timeout {
            Some(timeout) => {
                // Run decode on a background thread with a timeout to
                // catch infinite beam search loops (e.g. from extreme
                // hotword scores).
                //
                // Cast pointers to usize so they are Send. The
                // recognizer lives as long as the server. The stream
                // is either dropped normally (success) or deliberately
                // leaked (timeout) so both pointers stay valid.
                let rec_ptr = &self.recognizer as *const OfflineRecognizer as usize;
                let stream_ptr = &stream as *const sherpa_onnx::OfflineStream as usize;

                let (tx, rx) = std::sync::mpsc::channel();

                std::thread::spawn(move || {
                    unsafe {
                        let rec = &*(rec_ptr as *const OfflineRecognizer);
                        let s = &*(stream_ptr as *const sherpa_onnx::OfflineStream);
                        rec.decode(s);
                    }
                    let _ = tx.send(());
                });

                if rx.recv_timeout(timeout).is_err() {
                    // Leak the stream so the background thread (which
                    // is still running decode) doesn't hit freed memory.
                    std::mem::forget(stream);
                    warn!(
                        "STT decode timed out after {timeout:?} — aborting request. \
                         This usually means hotword scores are too high, causing \
                         an infinite beam search loop. The server may need to be \
                         restarted if the stuck thread does not finish."
                    );
                    anyhow::bail!(
                        "STT decode timed out after {timeout:?} — \
                         possible infinite beam search loop"
                    );
                }
            }
            None => {
                self.recognizer.decode(&stream);
            }
        }

        let result = stream
            .get_result()
            .context("sherpa-onnx returned no recognition result")?;
        debug!(tokens = ?result.tokens, "Recognition tokens");
        Ok(result.text)
    }
}
