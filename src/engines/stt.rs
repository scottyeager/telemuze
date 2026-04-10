//! Speech-to-Text engine using sherpa-onnx.
//!
//! Uses the Parakeet TDT ONNX model via sherpa-onnx's OfflineRecognizer
//! for fast, accurate speech recognition.

use anyhow::{Context, Result};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info};

/// Result from STT transcription including token-level timing.
#[derive(Debug, Clone)]
pub struct SttResult {
    pub text: String,
    pub tokens: Vec<String>,
    /// Per-token timestamps in seconds (relative to start of audio segment).
    pub timestamps: Vec<f32>,
}

/// Wraps the sherpa-onnx OfflineRecognizer for Parakeet TDT inference.
pub struct SttEngine {
    /// Primary recognizer using modified_beam_search (supports hotwords).
    recognizer: OfflineRecognizer,
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
        max_active_paths: i32,
        blank_penalty: f32,
        num_threads: i32,
    ) -> Result<Self> {
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
        config.model_config.num_threads = num_threads;
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
             num_threads={num_threads})"
        );
        let recognizer = OfflineRecognizer::create(&config)
            .context("Failed to create sherpa-onnx OfflineRecognizer (beam search)")?;

        Ok(Self { recognizer })
    }

    /// Transcribe mono 16kHz f32 PCM audio to text.
    pub fn transcribe(&self, pcm_16khz: &[f32]) -> Result<SttResult> {
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
    ) -> Result<SttResult> {
        let audio_duration_secs = pcm_16khz.len() as f64 / 16_000.0;

        let stream = match hotwords {
            Some(hw) if !hw.is_empty() => {
                debug!(hotwords = %hw, "Passing hotwords to sherpa-onnx");
                self.recognizer.create_stream_with_hotwords(hw)
            }
            _ => self.recognizer.create_stream(),
        };
        stream.accept_waveform(16000, pcm_16khz);

        let start = Instant::now();
        self.recognizer.decode(&stream);
        info!(
            "Beam search decode completed in {:?} ({audio_duration_secs:.1}s audio)",
            start.elapsed()
        );

        let result = stream
            .get_result()
            .context("sherpa-onnx returned no recognition result")?;
        debug!(tokens = ?result.tokens, timestamps = ?result.timestamps, "Recognition tokens");
        Ok(SttResult {
            text: result.text,
            tokens: result.tokens,
            timestamps: result.timestamps.unwrap_or_default(),
        })
    }
}
