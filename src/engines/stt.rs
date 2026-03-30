//! Speech-to-Text engine using sherpa-onnx.
//!
//! Uses the Parakeet TDT ONNX model via sherpa-onnx's OfflineRecognizer
//! for fast, accurate speech recognition.

use anyhow::{Context, Result};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};
use std::path::Path;
use tracing::info;

/// Wraps the sherpa-onnx OfflineRecognizer for Parakeet TDT inference.
pub struct SttEngine {
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
    pub fn new(model_dir: &Path) -> Result<Self> {
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

        info!("Creating sherpa-onnx recognizer from {:?}", model_dir);
        let recognizer = OfflineRecognizer::create(&config)
            .context("Failed to create sherpa-onnx OfflineRecognizer")?;

        Ok(Self { recognizer })
    }

    /// Transcribe mono 16kHz f32 PCM audio to text.
    pub fn transcribe(&self, pcm_16khz: &[f32]) -> Result<String> {
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(16000, pcm_16khz);
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .context("sherpa-onnx returned no recognition result")?;
        Ok(result.text)
    }
}
