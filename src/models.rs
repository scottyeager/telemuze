//! Model downloading and management.
//!
//! Download logic adapted from [Handy](https://github.com/cjpais/Handy)
//! by CJ Pais, licensed under the MIT License. Original source:
//! `src-tauri/src/managers/model.rs`
//!
//! Key adaptations for Telemuze:
//! - Removed Tauri dependency (AppHandle, event emission)
//! - Scoped to Parakeet STT and Silero VAD models only
//! - Downloads individual files from upstream sources (HuggingFace, GitHub)
//!   instead of repackaged archives
//! - Progress reported via `tracing` instead of UI events

use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// The kind of model.
#[derive(Debug, Clone)]
pub enum ModelKind {
    /// Speech-to-text (Parakeet ONNX directory).
    Stt,
    /// Voice Activity Detection (Silero VAD single ONNX file).
    Vad,
    /// Large Language Model (GGUF file for native inference).
    Llm,
}

/// A single file to download as part of a model.
#[derive(Debug, Clone)]
struct ModelFile {
    url: &'static str,
    filename: &'static str,
}

/// Metadata for a downloadable model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: &'static str,
    #[allow(dead_code)]
    pub name: &'static str,
    /// Directory name (for multi-file models) or filename (for single-file).
    pub dirname: &'static str,
    /// Whether the model is a directory of files vs a single file.
    pub is_directory: bool,
    pub kind: ModelKind,
    pub is_downloaded: bool,
    /// Files that make up this model.
    files: &'static [ModelFile],
}

// ── Model registry ──────────────────────────────────────────────────────────

// Parakeet TDT 0.6B v3 ONNX (int8) — converted by istupakov
// Original model: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
// ONNX export: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
const PARAKEET_FILES: &[ModelFile] = &[
    ModelFile {
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx",
        filename: "encoder-model.int8.onnx",
    },
    ModelFile {
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.int8.onnx",
        filename: "decoder_joint-model.int8.onnx",
    },
    ModelFile {
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/nemo128.onnx",
        filename: "nemo128.onnx",
    },
    ModelFile {
        url: "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt",
        filename: "vocab.txt",
    },
];

const SILERO_VAD_FILES: &[ModelFile] = &[ModelFile {
    url: "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
    filename: "silero_vad.onnx",
}];

const PARAKEET: ModelInfo = ModelInfo {
    id: "parakeet-tdt-0.6b-v3",
    name: "Parakeet TDT 0.6B v3 (int8)",
    dirname: "parakeet-tdt-0.6b-v3-int8",
    is_directory: true,
    kind: ModelKind::Stt,
    is_downloaded: false,
    files: PARAKEET_FILES,
};

const SILERO_VAD: ModelInfo = ModelInfo {
    id: "silero-vad",
    name: "Silero VAD v4",
    dirname: "silero_vad.onnx",
    is_directory: false,
    kind: ModelKind::Vad,
    is_downloaded: false,
    files: SILERO_VAD_FILES,
};

// Qwen3.5-0.8B GGUF — quantized by lmstudio-community
// Original model: https://huggingface.co/Qwen/Qwen3.5-0.8B
// GGUF: https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF
const QWEN_0_8B_FILES: &[ModelFile] = &[ModelFile {
    url: "https://huggingface.co/lmstudio-community/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
    filename: "Qwen3.5-0.8B-Q8_0.gguf",
}];

const QWEN_0_8B: ModelInfo = ModelInfo {
    id: "qwen3.5-0.8b",
    name: "Qwen3.5-0.8B (Q8_0)",
    dirname: "Qwen3.5-0.8B-Q8_0.gguf",
    is_directory: false,
    kind: ModelKind::Llm,
    is_downloaded: false,
    files: QWEN_0_8B_FILES,
};

// Qwen3.5-2B GGUF — quantized by lmstudio-community
// Original model: https://huggingface.co/Qwen/Qwen3.5-2B
// GGUF: https://huggingface.co/lmstudio-community/Qwen3.5-2B-GGUF
const QWEN_2B_FILES: &[ModelFile] = &[ModelFile {
    url: "https://huggingface.co/lmstudio-community/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf",
    filename: "Qwen3.5-2B-Q8_0.gguf",
}];

const QWEN_2B: ModelInfo = ModelInfo {
    id: "qwen3.5-2b",
    name: "Qwen3.5-2B (Q8_0)",
    dirname: "Qwen3.5-2B-Q8_0.gguf",
    is_directory: false,
    kind: ModelKind::Llm,
    is_downloaded: false,
    files: QWEN_2B_FILES,
};

/// All models that Telemuze knows about.
fn default_models() -> HashMap<&'static str, ModelInfo> {
    let mut m = HashMap::new();
    m.insert(PARAKEET.id, PARAKEET);
    m.insert(SILERO_VAD.id, SILERO_VAD);
    m.insert(QWEN_0_8B.id, QWEN_0_8B);
    m.insert(QWEN_2B.id, QWEN_2B);
    m
}

// ── ModelManager ────────────────────────────────────────────────────────────

/// Manages downloading and locating AI models on disk.
pub struct ModelManager {
    models_dir: PathBuf,
    models: Mutex<HashMap<&'static str, ModelInfo>>,
    cancel_flags: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl ModelManager {
    /// Create a new manager rooted at `models_dir`.
    ///
    /// The directory is created if it does not exist. Download status is
    /// detected from existing files on disk.
    pub fn new(models_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&models_dir)
            .with_context(|| format!("Failed to create models directory: {}", models_dir.display()))?;

        let mgr = Self {
            models_dir,
            models: Mutex::new(default_models()),
            cancel_flags: Arc::new(Mutex::new(HashMap::new())),
        };
        mgr.update_download_status()?;
        Ok(mgr)
    }

    /// Directory where models are stored.
    #[allow(dead_code)]
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Return the on-disk path for a model.
    #[allow(dead_code)]
    pub fn model_path(&self, model_id: &str) -> Result<PathBuf> {
        let models = self.models.lock().unwrap();
        let info = models.get(model_id).context("Unknown model id")?;
        Ok(self.models_dir.join(info.dirname))
    }

    /// Return the STT model directory path (first downloaded STT model).
    pub fn stt_model_path(&self) -> Result<PathBuf> {
        let models = self.models.lock().unwrap();
        for info in models.values() {
            if matches!(info.kind, ModelKind::Stt) && info.is_downloaded {
                return Ok(self.models_dir.join(info.dirname));
            }
        }
        bail!("No STT model downloaded")
    }

    /// Return the VAD model file path (first downloaded VAD model).
    pub fn vad_model_path(&self) -> Result<PathBuf> {
        let models = self.models.lock().unwrap();
        for info in models.values() {
            if matches!(info.kind, ModelKind::Vad) && info.is_downloaded {
                return Ok(self.models_dir.join(info.dirname));
            }
        }
        bail!("No VAD model downloaded")
    }

    /// Return the on-disk path for a specific LLM model by id.
    pub fn llm_model_path(&self, model_id: &str) -> Result<PathBuf> {
        let models = self.models.lock().unwrap();
        let info = models.get(model_id).context("Unknown LLM model id")?;
        if !info.is_downloaded {
            bail!("LLM model {} is not downloaded", model_id);
        }
        Ok(self.models_dir.join(info.dirname))
    }

    /// Check whether all required models (at least one STT + one VAD) are
    /// present on disk. LLM is optional (native inference is not required
    /// when an HTTP backend is configured).
    pub fn all_models_available(&self) -> bool {
        let models = self.models.lock().unwrap();
        let has_stt = models
            .values()
            .any(|m| matches!(m.kind, ModelKind::Stt) && m.is_downloaded);
        let has_vad = models
            .values()
            .any(|m| matches!(m.kind, ModelKind::Vad) && m.is_downloaded);
        has_stt && has_vad
    }

    /// Download a specific LLM model if not already present.
    pub async fn ensure_llm_model(&self, model_id: &str) -> Result<()> {
        let models = self.models.lock().unwrap();
        let info = models.get(model_id).context("Unknown LLM model id")?;
        if info.is_downloaded {
            return Ok(());
        }
        drop(models);
        self.download_model(model_id).await
    }

    /// Ensure all required (non-LLM) models are downloaded. Downloads any
    /// that are missing. LLM models are handled separately via
    /// `ensure_llm_model()`.
    pub async fn ensure_models(&self) -> Result<()> {
        let to_download: Vec<String> = {
            let models = self.models.lock().unwrap();
            models
                .iter()
                .filter(|(_, info)| !info.is_downloaded && !matches!(info.kind, ModelKind::Llm))
                .map(|(id, _)| id.to_string())
                .collect()
        };

        for id in to_download {
            self.download_model(&id).await?;
        }

        Ok(())
    }

    /// Download a single model by id.
    ///
    /// For directory models, creates the target directory and downloads each
    /// file into it. For single-file models, downloads directly. Supports
    /// resumable downloads via HTTP Range requests.
    pub async fn download_model(&self, model_id: &str) -> Result<()> {
        let (dirname, is_directory, files) = {
            let models = self.models.lock().unwrap();
            let info = models.get(model_id).context("Unknown model id")?;
            if info.is_downloaded {
                info!("Model {} already downloaded", model_id);
                return Ok(());
            }
            (
                info.dirname.to_string(),
                info.is_directory,
                info.files.to_vec(),
            )
        };

        // Set up cancellation flag
        let cancel_flag = Arc::new(AtomicBool::new(false));
        {
            let mut flags = self.cancel_flags.lock().unwrap();
            flags.insert(model_id.to_string(), cancel_flag.clone());
        }

        let base_dir = if is_directory {
            let dir = self.models_dir.join(&dirname);
            fs::create_dir_all(&dir)
                .with_context(|| format!("Failed to create model directory: {}", dir.display()))?;
            dir
        } else {
            self.models_dir.clone()
        };

        let client = reqwest::Client::new();

        for model_file in &files {
            if cancel_flag.load(Ordering::Relaxed) {
                info!("Download cancelled for model {}", model_id);
                // Clean up partial directory model
                if is_directory {
                    fs::remove_dir_all(&base_dir).ok();
                }
                self.cleanup_cancel(model_id);
                bail!("Download cancelled");
            }

            let dest_path = base_dir.join(model_file.filename);

            // Skip files that are already fully downloaded
            if dest_path.exists() {
                debug!("File {} already exists, skipping", model_file.filename);
                continue;
            }

            self.download_file(
                &client,
                model_file.url,
                &dest_path,
                model_file.filename,
                model_id,
                &cancel_flag,
            )
            .await?;
        }

        // Clean up cancel flag and update status
        {
            let mut flags = self.cancel_flags.lock().unwrap();
            flags.remove(model_id);
        }
        self.update_download_status()?;

        info!("Model {} ready", model_id);
        Ok(())
    }

    /// Cancel an in-progress download.
    #[allow(dead_code)]
    pub fn cancel_download(&self, model_id: &str) {
        let flags = self.cancel_flags.lock().unwrap();
        if let Some(flag) = flags.get(model_id) {
            flag.store(true, Ordering::Relaxed);
            info!("Cancellation requested for model {}", model_id);
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────

    /// Download a single file with resume support.
    async fn download_file(
        &self,
        client: &reqwest::Client,
        url: &str,
        dest_path: &Path,
        display_name: &str,
        model_id: &str,
        cancel_flag: &AtomicBool,
    ) -> Result<()> {
        let partial_path = PathBuf::from(format!("{}.partial", dest_path.display()));

        // Check for existing partial download
        let existing_size = if partial_path.exists() {
            fs::metadata(&partial_path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        info!(
            "[{}] Downloading {} (existing: {} bytes)",
            model_id, display_name, existing_size
        );

        let mut request = client.get(url);
        if existing_size > 0 {
            request = request.header("Range", format!("bytes={}-", existing_size));
        }

        let response = request.send().await.context("Download request failed")?;

        if !response.status().is_success()
            && response.status() != reqwest::StatusCode::PARTIAL_CONTENT
        {
            bail!(
                "Download failed with status {} for {}",
                response.status(),
                display_name,
            );
        }

        // Determine if we're resuming or starting fresh
        let (total_size, resuming) = if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
            let total = response
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.split('/').last())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            (total, true)
        } else {
            let total = response.content_length().unwrap_or(0);
            (total, false)
        };

        let mut file = if resuming {
            fs::OpenOptions::new()
                .append(true)
                .open(&partial_path)
                .with_context(|| {
                    format!("Failed to open partial file: {}", partial_path.display())
                })?
        } else {
            File::create(&partial_path).with_context(|| {
                format!("Failed to create download file: {}", partial_path.display())
            })?
        };

        let mut downloaded = if resuming { existing_size } else { 0 };
        let mut last_log = Instant::now();
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            if cancel_flag.load(Ordering::Relaxed) {
                bail!("Download cancelled");
            }

            let chunk = chunk.context("Error reading download stream")?;
            file.write_all(&chunk)
                .context("Failed to write to download file")?;
            downloaded += chunk.len() as u64;

            // Log progress at most every 2 seconds
            if last_log.elapsed() >= Duration::from_secs(2) {
                let pct = if total_size > 0 {
                    (downloaded as f64 / total_size as f64) * 100.0
                } else {
                    0.0
                };
                info!(
                    "[{}] {}: {:.1}% ({}/{} bytes)",
                    model_id, display_name, pct, downloaded, total_size
                );
                last_log = Instant::now();
            }
        }

        drop(file);

        // Move from .partial to final location
        fs::rename(&partial_path, dest_path).with_context(|| {
            format!(
                "Failed to move {} to {}",
                partial_path.display(),
                dest_path.display()
            )
        })?;

        info!("[{}] {} complete", model_id, display_name);
        Ok(())
    }

    /// Scan the models directory and update `is_downloaded` for each model.
    fn update_download_status(&self) -> Result<()> {
        let mut models = self.models.lock().unwrap();
        for info in models.values_mut() {
            let path = self.models_dir.join(info.dirname);
            info.is_downloaded = if info.is_directory {
                // A directory model is complete when all expected files exist
                path.is_dir()
                    && info
                        .files
                        .iter()
                        .all(|f| path.join(f.filename).is_file())
            } else {
                path.is_file()
            };

            if info.is_downloaded {
                debug!("Model {} found at {}", info.id, path.display());
            }
        }
        Ok(())
    }

    fn cleanup_cancel(&self, model_id: &str) {
        let mut flags = self.cancel_flags.lock().unwrap();
        flags.remove(model_id);
    }
}
