// Local command detection using Parakeet-TDT 110m transducer with hotword boosting.
//
// Replaces the KeywordSpotter approach with an OfflineRecognizer that decodes
// full speech segments and matches against command vocabulary.  The 110m model
// handles all command forms well when boosting command verbs, modifiers, and
// special keys (but not individual letters).

use anyhow::{bail, Context, Result};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

// ── Model constants ────────────────────────────────────────────────────────

const MODEL_DIR_NAME: &str = "sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000";
const MODEL_DIR_NAME_INT8: &str = "sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8";
const MODEL_ARCHIVE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000.tar.bz2";

const MODEL_ENCODER: &str = "encoder.onnx";
const MODEL_ENCODER_INT8: &str = "encoder.int8.onnx";
const MODEL_DECODER: &str = "decoder.onnx";
const MODEL_DECODER_INT8: &str = "decoder.int8.onnx";
const MODEL_JOINER: &str = "joiner.onnx";
const MODEL_JOINER_INT8: &str = "joiner.int8.onnx";
const MODEL_TOKENS: &str = "tokens.txt";
const MODEL_BPE_VOCAB: &str = "bpe.vocab";

// ── Defaults ───────────────────────────────────────────────────────────────

pub const DEFAULT_CMD_BOOST_FIRST: f32 = 3.0;
pub const DEFAULT_CMD_BOOST_PHRASE: f32 = 2.0;
pub const DEFAULT_CMD_BOOST_VOCAB: f32 = 1.0;
pub const DEFAULT_CMD_FIRST_PASS_MS: u32 = 800;
pub const DEFAULT_CMD_PREFILL_MS: u32 = 300;
pub const DEFAULT_CMD_SILENCE_MS: u32 = 500;
pub const DEFAULT_CMD_SPEC_SILENCE_MS: u32 = 32;
pub const DEFAULT_CMD_THREADS: i32 = 4;
pub const DEFAULT_CMD_BEAM_WIDTH: i32 = 4;

// ── Model management ──────────────────────────────────────────────────────

fn models_dir() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("telemuze/models")
}

/// Default model directory, preferring int8 if present.
pub fn default_model_dir() -> PathBuf {
    let base = models_dir();
    let int8_dir = base.join(MODEL_DIR_NAME_INT8);
    if int8_dir.join(MODEL_ENCODER_INT8).exists() {
        int8_dir
    } else {
        base.join(MODEL_DIR_NAME)
    }
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
    info!("Generated bpe.vocab ({} tokens)", lines.len());
    Ok(())
}

/// Download and extract the Parakeet TDT 110m model if not present.
pub fn ensure_model(model_dir: &Path) -> Result<()> {
    if has_model_files(model_dir) {
        generate_bpe_vocab(model_dir)?;
        return Ok(());
    }

    let parent = model_dir.parent().context("Model dir has no parent")?;
    std::fs::create_dir_all(parent)?;

    eprintln!("Downloading Parakeet-TDT 110m model...");
    let response = reqwest::blocking::get(MODEL_ARCHIVE_URL)
        .context("Failed to download model")?;
    if !response.status().is_success() {
        bail!("Download failed with status {}", response.status());
    }

    let bytes = response.bytes().context("Failed to read response body")?;
    eprintln!("Downloaded {} MB, extracting...", bytes.len() / 1_000_000);

    let decoder = bzip2::read::BzDecoder::new(bytes.as_ref());
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(parent).context("Failed to extract model archive")?;

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

// ── Recognizer initialization ─────────────────────────────────────────────

/// Configuration for the command recognizer.
pub struct CmdConfig {
    pub model_dir: PathBuf,
    pub num_threads: i32,
    pub beam_width: i32,
}

/// Initialize an OfflineRecognizer for command detection.
///
/// The recognizer-level `hotwords_score` is left at its default since every
/// word in the hotwords string carries an explicit per-word score.
pub fn init_recognizer(cfg: &CmdConfig) -> Result<OfflineRecognizer> {
    ensure_model(&cfg.model_dir)?;

    // Prefer int8 model files when present
    let resolve = |fp32: &str, int8: &str| -> String {
        let int8_path = cfg.model_dir.join(int8);
        if int8_path.exists() {
            int8_path.to_string_lossy().into_owned()
        } else {
            cfg.model_dir.join(fp32).to_string_lossy().into_owned()
        }
    };

    let encoder_path = resolve(MODEL_ENCODER, MODEL_ENCODER_INT8);
    let decoder_path = resolve(MODEL_DECODER, MODEL_DECODER_INT8);
    let joiner_path = resolve(MODEL_JOINER, MODEL_JOINER_INT8);
    let using_int8 = encoder_path.contains("int8");

    debug!("CMD encoder: {encoder_path}");
    debug!("CMD decoder: {decoder_path}");
    debug!("CMD joiner:  {joiner_path}");

    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer.encoder = Some(encoder_path);
    config.model_config.transducer.decoder = Some(decoder_path);
    config.model_config.transducer.joiner = Some(joiner_path);
    config.model_config.tokens = Some(
        cfg.model_dir.join(MODEL_TOKENS).to_string_lossy().into_owned(),
    );
    config.model_config.model_type = Some("nemo_transducer".to_string());
    config.model_config.num_threads = cfg.num_threads;
    config.decoding_method = Some("modified_beam_search".to_string());
    config.max_active_paths = cfg.beam_width;

    // Enable BPE hotword encoding if bpe.vocab exists
    let bpe_vocab = cfg.model_dir.join(MODEL_BPE_VOCAB);
    if bpe_vocab.exists() {
        config.model_config.modeling_unit = Some("bpe".into());
        config.model_config.bpe_vocab = Some(bpe_vocab.to_string_lossy().into_owned());
        debug!("BPE hotword encoding enabled");
    }

    let quant_str = if using_int8 { "INT8" } else { "FP32" };
    info!("Loading Parakeet-TDT 110m ({quant_str}) from {}", cfg.model_dir.display());

    OfflineRecognizer::create(&config)
        .context("Failed to create command recognizer — check model paths")
}

/// Initialize a pair of recognizers for two-pass command detection.
/// Returns (pass1_recognizer, pass2_recognizer).
pub fn init_recognizer_pair(cfg: &CmdConfig) -> Result<(OfflineRecognizer, OfflineRecognizer)> {
    let pass1 = init_recognizer(cfg)?;
    let pass2 = init_recognizer(cfg)?;
    Ok((pass1, pass2))
}


/// Normalize text for command matching: lowercase, strip punctuation, collapse whitespace.
pub fn normalize(text: &str) -> String {
    text.trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
