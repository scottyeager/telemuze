// Keyword spotting support using sherpa-onnx KeywordSpotter.
//
// Provides BPE and phone-based tokenization, model management, and
// KeywordSpotter initialization.  Two models are supported:
//
//  - **Gigaspeech** (English, BPE tokenization)
//  - **ZhEn** (Chinese + English, phone/pinyin tokenization)

use anyhow::{bail, Context, Result};
use sentencepiece_model::SentencePieceModel;
use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig};
use std::collections::HashMap;
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use tracing::debug;

pub const SAMPLE_RATE: u32 = 16_000;

// ── Model catalogue ────────────────────────────────────────────────────────

/// Supported KWS model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KwsModel {
    /// English-only, BPE tokenization (GigaSpeech corpus).
    Gigaspeech,
    /// Chinese + English, phone/pinyin tokenization.
    ZhEn,
}

impl KwsModel {
    pub fn dir_name(&self) -> &str {
        match self {
            Self::Gigaspeech => "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
            Self::ZhEn => "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
        }
    }

    pub fn url(&self) -> &str {
        match self {
            Self::Gigaspeech => "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2",
            Self::ZhEn => "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2",
        }
    }

    /// Optional filename substring used to pick a consistent set of model
    /// files when multiple chunk-size variants exist in the same directory.
    pub fn chunk_hint(&self) -> Option<&str> {
        match self {
            Self::Gigaspeech => None,
            Self::ZhEn => Some("chunk-16"),
        }
    }
}

impl std::fmt::Display for KwsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gigaspeech => write!(f, "gigaspeech"),
            Self::ZhEn => write!(f, "zh-en"),
        }
    }
}

impl std::str::FromStr for KwsModel {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "gigaspeech" => Ok(Self::Gigaspeech),
            "zh-en" | "zhen" => Ok(Self::ZhEn),
            _ => bail!("Unknown KWS model '{s}' — expected 'gigaspeech' or 'zh-en'"),
        }
    }
}

/// The default model used when none is specified.
pub const DEFAULT_MODEL: KwsModel = KwsModel::ZhEn;

/// Try to identify the model variant from a directory path by matching its
/// name against the known model directory names.  Returns `None` if the
/// directory name doesn't match any known model.
pub fn detect_model(model_dir: &Path) -> Option<KwsModel> {
    let name = model_dir.file_name()?.to_str()?;
    [KwsModel::Gigaspeech, KwsModel::ZhEn]
        .into_iter()
        .find(|m| name == m.dir_name())
}

// ── BPE tokenizer ───────────────────────────────────────────────────────────

/// A simple BPE encoder using a sentencepiece model file.
pub struct BpeTokenizer {
    /// token string → (piece_id, score)
    vocab: HashMap<String, (u32, f32)>,
    /// Whether the model vocabulary is uppercase (auto-detected)
    uppercase: bool,
}

impl BpeTokenizer {
    pub fn load(model_path: &Path) -> Result<Self> {
        let model = SentencePieceModel::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load BPE model: {e}"))?;

        let mut vocab = HashMap::new();
        let mut upper_count = 0u32;
        let mut lower_count = 0u32;
        for (id, piece) in model.pieces().iter().enumerate() {
            if let Some(ref s) = piece.piece {
                vocab.insert(s.clone(), (id as u32, piece.score.unwrap_or(0.0)));
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
    pub fn encode(&self, text: &str) -> Vec<String> {
        let text = if self.uppercase { text.to_uppercase() } else { text.to_lowercase() };
        let normalized = format!("▁{}", text.replace(' ', "▁"));

        let mut symbols: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();

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

// ── Phone tokenizer (for ZhEn model) ───────────────────────────────────────

/// A phone-based encoder using a CMU-style pronunciation lexicon (`en.phone`).
///
/// Each word is looked up in the lexicon and expanded to its ARPAbet phone
/// sequence.  The lexicon uses the format `WORD PHONE1 PHONE2 …` with one
/// entry per line and uppercase words.
pub struct PhoneTokenizer {
    /// UPPERCASE_WORD → vec of phone strings
    lexicon: HashMap<String, Vec<String>>,
}

impl PhoneTokenizer {
    pub fn load(phone_path: &Path) -> Result<Self> {
        let file = fs::File::open(phone_path)
            .with_context(|| format!("Cannot open phone lexicon {}", phone_path.display()))?;
        let reader = std::io::BufReader::new(file);

        let mut lexicon = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let word = match parts.next() {
                Some(w) => w.to_uppercase(),
                None => continue,
            };
            let phones: Vec<String> = parts.map(|s| s.to_string()).collect();
            if !phones.is_empty() {
                // Keep first pronunciation when duplicates exist (e.g. 'READ')
                lexicon.entry(word).or_insert(phones);
            }
        }

        Ok(Self { lexicon })
    }

    /// Encode a phrase into its phone sequence.  Returns an error if any word
    /// is not found in the lexicon.
    pub fn encode(&self, text: &str) -> Result<Vec<String>> {
        let mut phones = Vec::new();
        for word in text.split_whitespace() {
            let key = word.to_uppercase();
            match self.lexicon.get(&key) {
                Some(p) => phones.extend(p.iter().cloned()),
                None => bail!("Word '{word}' not found in phone lexicon"),
            }
        }
        Ok(phones)
    }
}

// ── Model management ────────────────────────────────────────────────────────

/// Find the encoder/decoder/joiner/tokens files in model_dir, preferring int8
/// variants.  When `chunk_hint` is `Some("chunk-16")`, only files whose name
/// contains that substring are considered (needed for models that ship with
/// multiple chunk-size variants).
pub fn find_model_files(
    model_dir: &Path,
    chunk_hint: Option<&str>,
) -> Result<(String, String, String, String)> {
    let entries: Vec<String> = fs::read_dir(model_dir)
        .context("Cannot read model directory")?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();

    let matches_chunk = |f: &str| -> bool {
        match chunk_hint {
            Some(hint) => f.contains(hint),
            None => true,
        }
    };

    let find = |prefix: &str| -> Result<String> {
        // Prefer int8 + matching chunk
        if let Some(f) = entries.iter().find(|f| {
            f.starts_with(prefix)
                && f.contains("int8")
                && f.ends_with(".onnx")
                && matches_chunk(f)
        }) {
            return Ok(model_dir.join(f).to_string_lossy().into_owned());
        }
        // Fall back to fp32 + matching chunk
        if let Some(f) = entries.iter().find(|f| {
            f.starts_with(prefix) && f.ends_with(".onnx") && matches_chunk(f)
        }) {
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
/// Input: comma-separated keywords. Output: newline-delimited tokenized lines.
///
/// The tokenization strategy depends on which model files are present:
///  - If `bpe.model` exists → BPE tokenization (Gigaspeech model).
///  - If `en.phone` exists  → Phone tokenization (ZhEn model).
pub fn tokenize_keywords(keywords: &str, model_dir: &Path) -> Result<String> {
    let bpe_path = model_dir.join("bpe.model");
    let phone_path = model_dir.join("en.phone");

    enum Tok {
        Bpe(BpeTokenizer),
        Phone(PhoneTokenizer),
    }

    let tok = if bpe_path.exists() {
        Tok::Bpe(BpeTokenizer::load(&bpe_path)?)
    } else if phone_path.exists() {
        Tok::Phone(PhoneTokenizer::load(&phone_path)?)
    } else {
        bail!(
            "Neither bpe.model nor en.phone found in {} — cannot tokenize plain-text keywords",
            model_dir.display()
        );
    };

    let mut lines = Vec::new();
    for kw in keywords.split(',') {
        let kw = kw.trim();
        if kw.is_empty() {
            continue;
        }
        let tokens = match &tok {
            Tok::Bpe(t) => t.encode(kw),
            Tok::Phone(t) => t.encode(kw)?,
        };
        let label = kw.replace(' ', "_").to_uppercase();
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
pub fn ensure_model(model_dir: &Path, model: KwsModel) -> Result<()> {
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

    let url = model.url();
    eprintln!("Downloading KWS model ({model})...");
    let response = reqwest::blocking::get(url).context("Failed to download model")?;
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

/// Default model directory under XDG data dir for the given model.
pub fn default_model_dir_for(model: KwsModel) -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("telemuze")
        .join("models")
        .join(model.dir_name())
}

/// Default model directory under XDG data dir (uses [`DEFAULT_MODEL`]).
pub fn default_model_dir() -> PathBuf {
    default_model_dir_for(DEFAULT_MODEL)
}

// ── KeywordSpotter initialization ───────────────────────────────────────────

/// Configuration for initializing the keyword spotter.
pub struct KwsConfig {
    pub model: KwsModel,
    pub model_dir: PathBuf,
    pub keywords: String,
    pub keywords_score: f32,
    pub keywords_threshold: f32,
    pub num_threads: i32,
}

/// Initialize the KeywordSpotter and create a stream.
/// The `keywords` field should be comma-separated plain-text keywords.
pub fn init_keyword_spotter(cfg: &KwsConfig) -> Result<KeywordSpotter> {
    ensure_model(&cfg.model_dir, cfg.model)?;

    let (encoder, decoder, joiner, tokens) =
        find_model_files(&cfg.model_dir, cfg.model.chunk_hint())?;
    debug!("KWS encoder: {encoder}");
    debug!("KWS decoder: {decoder}");
    debug!("KWS joiner:  {joiner}");
    debug!("KWS tokens:  {tokens}");

    let keywords_buf = tokenize_keywords(&cfg.keywords, &cfg.model_dir)?;

    let mut config = KeywordSpotterConfig::default();
    config.model_config.transducer.encoder = Some(encoder);
    config.model_config.transducer.decoder = Some(decoder);
    config.model_config.transducer.joiner = Some(joiner);
    config.model_config.tokens = Some(tokens);
    config.model_config.num_threads = cfg.num_threads;
    config.model_config.provider = Some("cpu".into());
    config.keywords_score = cfg.keywords_score;
    config.keywords_threshold = cfg.keywords_threshold;
    config.keywords_buf = Some(keywords_buf);

    KeywordSpotter::create(&config)
        .context("Failed to create KeywordSpotter — check model paths and keywords format")
}
