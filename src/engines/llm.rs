//! LLM engine for smart dictation post-processing.
//!
//! Supports two backends:
//! - **Native**: runs a GGUF model in-process via llama.cpp (default)
//! - **HTTP**: calls an external OpenAI-compatible chat completions API
//!
//! When neither backend is available, the engine is disabled and
//! `correct_dictation` returns the raw STT text unchanged.

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;
use tracing::{debug, info, warn};

/// Max output tokens — dictation correction output should be roughly the
/// same length as input, so a small multiple suffices. This caps generation
/// time on CPU.
const MAX_OUTPUT_TOKENS: i32 = 512;

/// Context size for native inference. Covers prompt + output.
const CONTEXT_SIZE: u32 = 2048;

/// LLM engine with pluggable backends.
pub struct LlmEngine {
    inner: LlmInner,
}

/// Holds the llama.cpp model and a reusable context.
///
/// The context is created once and reused across requests by clearing the
/// KV cache, avoiding the ~500 MB allocation/deallocation per request.
///
/// Safety: `LlamaContext` borrows `LlamaModel` with a lifetime parameter,
/// but both are stored together and the context is always dropped before
/// the model (declaration order). The lifetime is transmuted to `'static`
/// to allow storage in the struct.
struct NativeState {
    #[allow(dead_code)]
    backend: LlamaBackend,
    model: LlamaModel,
    ctx: Mutex<LlamaContext<'static>>,
    temperature: f32,
}

// Safety: LlamaModel is read-only after loading (thread-safe for tokenization,
// chat templates). LlamaContext is behind a Mutex, ensuring single-threaded access.
unsafe impl Send for NativeState {}
unsafe impl Sync for NativeState {}

enum LlmInner {
    Native(Box<NativeState>),
    Http {
        client: reqwest::Client,
        api_url: String,
    },
    #[allow(dead_code)]
    Disabled,
}

// ── HTTP-backend request/response types ─────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

// ── Construction ────────────────────────────────────────────────────────────

impl LlmEngine {
    /// Create an LLM engine backed by a local GGUF model via llama.cpp.
    pub fn new_native(gguf_path: &Path, temperature: f32) -> Result<Self> {
        info!("Loading native LLM from {}...", gguf_path.display());

        // Suppress llama.cpp's verbose internal logging
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

        let mut backend =
            LlamaBackend::init().context("Failed to initialize llama.cpp backend")?;
        backend.void_logs();

        let model_params = LlamaModelParams::default();
        let model_params = std::pin::pin!(model_params);

        let model = LlamaModel::load_from_file(&backend, gguf_path, &model_params)
            .map_err(|e| anyhow::anyhow!("Failed to load GGUF model: {e:?}"))?;

        // Create a persistent context — reused across requests via KV cache clearing
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_SIZE));

        let ctx = model
            .new_context(&backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create inference context: {e:?}"))?;

        // Safety: model outlives ctx (both stored in NativeState, ctx dropped first via Mutex)
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };

        info!("Native LLM engine ready.");

        Ok(Self {
            inner: LlmInner::Native(Box::new(NativeState {
                backend,
                model,
                ctx: Mutex::new(ctx),
                temperature,
            })),
        })
    }

    /// Create an LLM engine that calls an external OpenAI-compatible API.
    pub fn new_http(api_url: &str) -> Self {
        info!("LLM engine configured at {api_url}");
        Self {
            inner: LlmInner::Http {
                client: reqwest::Client::new(),
                api_url: api_url.to_string(),
            },
        }
    }

    /// Create a disabled LLM engine (returns raw STT text).
    #[allow(dead_code)]
    pub fn disabled() -> Self {
        warn!("LLM engine disabled — smart dictation will return raw STT output");
        Self {
            inner: LlmInner::Disabled,
        }
    }

    /// Correct raw STT output using the LLM.
    ///
    /// `terms_content` is the raw contents of the terms file, passed
    /// verbatim into the system prompt for the LLM to interpret.
    pub async fn correct_dictation(
        &self,
        raw_text: &str,
        terms_content: &str,
    ) -> Result<String> {
        if raw_text.trim().is_empty() {
            return Ok(String::new());
        }

        match &self.inner {
            LlmInner::Native(state) => {
                Self::correct_native(state, raw_text, terms_content)
            }
            LlmInner::Http { client, api_url } => {
                Self::correct_http(client, api_url, raw_text, terms_content)
                    .await
            }
            LlmInner::Disabled => {
                debug!("LLM disabled — returning raw STT text");
                Ok(raw_text.to_string())
            }
        }
    }

    // ── Native backend ──────────────────────────────────────────────────

    fn correct_native(
        state: &NativeState,
        raw_text: &str,
        terms_content: &str,
    ) -> Result<String> {
        let system_prompt = build_system_prompt(terms_content);

        // Build the prompt manually using ChatML format with a pre-closed
        // <think> block to suppress Qwen3.5's thinking mode. The model's
        // template does this when enable_thinking=false — an empty
        // <think></think> signals the model to skip reasoning.
        let prompt = format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n\
             <|im_start|>user\n{raw_text}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );

        debug!("LLM prompt: {prompt}");

        // Tokenize
        let tokens = state
            .model
            .str_to_token(&prompt, AddBos::Never)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e:?}"))?;

        // Lock the persistent context and clear KV cache for this request
        let mut ctx = state.ctx.lock().unwrap();
        ctx.clear_kv_cache();

        // Encode prompt into batch
        let mut batch = LlamaBatch::new(512, 1);
        let last_idx = (tokens.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens.iter()) {
            batch
                .add(*token, i, &[0], i == last_idx)
                .context("Failed to add token to batch")?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Failed to decode prompt: {e:?}"))?;

        // Sample output tokens using Qwen3.5 recommended non-thinking params:
        // top_k=20, presence_penalty=2.0, temperature from config
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(
                256,  // penalty_last_n: look back window
                1.0,  // penalty_repeat
                0.0,  // penalty_freq
                2.0,  // penalty_present
            ),
            LlamaSampler::top_k(20),
            LlamaSampler::temp(state.temperature),
            LlamaSampler::dist(1234),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();

        while n_cur < tokens.len() as i32 + MAX_OUTPUT_TOKENS {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if state.model.is_eog_token(token) {
                break;
            }

            let piece = state
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| anyhow::anyhow!("Failed to decode token: {e:?}"))?;
            output.push_str(&piece);

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .context("Failed to add token to batch")?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("Failed to decode token: {e:?}"))?;
        }

        let result = output.trim().to_string();
        debug!("LLM correction: '{}' -> '{}'", raw_text, result);
        Ok(result)
    }

    // ── HTTP backend ────────────────────────────────────────────────────

    async fn correct_http(
        client: &reqwest::Client,
        api_url: &str,
        raw_text: &str,
        terms_content: &str,
    ) -> Result<String> {
        let system_prompt = build_system_prompt(terms_content);

        let request = ChatRequest {
            model: "local".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: raw_text.to_string(),
                },
            ],
            temperature: 0.1,
            max_tokens: 2048,
        };

        let response = client
            .post(api_url)
            .json(&request)
            .send()
            .await
            .context("Failed to reach LLM API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM API returned {status}: {body}");
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .context("Failed to parse LLM API response")?;

        let text = chat_response
            .choices
            .first()
            .map(|c| c.message.content.trim().to_string())
            .unwrap_or_else(|| raw_text.to_string());

        debug!("LLM correction: '{}' -> '{}'", raw_text, text);
        Ok(text)
    }
}

/// Build the system prompt for dictation correction.
///
/// The terms file is a simple list of correct terms (one per line).
/// The model is expected to recognize when STT output contains words
/// that sound like these terms and substitute the correct spelling.
fn build_system_prompt(terms_content: &str) -> String {
    let terms_section = if terms_content.is_empty() {
        String::from("No custom terms configured.")
    } else {
        let terms: Vec<&str> = terms_content
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        format!("Custom terms:\n{}", terms.join("\n"))
    };

    format!(
        "You are a speech-to-text post-processor. The input may contain words or phrases \
        that were misrecognized by speech recognition. When a word or phrase sounds like \
        one of the custom terms below, replace it with the correct term. \
        Do NOT change anything else. Output ONLY the corrected text.\n\
        \n\
        {terms_section}"
    )
}

