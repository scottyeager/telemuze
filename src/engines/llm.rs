//! LLM engine for smart dictation post-processing.
//!
//! Calls a local OpenAI-compatible chat completions API (e.g., mistral.rs
//! server, llama.cpp server, or any compatible endpoint) to correct
//! raw STT output with context-aware grammar fixing and custom term injection.
//!
//! This HTTP-based approach avoids dependency conflicts between the STT
//! and LLM Rust crates, and allows swapping in any LLM backend.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Wraps an HTTP client for calling a local LLM API.
pub struct LlmEngine {
    client: reqwest::Client,
    api_url: String,
    enabled: bool,
}

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

impl LlmEngine {
    /// Create an LLM engine that calls a local OpenAI-compatible API.
    ///
    /// The `llm_api_url` should point to a chat completions endpoint, e.g.:
    /// `http://127.0.0.1:8081/v1/chat/completions`
    ///
    /// If the URL is empty or the endpoint is unreachable, the engine
    /// degrades gracefully by returning raw STT text.
    pub async fn new(llm_api_url: &str) -> Result<Self> {
        let client = reqwest::Client::new();
        let enabled = !llm_api_url.is_empty();

        if enabled {
            info!("LLM engine configured at {llm_api_url}");
        } else {
            warn!("No LLM API URL configured — smart dictation will return raw STT output");
        }

        Ok(Self {
            client,
            api_url: llm_api_url.to_string(),
            enabled,
        })
    }

    /// Correct raw STT output using the LLM.
    ///
    /// Sends a chat completion request with a strict system prompt to fix
    /// grammar, punctuation, and inject known custom terms while preserving
    /// the speaker's exact meaning. Uses temperature 0.1 for deterministic output.
    pub async fn correct_dictation(
        &self,
        raw_text: &str,
        custom_terms: &[String],
    ) -> Result<String> {
        if !self.enabled {
            debug!("LLM disabled — returning raw STT text");
            return Ok(raw_text.to_string());
        }

        let terms_str = if custom_terms.is_empty() {
            String::from("(none)")
        } else {
            custom_terms.join(", ")
        };

        let system_prompt = format!(
            "You are a dictation post-processor. Your ONLY job is to clean up speech-to-text output.\n\
            \n\
            Rules:\n\
            1. Fix grammar, punctuation, and capitalization.\n\
            2. If a word sounds like one of these custom terms, replace it: [{terms_str}]\n\
            3. Do NOT add, remove, or rephrase content. Preserve the speaker's exact meaning.\n\
            4. Do NOT add any commentary, explanation, or markdown.\n\
            5. Output ONLY the corrected text, nothing else."
        );

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

        let response = self
            .client
            .post(&self.api_url)
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
