//! Smart dictation endpoints.
//!
//! `POST /v1/dictate/smart`  — Audio → STT → LLM correction → plain text
//! `POST /v1/dictate/correct` — Text → LLM correction → plain text (no audio)

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use std::sync::Arc;
use tracing::{debug, error, info};

use crate::audio;
use crate::engines::dictionary;
use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/dictate/smart", post(handle_smart_dictation))
        .route("/v1/dictate/correct", post(handle_correct))
}

async fn handle_smart_dictation(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut file_data: Option<Vec<u8>> = None;
    let mut file_is_raw_pcm = false;
    let mut raw_hotwords: Option<String> = None;
    let mut hotwords_score: Option<f32> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                file_is_raw_pcm = field.content_type() == Some("audio/pcm");
                match field.bytes().await {
                    Ok(bytes) => file_data = Some(bytes.to_vec()),
                    Err(e) => {
                        error!("Failed to read file field: {e}");
                        return (StatusCode::BAD_REQUEST, "Failed to read uploaded file")
                            .into_response();
                    }
                }
            }
            "hotwords" => {
                if let Ok(text) = field.text().await {
                    raw_hotwords = Some(text);
                }
            }
            "hotwords_score" => {
                if let Ok(text) = field.text().await {
                    hotwords_score = text.parse().ok();
                }
            }
            _ => {}
        }
    }

    let hotwords = raw_hotwords.map(|hw| crate::hotwords::parse_hotwords(&hw, hotwords_score));

    let file_data = match file_data {
        Some(d) => d,
        None => {
            return (StatusCode::BAD_REQUEST, "Missing 'file' field").into_response();
        }
    };

    info!("Smart dictation request: {} bytes ({})", file_data.len(), if file_is_raw_pcm { "raw pcm" } else { "file" });

    // Step 1: Decode audio to PCM
    let pcm = if file_is_raw_pcm {
        match audio::decode_raw_f32le(&file_data) {
            Ok(p) => p,
            Err(e) => {
                error!("PCM decode failed: {e}");
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("PCM decode failed: {e}"),
                )
                    .into_response();
            }
        }
    } else {
        match audio::decode_to_pcm(&file_data) {
            Ok(p) => p,
            Err(e) => {
                error!("Audio decode failed: {e}");
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("Audio decode failed: {e}"),
                )
                    .into_response();
            }
        }
    };

    if hotwords.is_some() {
        info!("Hotwords: {:?}", hotwords.as_deref().unwrap());
    }

    // Step 2: STT transcription
    let raw_text = match state.stt_engine.lock().unwrap().transcribe_with_hotwords(&pcm, hotwords.as_deref()) {
        Ok(result) => result.text,
        Err(e) => {
            error!("STT failed: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Transcription failed: {e}"),
            )
                .into_response();
        }
    };

    info!("Raw STT: '{}'", raw_text);

    // Step 3: Phonetic/fuzzy dictionary pipeline
    let candidates = state.dictionary.find_candidates(&raw_text, &state.pipeline_config);
    let candidate_hints = if candidates.is_empty() {
        None
    } else {
        Some(dictionary::format_candidates_for_llm(&candidates))
    };

    // Step 4: LLM correction (with candidate hints if available)
    let corrected = match state.llm_engine.as_ref() {
        None => {
            debug!("LLM correction disabled, returning raw STT");
            raw_text
        }
        Some(engine) => match engine
            .correct_dictation(&raw_text, &state.terms_content, candidate_hints.as_deref())
            .await
        {
            Ok(text) => text,
            Err(e) => {
                error!("LLM correction failed, returning raw STT: {e}");
                raw_text
            }
        },
    };

    info!("Corrected: '{}'", corrected);

    // Return plain text — no JSON wrapper needed
    Response::builder()
        .header("content-type", "text/plain; charset=utf-8")
        .body(corrected)
        .unwrap()
        .into_response()
}

/// Text-only LLM correction — skips audio/STT, useful for testing.
async fn handle_correct(
    State(state): State<Arc<AppState>>,
    body: String,
) -> impl IntoResponse {
    let raw_text = body.trim();
    if raw_text.is_empty() {
        return (StatusCode::BAD_REQUEST, "Empty input").into_response();
    }

    debug!("Correct request: '{}'", raw_text);

    // Run phonetic/fuzzy pipeline
    let candidates = state.dictionary.find_candidates(raw_text, &state.pipeline_config);
    let candidate_hints = if candidates.is_empty() {
        None
    } else {
        Some(dictionary::format_candidates_for_llm(&candidates))
    };

    let corrected = match state.llm_engine.as_ref() {
        None => {
            debug!("LLM correction disabled, returning input text");
            raw_text.to_string()
        }
        Some(engine) => match engine
            .correct_dictation(raw_text, &state.terms_content, candidate_hints.as_deref())
            .await
        {
            Ok(text) => text,
            Err(e) => {
                error!("LLM correction failed: {e}");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("LLM correction failed: {e}"),
                )
                    .into_response();
            }
        },
    };

    info!("Correct: '{}' -> '{}'", raw_text, corrected);

    Response::builder()
        .header("content-type", "text/plain; charset=utf-8")
        .body(corrected)
        .unwrap()
        .into_response()
}
