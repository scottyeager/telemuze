//! Endpoint B: Smart dictation endpoint.
//!
//! `POST /v1/dictate/smart`
//!
//! The primary endpoint for desktop dictation. Audio is transcribed via STT
//! then passed through the local LLM for context-aware grammar correction
//! and custom term injection. Returns plain text (no JSON) for zero-overhead
//! client-side processing.

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use std::sync::Arc;
use tracing::{error, info};

use crate::audio;
use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/v1/dictate/smart", post(handle_smart_dictation))
}

async fn handle_smart_dictation(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut file_data: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            match field.bytes().await {
                Ok(bytes) => file_data = Some(bytes.to_vec()),
                Err(e) => {
                    error!("Failed to read file field: {e}");
                    return (StatusCode::BAD_REQUEST, "Failed to read uploaded file")
                        .into_response();
                }
            }
        }
    }

    let file_data = match file_data {
        Some(d) => d,
        None => {
            return (StatusCode::BAD_REQUEST, "Missing 'file' field").into_response();
        }
    };

    info!("Smart dictation request: {} bytes", file_data.len());

    // Step 1: Decode audio to PCM
    let pcm = match audio::decode_to_pcm(&file_data) {
        Ok(p) => p,
        Err(e) => {
            error!("Audio decode failed: {e}");
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("Audio decode failed: {e}"),
            )
                .into_response();
        }
    };

    // Step 2: STT transcription
    let raw_text = match state.stt_engine.lock().unwrap().transcribe(&pcm) {
        Ok(text) => text,
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

    // Step 3: LLM correction with custom dictionary
    let corrected = match state
        .llm_engine
        .correct_dictation(&raw_text, &state.terms_content)
        .await
    {
        Ok(text) => text,
        Err(e) => {
            error!("LLM correction failed, returning raw STT: {e}");
            // Graceful degradation: return raw STT if LLM fails
            raw_text
        }
    };

    info!("Corrected: '{}'", corrected);

    // Return plain text — no JSON wrapper needed
    Response::builder()
        .header("content-type", "text/plain; charset=utf-8")
        .body(corrected)
        .unwrap()
        .into_response()
}
