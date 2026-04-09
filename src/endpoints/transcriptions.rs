//! Endpoint A: OpenAI-compatible transcription endpoint.
//!
//! `POST /v1/audio/transcriptions`
//!
//! Drop-in replacement for OpenAI's Whisper API. Accepts multipart/form-data
//! with `file` and optional `model` fields. Bypasses the LLM entirely for
//! maximum compatibility with existing clients (e.g., Whispering GUI).

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::post;
use axum::Router;
use serde::Serialize;
use std::sync::Arc;
use tracing::{error, info};

use crate::audio;
use crate::state::AppState;

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/v1/audio/transcriptions", post(handle_transcription))
}

async fn handle_transcription(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut file_data: Option<Vec<u8>> = None;
    let mut file_is_raw_pcm = false;
    let mut raw_hotwords: Option<String> = None;
    let mut hotwords_score: Option<f32> = None;

    // Extract the file from multipart form data
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => {
                file_is_raw_pcm = field.content_type() == Some("audio/pcm");
                match field.bytes().await {
                    Ok(bytes) => file_data = Some(bytes.to_vec()),
                    Err(e) => {
                        error!("Failed to read file field: {e}");
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({"error": "Failed to read uploaded file"})),
                        )
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
            _ => {} // `model` and other fields accepted but ignored
        }
    }

    let hotwords = raw_hotwords.map(|hw| crate::hotwords::parse_hotwords(&hw, hotwords_score));

    let file_data = match file_data {
        Some(d) => d,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing 'file' field in multipart form"})),
            )
                .into_response();
        }
    };

    info!("Transcription request: {} bytes ({})", file_data.len(), if file_is_raw_pcm { "raw pcm" } else { "file" });

    let pcm = if file_is_raw_pcm {
        match audio::decode_raw_f32le(&file_data) {
            Ok(p) => p,
            Err(e) => {
                error!("PCM decode failed: {e}");
                return (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(serde_json::json!({"error": format!("PCM decode failed: {e}")})),
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
                    Json(serde_json::json!({"error": format!("Audio decode failed: {e}")})),
                )
                    .into_response();
            }
        }
    };

    if hotwords.is_some() {
        info!("Hotwords: {:?}", hotwords.as_deref().unwrap());
    }

    // Run STT (no LLM correction for OpenAI compatibility)
    match state.stt_engine.lock().unwrap().transcribe_with_hotwords(&pcm, hotwords.as_deref()) {
        Ok(result) => {
            info!("Transcription complete: {} chars", result.text.len());
            Json(TranscriptionResponse { text: result.text }).into_response()
        }
        Err(e) => {
            error!("STT failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Transcription failed: {e}")})),
            )
                .into_response()
        }
    }
}
