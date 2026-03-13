//! Endpoint C: Long-form media transcription.
//!
//! `POST /v1/transcribe/long`
//!
//! Handles hour-long podcasts, Zoom meetings, and video files.
//! Uses VAD to chunk audio into speech segments, transcribes each
//! segment independently, and returns timestamped results.
//! Bypasses the LLM for speed and exact timing accuracy.

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
struct LongFormResponse {
    full_text: String,
    segments: Vec<SegmentResponse>,
}

#[derive(Serialize)]
struct SegmentResponse {
    start: f64,
    end: f64,
    text: String,
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/v1/transcribe/long", post(handle_long_form))
}

async fn handle_long_form(
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
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error": "Failed to read uploaded file"})),
                    )
                        .into_response();
                }
            }
        }
    }

    let file_data = match file_data {
        Some(d) => d,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Missing 'file' field"})),
            )
                .into_response();
        }
    };

    info!("Long-form transcription request: {} bytes", file_data.len());

    // Step 1: Decode to PCM
    let pcm = match audio::decode_to_pcm(&file_data) {
        Ok(p) => p,
        Err(e) => {
            error!("Audio decode failed: {e}");
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({"error": format!("Audio decode failed: {e}")})),
            )
                .into_response();
        }
    };

    let duration_secs = pcm.len() as f64 / 16_000.0;
    info!("Audio duration: {:.1}s", duration_secs);

    // Step 2: VAD segmentation
    let speech_segments = match state.vad_engine.lock().unwrap().segment_audio(&pcm) {
        Ok(s) => s,
        Err(e) => {
            error!("VAD segmentation failed: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("VAD segmentation failed: {e}")})),
            )
                .into_response();
        }
    };

    info!("VAD found {} speech segments", speech_segments.len());

    // Step 3: Transcribe each segment
    let mut result_segments = Vec::with_capacity(speech_segments.len());
    let mut full_text_parts = Vec::with_capacity(speech_segments.len());

    for (i, seg) in speech_segments.iter().enumerate() {
        match state.stt_engine.lock().unwrap().transcribe(&seg.samples) {
            Ok(text) => {
                if !text.trim().is_empty() {
                    info!(
                        "Segment {}/{}: [{:.1}s - {:.1}s] '{}'",
                        i + 1,
                        speech_segments.len(),
                        seg.start_secs,
                        seg.end_secs,
                        text
                    );
                    full_text_parts.push(text.clone());
                    result_segments.push(SegmentResponse {
                        start: seg.start_secs,
                        end: seg.end_secs,
                        text,
                    });
                }
            }
            Err(e) => {
                error!(
                    "STT failed for segment {} [{:.1}s - {:.1}s]: {e}",
                    i + 1,
                    seg.start_secs,
                    seg.end_secs
                );
                // Continue with other segments rather than failing entirely
            }
        }
    }

    let full_text = full_text_parts.join(" ");
    info!(
        "Long-form transcription complete: {} segments, {} chars",
        result_segments.len(),
        full_text.len()
    );

    Json(LongFormResponse {
        full_text,
        segments: result_segments,
    })
    .into_response()
}
