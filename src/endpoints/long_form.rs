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
use crate::engines::diarization::assign_speakers;
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
    tokens: Vec<String>,
    /// Per-token timestamps in seconds, absolute (offset by segment start).
    token_timestamps: Vec<f64>,
    /// 0-based speaker index from diarization. None when diarization is disabled
    /// or no diarization segment overlaps this ASR segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker: Option<i32>,
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/v1/transcribe/long", post(handle_long_form))
}

async fn handle_long_form(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut file_data: Option<Vec<u8>> = None;
    let mut raw_hotwords: Option<String> = None;
    let mut hotwords_score: Option<f32> = None;
    let mut num_speakers: Option<i32> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();

        match name.as_str() {
            "file" => match field.bytes().await {
                Ok(bytes) => file_data = Some(bytes.to_vec()),
                Err(e) => {
                    error!("Failed to read file field: {e}");
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error": "Failed to read uploaded file"})),
                    )
                        .into_response();
                }
            },
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
            "num_speakers" => {
                if let Ok(text) = field.text().await {
                    num_speakers = text.parse::<i32>().ok().filter(|&n| n > 0);
                }
            }
            _ => {}
        }
    }

    let hotwords = raw_hotwords.map(|hw| crate::hotwords::parse_hotwords(&hw, hotwords_score));

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

    if hotwords.is_some() {
        info!("Hotwords: {:?}", hotwords.as_deref().unwrap());
    }

    // VAD segmentation + per-segment STT
    let segments = match state.vad_transcribe_with_hotwords(&pcm, hotwords.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            error!("VAD transcription failed: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("VAD transcription failed: {e}")})),
            )
                .into_response();
        }
    };

    // Run diarization if the engine is loaded, then assign speakers to segments.
    let speaker_labels: Vec<Option<i32>> = if let Some(diar_mutex) = &state.diarization_engine {
        match diar_mutex.lock().unwrap().diarize(&pcm, num_speakers) {
            Ok(diar_segs) => {
                let starts: Vec<f64> = segments.iter().map(|s| s.start_secs).collect();
                let ends: Vec<f64> = segments.iter().map(|s| s.end_secs).collect();
                assign_speakers(&starts, &ends, &diar_segs)
            }
            Err(e) => {
                error!("Diarization failed: {e}");
                vec![None; segments.len()]
            }
        }
    } else {
        vec![None; segments.len()]
    };

    let result_segments: Vec<SegmentResponse> = segments
        .iter()
        .zip(speaker_labels.iter())
        .map(|(s, &speaker)| SegmentResponse {
            start: s.start_secs,
            end: s.end_secs,
            text: s.text.clone(),
            tokens: s.tokens.clone(),
            token_timestamps: s
                .token_timestamps
                .iter()
                .map(|&t| s.start_secs + t as f64)
                .collect(),
            speaker,
        })
        .collect();

    let full_text: String = segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
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
