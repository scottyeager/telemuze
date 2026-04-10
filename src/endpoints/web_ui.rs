//! Simple web UI for long-form transcription.
//!
//! Serves a self-contained HTML page at `GET /` with drag-and-drop
//! file upload, progress indication, and transcript display.

use axum::response::Html;
use axum::routing::get;
use axum::Router;
use std::sync::Arc;

use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/", get(serve_ui))
}

async fn serve_ui() -> Html<&'static str> {
    Html(include_str!("web_ui.html"))
}
