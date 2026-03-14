mod audio;
mod config;
mod endpoints;
mod engines;
mod models;
mod state;
mod telegram;

use anyhow::Result;
use axum::Router;
use clap::Parser;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::config::Config;
use crate::state::AppState;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "telemuze=info,tower_http=info".into()),
        )
        .init();

    let config = Config::parse();
    info!("Telemuze starting up...");
    info!("Loading models into memory...");

    let state = AppState::new(&config).await?;
    let shared_state = Arc::new(state);

    let has_token = !config.telegram_bot_token.is_empty();
    let has_api_id = config.telegram_api_id != 0;
    let has_api_hash = !config.telegram_api_hash.is_empty();

    if has_token && has_api_id && has_api_hash {
        info!("Starting Telegram bot...");
        let bot_state = shared_state.clone();
        let api_id = config.telegram_api_id;
        let api_hash = config.telegram_api_hash.clone();
        let token = config.telegram_bot_token.clone();
        tokio::spawn(async move {
            if let Err(e) = telegram::run_bot(api_id, api_hash, token, bot_state).await {
                tracing::error!("Telegram bot error: {e}");
            }
        });
    } else if has_token || has_api_id || has_api_hash {
        let mut missing = Vec::new();
        if !has_token { missing.push("TELEGRAM_BOT_TOKEN"); }
        if !has_api_id { missing.push("TELEGRAM_API_ID"); }
        if !has_api_hash { missing.push("TELEGRAM_API_HASH"); }
        tracing::warn!(
            "Telegram bot not started: missing {}. All three are required.",
            missing.join(", ")
        );
    } else {
        info!("Telegram bot disabled (set TELEGRAM_BOT_TOKEN, TELEGRAM_API_ID, and TELEGRAM_API_HASH to enable)");
    }

    let app = Router::new()
        .merge(endpoints::transcriptions::router())
        .merge(endpoints::dictate::router())
        .merge(endpoints::long_form::router())
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(shared_state);

    let bind_addr = format!("{}:{}", config.host, config.port);
    info!("Listening on {bind_addr}");

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
