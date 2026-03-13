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

    if !config.telegram_bot_token.is_empty() {
        let bot_state = shared_state.clone();
        let api_id = config.telegram_api_id;
        let api_hash = config.telegram_api_hash.clone();
        let token = config.telegram_bot_token.clone();
        tokio::spawn(async move {
            if let Err(e) = telegram::run_bot(api_id, api_hash, token, bot_state).await {
                tracing::error!("Telegram bot error: {e}");
            }
        });
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
