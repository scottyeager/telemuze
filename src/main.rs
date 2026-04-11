mod audio;
mod config;
mod endpoints;
mod engines;
pub mod hotwords;
mod long_form;
mod models;
mod state;
mod telegram;
mod worker;

use anyhow::Result;
use axum::extract::DefaultBodyLimit;
use axum::Router;
use clap::Parser;
use std::net::{Ipv6Addr, SocketAddr, SocketAddrV6};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::config::Config;
use crate::state::AppState;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("transcribe") {
        // Worker mode: minimal logging, one-shot. Stdout is reserved
        // for the JSON output the parent process parses, so tracing
        // must write to stderr.
        tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "telemuze=info".into()),
            )
            .init();
        return worker::run(&args[2..]);
    }

    server_main()
}

#[tokio::main]
async fn server_main() -> Result<()> {
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
        tokio::spawn(telegram::run_bot(api_id, api_hash, token, bot_state));
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
        .merge(endpoints::web_ui::router())
        .merge(endpoints::transcriptions::router())
        .merge(endpoints::dictate::router())
        .merge(endpoints::long_form::router())
        .layer(DefaultBodyLimit::max(500 * 1024 * 1024)) // 500 MB
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(shared_state);

    match config.host.as_deref() {
        Some(host) => {
            let bind_addr = format!("{}:{}", host, config.port);
            info!("Listening on {bind_addr}");
            let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
            axum::serve(listener, app).await?;
        }
        None => {
            let v4_addr: SocketAddr = ([0, 0, 0, 0], config.port).into();
            let v6_addr = SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, config.port, 0, 0);
            info!("Listening on {v4_addr} and [{}]:{}", v6_addr.ip(), v6_addr.port());

            let v4_listener = tokio::net::TcpListener::bind(v4_addr).await?;
            let v6_listener = bind_v6_only(v6_addr)?;

            let app_v6 = app.clone();
            tokio::try_join!(
                async move { axum::serve(v4_listener, app).await },
                async move { axum::serve(v6_listener, app_v6).await },
            )?;
        }
    }

    Ok(())
}

/// Bind an IPv6-only TCP listener. Setting `IPV6_V6ONLY` keeps this
/// socket from claiming the IPv4 address space, so a separate IPv4
/// listener on the same port can coexist regardless of the kernel's
/// `net.ipv6.bindv6only` sysctl.
fn bind_v6_only(addr: SocketAddrV6) -> Result<tokio::net::TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};

    let socket = Socket::new(Domain::IPV6, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_only_v6(true)?;
    socket.set_reuse_address(true)?;
    socket.set_nonblocking(true)?;
    socket.bind(&SocketAddr::V6(addr).into())?;
    socket.listen(1024)?;
    Ok(tokio::net::TcpListener::from_std(socket.into())?)
}
