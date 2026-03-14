//! Telegram bot interface for Telemuze.
//!
//! Connects via MTProto (grammers) so it can handle large file downloads
//! (up to 2 GB) without needing a self-hosted Bot API server.
//! Voice notes go through the smart dictation pipeline (STT + LLM),
//! while audio/video file attachments use the long-form pipeline (VAD + STT).

use std::sync::Arc;

use anyhow::Result;
use grammers_client::client::{Client, UpdatesConfiguration};
use grammers_client::media::Media;
use grammers_client::update::Update;
use grammers_client::SenderPool;
use grammers_session::storages::MemorySession;
use tracing::{error, info};

use crate::audio;
use crate::state::AppState;

/// Maximum length for a single Telegram message.
const TELEGRAM_MAX_LEN: usize = 4096;

pub async fn run_bot(
    api_id: i32,
    api_hash: String,
    token: String,
    state: Arc<AppState>,
) -> Result<()> {
    info!("Connecting Telegram bot...");

    let session = Arc::new(MemorySession::default());
    let pool = SenderPool::new(Arc::clone(&session), api_id);

    let client = Client::new(pool.handle);

    // The runner drives all network I/O; it must be spawned.
    tokio::spawn(pool.runner.run());

    client.bot_sign_in(&token, &api_hash).await?;
    info!("Telegram bot connected");

    let mut update_stream = client
        .stream_updates(pool.updates, UpdatesConfiguration::default())
        .await;

    loop {
        let update = update_stream.next().await?;
        match update {
            Update::NewMessage(message) if !message.outgoing() => {
                let client = client.clone();
                let state = state.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_message(&client, &message, &state).await {
                        error!("Error handling Telegram message: {e}");
                        let _ = message.reply(format!("Error: {e}")).await;
                    }
                });
            }
            _ => {}
        }
    }
}

async fn handle_message(
    client: &Client,
    message: &grammers_client::update::Message,
    state: &AppState,
) -> Result<()> {
    // Check if the sender is in the allowed users list.
    if !state.telegram_allowed_users.is_empty() {
        let allowed = message
            .sender()
            .and_then(|peer| peer.username())
            .is_some_and(|username| {
                state
                    .telegram_allowed_users
                    .contains(&username.to_lowercase())
            });

        if !allowed {
            info!(
                "Telegram: rejecting message from unauthorized user {:?}",
                message.sender_id()
            );
            return Ok(());
        }
    }

    let media = match message.media() {
        Some(m) => m,
        None => {
            message
                .reply("Send me a voice note for smart dictation, or an audio/video file for long-form transcription.")
                .await?;
            return Ok(());
        }
    };

    match &media {
        Media::Document(doc) => {
            let mime = doc.mime_type().unwrap_or("");
            let is_voice = mime.starts_with("audio/ogg");
            let is_audio_video = mime.starts_with("audio/") || mime.starts_with("video/");

            if is_voice {
                info!(
                    "Telegram: voice note from {:?}",
                    message.sender_id()
                );
                let bytes = download_media(client, &media).await?;
                transcribe_and_reply(message, &bytes, state, "voice note").await?;
            } else if is_audio_video {
                info!(
                    "Telegram: audio/video file ({mime}) from {:?}",
                    message.sender_id()
                );
                let bytes = download_media(client, &media).await?;
                transcribe_and_reply(message, &bytes, state, "long-form").await?;
            } else {
                message
                    .reply(format!("Unsupported file type: {mime}"))
                    .await?;
            }
        }
        _ => {
            message
                .reply("Send me a voice note or audio/video file to transcribe.")
                .await?;
        }
    }

    Ok(())
}

async fn download_media(client: &Client, media: &Media) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut download = client.iter_download(media);
    while let Some(chunk) = download.next().await? {
        bytes.extend(chunk);
    }
    info!("Downloaded {} bytes from Telegram", bytes.len());
    Ok(bytes)
}

/// Transcribe audio bytes via VAD+STT and reply with the result.
async fn transcribe_and_reply(
    message: &grammers_client::update::Message,
    bytes: &[u8],
    state: &AppState,
    label: &str,
) -> Result<()> {
    let pcm = audio::decode_to_pcm(bytes)?;
    let duration_secs = pcm.len() as f64 / 16_000.0;
    info!("Telegram {label}: {:.1}s of audio", duration_secs);

    let segments = state.vad_transcribe(&pcm)?;
    let full_text: String = segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");

    if full_text.is_empty() {
        message.reply("No speech detected.").await?;
    } else {
        send_long_reply(message, &full_text).await?;
    }
    Ok(())
}

/// Split text into chunks of at most TELEGRAM_MAX_LEN and send each as a reply.
async fn send_long_reply(
    message: &grammers_client::update::Message,
    text: &str,
) -> Result<()> {
    if text.len() <= TELEGRAM_MAX_LEN {
        message.reply(text).await?;
        return Ok(());
    }

    for chunk in split_text(text, TELEGRAM_MAX_LEN) {
        message.reply(chunk).await?;
    }
    Ok(())
}

fn split_text(text: &str, max_len: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let remaining = text.len() - start;
        if remaining <= max_len {
            chunks.push(&text[start..]);
            break;
        }

        // Find a split point at a space boundary within max_len
        let end = start + max_len;
        let split_at = text[start..end]
            .rfind(' ')
            .map(|pos| start + pos)
            .unwrap_or(end);

        // Ensure we're at a char boundary
        let split_at = if !text.is_char_boundary(split_at) {
            let mut pos = split_at;
            while pos > start && !text.is_char_boundary(pos) {
                pos -= 1;
            }
            pos
        } else {
            split_at
        };

        chunks.push(&text[start..split_at]);
        start = if split_at < text.len() && text.as_bytes()[split_at] == b' ' {
            split_at + 1
        } else {
            split_at
        };
    }

    chunks
}
