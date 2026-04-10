//! Telegram bot interface for Telemuze.
//!
//! Connects via MTProto (grammers) so it can handle large file downloads
//! (up to 2 GB) without needing a self-hosted Bot API server.
//! Voice notes go through the smart dictation pipeline (STT + LLM),
//! while audio/video file attachments use the long-form pipeline (VAD + STT).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use grammers_client::client::{Client, UpdatesConfiguration};
use grammers_client::media::Media;
use grammers_client::message::{InputMessage, Message as GrammersMessage};
use grammers_client::update::Update;
use grammers_client::InvocationError;
use grammers_client::SenderPool;
use grammers_session::storages::SqliteSession;
use tokio::time::sleep;
use tracing::{error, info, warn};

use crate::audio;
use crate::long_form;
use crate::state::AppState;

/// Duration threshold (seconds) above which Telegram audio is routed
/// through the long-form subprocess pipeline. Below this we keep the
/// in-process VAD+STT loop used for voice dictations.
const LONG_FORM_THRESHOLD_SECS: f64 = 60.0;

/// Maximum length for a single Telegram message.
const TELEGRAM_MAX_LEN: usize = 4096;

/// Default delay before reconnecting after an error.
const DEFAULT_RETRY_DELAY: Duration = Duration::from_secs(5);

/// Return the path used to persist the Telegram MTProto session.
fn session_path() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("telemuze")
        .join("telegram.session")
}

/// Run the Telegram bot, reconnecting automatically on errors.
pub async fn run_bot(
    api_id: i32,
    api_hash: String,
    token: String,
    state: Arc<AppState>,
) -> ! {
    loop {
        match run_bot_once(api_id, api_hash.clone(), token.clone(), &state).await {
            Ok(()) => {
                warn!("Telegram bot exited unexpectedly, reconnecting in {DEFAULT_RETRY_DELAY:?}...");
                sleep(DEFAULT_RETRY_DELAY).await;
            }
            Err(e) => {
                let delay = flood_wait_delay(&e).unwrap_or(DEFAULT_RETRY_DELAY);
                error!("Telegram bot error: {e:#}, reconnecting in {delay:?}...");
                sleep(delay).await;
            }
        }
    }
}

/// Extract the FLOOD_WAIT delay from an error chain, if present.
fn flood_wait_delay(err: &anyhow::Error) -> Option<Duration> {
    for cause in err.chain() {
        if let Some(InvocationError::Rpc(rpc)) = cause.downcast_ref::<InvocationError>() {
            if rpc.name == "FLOOD_WAIT" {
                if let Some(secs) = rpc.value {
                    return Some(Duration::from_secs(secs as u64 + 1));
                }
            }
        }
    }
    None
}

async fn run_bot_once(
    api_id: i32,
    api_hash: String,
    token: String,
    state: &Arc<AppState>,
) -> Result<()> {
    let path = session_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let session = Arc::new(SqliteSession::open(&path).await?);
    let pool = SenderPool::new(Arc::clone(&session), api_id);

    let client = Client::new(pool.handle);

    // The runner drives all network I/O; it must be spawned.
    tokio::spawn(pool.runner.run());

    // Reuse existing session if still authorized; only sign in when needed.
    if !client.is_authorized().await? {
        info!("Signing in to Telegram...");
        client.bot_sign_in(&token, &api_hash).await?;
    }
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
                        error!("Error handling Telegram message: {e:#}");
                        let _ = message.reply(format!("Error: {e:#}")).await;
                    }
                });
            }
            _ => {}
        }
    }
}

/// A status message that can be updated and eventually deleted.
struct StatusMessage {
    msg: GrammersMessage,
}

impl StatusMessage {
    /// Send an initial status reply to the user's message.
    async fn new(
        reply_to: &grammers_client::update::Message,
        text: &str,
    ) -> Result<Self> {
        let msg = reply_to.reply(text).await?;
        Ok(Self { msg })
    }

    /// Update the status message text.
    async fn update(&self, text: &str) {
        if let Err(e) = self.msg.edit(InputMessage::new().text(text)).await {
            warn!("Failed to update status message: {e}");
        }
    }

    /// Delete the status message.
    async fn delete(self) {
        if let Err(e) = self.msg.delete().await {
            warn!("Failed to delete status message: {e}");
        }
    }
}

async fn handle_message(
    client: &Client,
    message: &grammers_client::update::Message,
    state: &Arc<AppState>,
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
                let status = StatusMessage::new(message, "Receiving voice note...").await?;
                let bytes = download_media(client, &media).await?;
                transcribe_and_reply(client, message, &bytes, state, "voice note", status).await?;
            } else if is_audio_video {
                info!(
                    "Telegram: audio/video file ({mime}) from {:?}",
                    message.sender_id()
                );
                let status = StatusMessage::new(message, "Receiving file...").await?;
                let bytes = download_media(client, &media).await?;
                transcribe_and_reply(client, message, &bytes, state, "long-form", status).await?;
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

/// Transcribe audio bytes and reply with the result. Routes to the
/// in-process VAD+STT loop for short clips (≤60 s, voice-dictation style)
/// or to the long-form subprocess pipeline for longer audio (multi-speaker
/// meetings, podcasts), matching the HTTP endpoint's behavior.
async fn transcribe_and_reply(
    client: &Client,
    message: &grammers_client::update::Message,
    bytes: &[u8],
    state: &Arc<AppState>,
    label: &str,
    status: StatusMessage,
) -> Result<()> {
    status.update("Decoding audio...").await;
    let pcm = audio::decode_to_pcm(bytes)?;
    let duration_secs = pcm.len() as f64 / 16_000.0;
    info!("Telegram {label}: {:.1}s of audio", duration_secs);

    let duration_display = long_form::format_duration(duration_secs);

    if duration_secs <= LONG_FORM_THRESHOLD_SECS {
        transcribe_short(message, &pcm, state, &duration_display, status).await?;
    } else {
        transcribe_long(client, message, &pcm, state, &duration_display, status).await?;
    }
    Ok(())
}

/// In-process path for short voice dictations (≤60 s). No subprocess,
/// no queue, no diarization — replies with plain text.
async fn transcribe_short(
    message: &grammers_client::update::Message,
    pcm: &[f32],
    state: &AppState,
    duration_display: &str,
    status: StatusMessage,
) -> Result<()> {
    status
        .update(&format!("Transcribing {duration_display} of audio..."))
        .await;

    let segments = state.vad_segment(pcm)?;
    let total = segments.len();

    let mut results = Vec::with_capacity(total);
    for (i, seg) in segments.iter().enumerate() {
        if total > 1 {
            let position = long_form::format_duration(seg.start_secs);
            status
                .update(&format!(
                    "Transcribing {duration_display} of audio... (segment {}/{total}, {position})",
                    i + 1,
                ))
                .await;
        }

        if let Some(result) = state.transcribe_segment(seg, i, total, None) {
            results.push(result);
        }
    }

    let full_text: String = results
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    status.delete().await;

    if full_text.is_empty() {
        message.reply("No speech detected.").await?;
    } else {
        message.reply(full_text.as_str()).await?;
    }
    Ok(())
}

/// Long-form path (>60 s): routes through the same subprocess pipeline
/// the HTTP endpoint uses, with speaker labels and timestamps in the
/// rendered output. Queues behind any other long-form job automatically.
async fn transcribe_long(
    client: &Client,
    message: &grammers_client::update::Message,
    pcm: &[f32],
    state: &Arc<AppState>,
    duration_display: &str,
    status: StatusMessage,
) -> Result<()> {
    // The semaphore acquisition inside long_form_transcribe is FIFO;
    // we show a coarse "Queued" state up front and then transition to
    // "Transcribing" once we're running.
    status
        .update(&format!(
            "Queued for long-form transcription ({duration_display})..."
        ))
        .await;

    let outcome = state.long_form_transcribe(pcm, None).await?;

    status
        .update(&format!("Finalizing transcript ({duration_display})..."))
        .await;

    let subs = long_form::finalize(outcome);
    let text = long_form::format_as_text(&subs);

    status.delete().await;

    if text.is_empty() {
        message.reply("No speech detected.").await?;
    } else {
        send_reply(client, message, &text).await?;
    }
    Ok(())
}

/// Send the transcript as a reply. If it fits in a single message, send as text.
/// If it exceeds the Telegram limit, upload as a .txt file instead.
async fn send_reply(
    client: &Client,
    message: &grammers_client::update::Message,
    text: &str,
) -> Result<()> {
    if text.len() <= TELEGRAM_MAX_LEN {
        message.reply(text).await?;
        return Ok(());
    }

    // Too long for a message — send as a text file.
    let bytes = text.as_bytes();
    let mut cursor = std::io::Cursor::new(bytes);
    let uploaded = client
        .upload_stream(&mut cursor, bytes.len(), "transcript.txt".into())
        .await?;
    let input = InputMessage::new()
        .text("Transcript too long for a message — sent as file.")
        .file(uploaded);
    message.reply(input).await?;

    Ok(())
}

