# Telemuze

Telemuze is a speech-to-text server with optional LLM post-processing. It supports Nvidia's Parakeet STT model and Silero voice activity detection, for now.

We borrow inspiration and code from the excellent [Handy](https://github.com/cjpais/Handy) app and libraries developed for that project.

There's also a Telegram bot feature, enabling access to the server over Telegram.

## Prerequisites

- Rust toolchain (stable) — install via [rustup](https://rustup.rs/)
- C compiler and `clang` (needed by ONNX Runtime / bindgen)
- On Debian/Ubuntu: `sudo apt install build-essential clang libssl-dev pkg-config`

## Build

```sh
cargo build --release
```

The binary is written to `target/release/telemuze`.

## Run

```sh
# Minimal — models are auto-downloaded on first run
./target/release/telemuze

# With LLM correction (requires an external OpenAI-compatible API for now)
./target/release/telemuze --llm-api-url http://127.0.0.1:8081/v1/chat/completions
```

By default the server listens on `0.0.0.0:7313`. Override with `--host` / `--port` or the environment variables below.

## Configuration

All options can be set via CLI flags or environment variables.

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--host` | `TELEMUZE_HOST` | `0.0.0.0` | Bind address |
| `--port` | `TELEMUZE_PORT` | `7313` | Listen port |
| `--stt-model-path` | `TELEMUZE_STT_MODEL_PATH` | auto-download | Path to Parakeet ONNX model directory |
| `--vad-model-path` | `TELEMUZE_VAD_MODEL_PATH` | auto-download | Path to Silero VAD ONNX model file |
| `--models-dir` | `TELEMUZE_MODELS_DIR` | `~/.local/share/telemuze/models` | Directory for downloaded models |
| `--llm-api-url` | `TELEMUZE_LLM_API_URL` | *(disabled)* | OpenAI-compatible chat completions URL (see [Planned](#planned)) |
| `--custom-terms` | `TELEMUZE_CUSTOM_TERMS` | | Comma-separated custom dictionary terms |
| `--telegram-api-id` | `TELEGRAM_API_ID` | | Telegram API ID (from https://my.telegram.org) |
| `--telegram-api-hash` | `TELEGRAM_API_HASH` | | Telegram API hash |
| `--telegram-bot-token` | `TELEGRAM_BOT_TOKEN` | | Telegram bot token (from @BotFather) |

## API Endpoints

- **`POST /v1/audio/transcriptions`** — OpenAI-compatible transcription endpoint. Upload audio as multipart form data.
- **`POST /v1/dictate/smart`** — Smart dictation: STT + optional LLM correction (currently requires an external API). Returns plain text.
- **`POST /v1/transcribe/long`** — Long-form transcription with VAD-based chunking.

## Desktop Client

A toggle-style dictation script is included in `scripts/smart_dictation_client.sh`. Bind it to a global hotkey — first press starts recording, second press transcribes and types the result into the focused window.

Requires: `arecord`, `curl`, and `wtype` (Wayland) or `xdotool` (X11).

```sh
# Configure the server URL (defaults to http://127.0.0.1:7313)
export TELEMUZE_URL=http://127.0.0.1:7313
```

## Planned

- **In-process LLM inference** — LLM post-processing currently requires an external OpenAI-compatible server (e.g., llama.cpp, mistral.rs, Ollama). The plan is to bring LLM inference in-process so Telemuze is fully self-contained.

