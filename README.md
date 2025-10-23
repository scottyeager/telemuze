# Telemuze

Telemuze is a Telegram bot that transcribes audio/video to plain text using WhisperX. It's designed to use VMs on the ThreeFold Grid to do the transcription work. One VM is a long running "listener" that uses minimal resources. When a job arrives, an ephemeral "composer" VM is spawned to do the heavy lifting.

## Quickstart

```
TODO: Provide instructions here based on prebuilt flists available from TF Hub
```

## Run with local listener

To run the listener locally, Python can be used directly. Some requirements must be satisfied:

* [`tfcmd`](https://github.com/threefoldtech/tfgrid-sdk-go/tree/development/grid-cli) is installed (or using local composer)
* TF Chain account funded and ready to create deployments
* A Telegram bot (see @BotFather)
* [uv](https://github.com/astral-sh/uv) (you can get by without it if you know what you're doing, but really, it's great)

Make a venv, install the dependencies, and run the code with minimal config:

```bash
uv venv
uv pip install listener/requirements.text
TELEGRAM_BOT_TOKEN="ABC:123" TF_MNEMONIC="your words here" TF_NODE_ID="13" ALLOWED_USERNAMES="your_tg_username" uv run main.py
```

The node id will be used to deploy the composers that will perform the transcription. It's also possible to use a predeployed composer, such as a Docker container, via an override IP as described below. There are many more config parameters--see the top of `main.py` for reference.

Telemuze will put some files under `~/.telemuze` if you run the listener locally. Sry, will fix that later.

## Composer override

Mainly for testing purposes, it's possible to configure an IP address to be used as the composer. This could be a Docker container, which is useful for local testing.

Set `COMPOSER_IP_OVERRIDE`, and then the listener won't attempt to deploy a composer. It will instead simply attempt to connect to the provided IP address over SSH.

There's also `SSH_KEY_OVERRIDE_PATH`, to set the path to an SSH private key file to use when connecting to the composer VM. Telemuze generates its own SSH key on first run and stores it under `~/.telemuze` for subsequent runs. Just using that probably makes more sense than overriding the key. This might be removed later.

You are responsible for making sure the SSH key that the listener uses has been loaded into the composer, when using an override IP. The composer image doesn't have anything fancy to help with that, yet. Under normal operation, the listener will inject its key into the composer VM.

## Telegram Bot API

Telegram bots usually rely on an API server provided by Telegram. That server offers a convenient interface with Telegram's backend, but it comes with certain restrictions. Most notably for this project, there is a file size limitation of 20mb for files sent to the bot.

There are basically two ways around these restrictions:

1. Use a client that communicates directly with the Telegram backend (eg, Telethon)
2. Host a copy of the Bot API server

Telemuze opts for the second path, allowing users to optionally run the API server alongside the bot.

Along with the regular bot token, this also requires obtaining an app id and hash from https://my.telegram.org/

See the next section for how to launch a Bot API server inside the Telemuze Docker image.

## Docker

This repo contains a single Dockerfile that bundles the following:

* Telemuze listener and composer code and dependencies
* Whisper tiny and turbo models precached
* Telegram Bot API server prebuilt binary

The Docker image can be used directly or converted into an flist to run as a VM on the ThreeFold Grid. Prebuilt Docker images are available from the [packages section](https://github.com/scottyeager/telemuze/pkgs/container/telemuze) of this repo.

### Building

To build the Dockerfile yourself, run:

```
docker buildx build -t telemuze .
```

### Running

Here's a template invocation, showing the required environment variables. Fill in your own values:

```
docker run --rm \
  -e TELEGRAM_BOT_TOKEN="ABC:123" \
  -e TF_MNEMONIC="your threefold mnemonic here" \
  -e TF_NODE_ID="13" \
  -e ALLOWED_USERNAMES="your_telegram_username" \
  telemuze
```

### Running with Bot API server

To start and use the bundled Bot API server, just add your credentials as follows:

```
docker run --rm \
  -e TELEGRAM_BOT_TOKEN="ABC:123" \
  -e TF_MNEMONIC="your threefold mnemonic here" \
  -e TF_NODE_ID="13" \
  -e ALLOWED_USERNAMES="your_telegram_username" \
  -e TELEGRAM_API_ID="your_api_id" \
  -e TELEGRAM_API_HASH="your_api_hash" \
  telemuze
```

When both the id and hash are present, the server will start automatically and the listener will use it.
