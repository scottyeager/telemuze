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

## Docker images

There are two Dockerfiles, one for each role. These can be used for local testing or for deployment to the ThreeFold Grid as flists.

Here's an example of building the containers for local use. To push them remotely, you'll need to adjust the tags according to your username on Docker Hub, for example.

- Listener
```
docker buildx build -t telemuze-listener -f listener/Dockerfile .
```
- Composer
```
docker buildx build -t telemuze-listener -f composer/Dockerfile .
```
