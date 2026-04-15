# Telemuze

Telemuze is a personal speech-to-text server and desktop client for Linux. It supports Nvidia's Parakeet STT models, Nvidia Sortformer for diarization, and Silero voice activity detection.

Features at a glance:

* **Client**
  * Dictate into any application
  * Supports full hands free operation with wake/sleep voice commands
  * Trigger arbitrary key presses ("press enter") and mouse clicks
* **Server**
  * Web UI for file uploads and speaker labeling for up to four speakers
  * Telegram bot support (forward voice notes to the bot, send large files)
  * Automatic conversion of all video and audio formats supported by `ffmpeg`


## Quickstart

Pre-built binaries and Docker images are available.

## Start the server with Docker

Generate a starter `server.toml` with default settings:

```sh
docker run --rm ghcr.io/scottyeager/telemuze:latest --dump-config > server.toml
```

Edit it as needed, then mount it into the container on each run:

```sh
docker run -p 7313:7313 \
  -v telemuze-models:/root/.local/share/telemuze \
  -v "$PWD/server.toml:/root/.config/telemuze/server.toml" \
  ghcr.io/scottyeager/telemuze:latest
```

If you also use a `terms.txt` file, create it and mount it in the same path inside the container.

## Install the server using self-extracting binary

```sh
curl -Lo telemuze https://github.com/scottyeager/telemuze/releases/latest/download/telemuze-linux-x86_64
chmod +x telemuze
# Move the binary somewhere in your PATH, for example:
sudo mv teleumuze /usr/local/bin
telemuze
```

On first run the binary extracts the server and its shared libraries into `~/.local/share/telemuze/`, then execs the server. Subsequent runs skip extraction if the versi
on is unchanged. All flags are passed through to the server.

## Download and install pre-compiled client

```sh
curl -Lo telemuze-listen https://github.com/scottyeager/telemuze/releases/latest/download/telemuze-listen-linux-x86_64
chmod +x telemuze-listen
# Move the binary somewhere in your PATH, for example:
sudo mv teleumuze-listen /usr/local/bin
telemuze-listen
```

## Uninstall

Telemuze writes files in two places. Just delete them like so:

```
rm -rf ~/.local/share/telemuze/
rm -rf ~/.config/telemuze/
```



## Build

Due to the use of ONNX runtime, both Rust and C++ compilers are needed to build the full project. 

### Prerequisites

- Rust toolchain (stable) — install via [rustup](https://rustup.rs/)
- C compiler and `clang` (needed by ONNX Runtime / bindgen)
- On Debian/Ubuntu: `sudo apt install build-essential clang libssl-dev pkg-config`

```sh
cargo build --release
```

The binary is written to `target/release/telemuze`.

### Run

```sh
# Minimal — models are auto-downloaded on first run
./target/release/telemuze
```

By default the server listens on `0.0.0.0:7313`. Override with `--host` / `--port`.

## Configuration

Configuration can be done by config file, CLI flags, or environment variables. Under typical use, a config file is recommended.

Both the client and the server will write a config file containing the defaults on the first run of `--update-config`:

```
# Server config at ~/.config/telemuze/server.toml
telemuze --update-config

# Client config at ~/.config/telemuze/listen.toml
telemuze-listen --update-config
```

You can then edit the config options to your liking and launch the program. You'll find comments inline explaining the various options.

If a new version introduces new config options, running the update config command again will update the config file with any new options and their default. As a word of warning, this will also delete any config values that have been deprecated. Taking a backup of the existing config file first might be a good idea.

## Full Client

The desktop client is available as a compiled binary that should work on reasonably modern glibc based Linux distros. Both Xorg and Wayland (with SNI) are supported. Personally I use X and thus that's the better tested side.

### Features
* System tray icon indicating status
* Always-on listening, with transcription only during active speech. Uses two stage gating of audio energy and then VAD to reduce CPU load when no speech present
* Keyboard and mouse control features ("press control-alt-tab", click in different parts of the screen, scrolling, etc)
* Undo the last text input by saying "undo"
* Multiple input methods, including simulated typing, ctrl-c paste, and ctrl-shift-c paste. Different methods can be configured per program

### Running

First, start the client binary (consider a systemd service or similar for convenience):

```
telemuze-listen
```

You can then toggle listening on/off by clicking (Xorg) or double clicking (Wayland) on the tray icon.

The client also supports signaling to the running process, which can be useful for binding to hotkeys, for example:

```
telemuze-listen toggle
```

## Bash Client

There's a very basic `bash` based client in `scripts/smart_dictation_client.sh`. It's not under active development and was never tested on Wayland. Might still be a starting place for anyone wanting a lighter weight client than the full desktop client.

Requires: `arecord`, `curl`, and `wtype` (Wayland) or `xdotool` (X11).

```sh
# Configure the server URL (defaults to http://127.0.0.1:7313)
export TELEMUZE_URL=http://127.0.0.1:7313
```

## WIP

* **In-process LLM inference** — While this is implemented in a basic way, making a small LLM really useful in this setting is more challenging
  * The target is models that perform decently on CPU and don't require huge amounts of RAM (~2B params and under)
  * One use case would be cleaning up dictation outputs, to remove filler works like "uh" and "um", for example
  * Identifying the names of speakers when doing diarization could be another use case
* **Phonetic and fuzzy word replacement** — The idea here is to provide a dictionary of special words and then try to fix mistakes where the speech model doesn't correctly write them. Promising results, but needs more work
  * This was an initial idea before discovering hotword boosting
  * The code is still present for future experimentation but not currently used

## Acknowledgements

[Sherpa ONNX](https://github.com/k2-fsa/sherpa-onnx) does a lot of the heavy lifting for STT.

[Handy](https://github.com/cjpais/Handy) provided a lot of inspiration for what could be possible with local speech-to-text on Linux. The Telemuze client takes heavy inspiration from Handy.

[parakeet-rs](https://github.com/altunenes/parakeet-rs) for demonstrating the use of Nvidia's Sortformer diarization model with ONNX (added to Sherpa ONNX on [my fork](https://github.com/scottyeager/sherpa-onnx) for now).
