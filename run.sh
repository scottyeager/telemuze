#!/bin/bash
set -euo pipefail
cd /home/scott/code/telemuze
source .env

# The binary's rpath is $ORIGIN (same dir as binary); for local dev, point the
# dynamic linker at ~/.local/lib where the sherpa-onnx libraries are installed.
export LD_LIBRARY_PATH="${HOME}/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Uncomment for debug logging (shows recognition tokens, hotword encoding, etc.)
# export RUST_LOG=telemuze=debug

exec ./target/debug/telemuze --stt-model-path ~/.local/share/telemuze/models/parakeet-unified-en-0.6b
