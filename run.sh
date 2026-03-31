#!/bin/bash
set -euo pipefail
cd /home/scott/code/telemuze
source .env

# Uncomment for debug logging (shows recognition tokens, hotword encoding, etc.)
# export RUST_LOG=telemuze=debug

exec ./target/debug/telemuze
