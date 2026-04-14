#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Building telemuze server ==="
cargo build --release

echo ""
echo "=== Building distribution binary ==="

# Resolve the actual .so paths (follow symlinks). sherpa-onnx-sys's build
# script copies the runtime shared libs into target/<profile>/ next to
# the binary, so that's the default source.
SHERPA_SO=$(readlink -f "${TELEMUZE_SHERPA_SO:-$(pwd)/target/release/libsherpa-onnx-c-api.so}")
ONNXRUNTIME_SO=$(readlink -f "${TELEMUZE_ONNXRUNTIME_SO:-$(pwd)/target/release/libonnxruntime.so}")

TELEMUZE_SERVER_BIN="$(pwd)/target/release/telemuze" \
TELEMUZE_SHERPA_SO="$SHERPA_SO" \
TELEMUZE_ONNXRUNTIME_SO="$ONNXRUNTIME_SO" \
cargo build --release -p telemuze-dist

echo ""
echo "=== Done ==="
ls -lh target/release/telemuze-dist
echo ""
echo "Distribution binary: target/release/telemuze-dist"
echo "  (self-contained — embeds server + shared libraries)"
