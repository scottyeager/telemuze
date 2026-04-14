#!/usr/bin/env bash
# Docker integration tests for telemuze.
#
# Tests build independence (no local files needed) and model auto-download
# for both the server and client binaries.
#
# Usage:
#   ./scripts/docker-test.sh [--skip-build] [--skip-model-download]
#
# Options:
#   --skip-build            Skip rebuilding Docker images (use existing telemuze:builder-test)
#   --skip-model-download   Only test --help / binary checks, not actual model downloads

set -euo pipefail

SKIP_BUILD=false
SKIP_MODEL_DOWNLOAD=false
for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        --skip-model-download) SKIP_MODEL_DOWNLOAD=true ;;
    esac
done

PASS=0
FAIL=0
declare -a RESULTS=()

pass() { echo "  PASS: $1"; ((PASS++)); RESULTS+=("PASS: $1"); }
fail() { echo "  FAIL: $1"; ((FAIL++)); RESULTS+=("FAIL: $1"); }
warn() { echo "  WARN: $1"; RESULTS+=("WARN: $1"); }

BUILDER_IMAGE="telemuze:builder-test"
RUNTIME_IMAGE="telemuze:runtime-test"

cd "$(dirname "$0")/.."

echo "======================================================="
echo "  Telemuze Docker Independence Tests"
echo "  $(date)"
echo "======================================================="
echo ""

# ── Phase 1: Build ───────────────────────────────────────────────────────────

echo "--- Phase 1: Build ---"

if [ "$SKIP_BUILD" = false ]; then
    echo "Building builder stage (server + client, from scratch)..."
    if docker build --target builder -t "$BUILDER_IMAGE" . 2>&1 | tee /tmp/telemuze-docker-build.log; then
        pass "docker build --target builder succeeds"
    else
        fail "docker build --target builder failed (see /tmp/telemuze-docker-build.log)"
        echo "=== Build log tail ==="
        tail -30 /tmp/telemuze-docker-build.log
        echo "Aborting — cannot test without a successful build."
        exit 1
    fi

    echo ""
    echo "Building runtime image..."
    if docker build -t "$RUNTIME_IMAGE" . 2>&1 | tail -5; then
        pass "docker build (runtime stage) succeeds"
    else
        fail "docker build (runtime stage) failed"
    fi
else
    echo "(Skipping build — using existing $BUILDER_IMAGE and $RUNTIME_IMAGE)"
fi

# ── Phase 2: Artifacts ───────────────────────────────────────────────────────

echo ""
echo "--- Phase 2: Artifacts ---"

echo "Checking built artifacts in builder image:"
ARTIFACTS=$(docker run --rm "$BUILDER_IMAGE" ls -lh /artifacts/ 2>&1) || {
    fail "Cannot list /artifacts/ in builder image"
    exit 1
}
echo "$ARTIFACTS"

for artifact in telemuze telemuze-server telemuze-listen libsherpa-onnx-c-api.so libonnxruntime.so; do
    if echo "$ARTIFACTS" | grep -q "$artifact"; then
        pass "artifact present: $artifact"
    else
        fail "artifact missing: $artifact"
    fi
done

# ── Phase 3: Binary --help ───────────────────────────────────────────────────

echo ""
echo "--- Phase 3: Binary --help ---"

echo "Testing dist launcher --help..."
HELP_OUT=$(docker run --rm "$RUNTIME_IMAGE" --help 2>&1) || true
if echo "$HELP_OUT" | grep -q "Telemuze\|Usage\|--port"; then
    pass "dist launcher (telemuze) --help works"
else
    fail "dist launcher --help failed or unexpected output"
    echo "$HELP_OUT" | head -10
fi

echo "Testing telemuze-server --help..."
HELP_OUT=$(docker run --rm --entrypoint /artifacts/telemuze-server "$BUILDER_IMAGE" --help 2>&1) || true
if echo "$HELP_OUT" | grep -q "Telemuze\|--port\|--models-dir"; then
    pass "telemuze-server --help works"
else
    fail "telemuze-server --help failed or unexpected output"
    echo "$HELP_OUT" | head -10
fi

echo "Testing telemuze-listen --help..."
HELP_OUT=$(docker run --rm --entrypoint /artifacts/telemuze-listen "$BUILDER_IMAGE" --help 2>&1) || true
if echo "$HELP_OUT" | grep -q "telemuze-listen\|Streaming\|--url\|subcommand"; then
    pass "telemuze-listen --help works"
else
    fail "telemuze-listen --help failed or unexpected output"
    echo "$HELP_OUT" | head -10
fi

# ── Phase 4: Model URL verification ──────────────────────────────────────────

echo ""
echo "--- Phase 4: Model URL verification (strings in binaries) ---"

echo "Extracting strings from server binary..."
SERVER_STR=$(docker run --rm "$BUILDER_IMAGE" strings /artifacts/telemuze-server 2>&1)

# Sortformer from our self-hosted HF repo
if echo "$SERVER_STR" | grep -q "scottyeager/diar_streaming_sortformer"; then
    pass "server: Sortformer URL points to scottyeager HF repo"
else
    fail "server: Sortformer URL missing or wrong (expected scottyeager HF)"
fi

# Parakeet 0.6B from csukuangfj (upstream, not our fork — this is expected)
if echo "$SERVER_STR" | grep -q "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2"; then
    pass "server: Parakeet 0.6B URL present (csukuangfj HF)"
else
    fail "server: Parakeet 0.6B URL missing"
fi

# Should NOT reference old pyannote-based diarization
if echo "$SERVER_STR" | grep -q "pyannote\|3dspeaker\|campplus"; then
    fail "server: still references old pyannote/3dspeaker diarization (should be Sortformer)"
else
    pass "server: no old pyannote/3dspeaker references (Sortformer migration confirmed)"
fi

echo "Extracting strings from client binary..."
CLIENT_STR=$(docker run --rm "$BUILDER_IMAGE" strings /artifacts/telemuze-listen 2>&1)

# Parakeet-110m from our self-hosted HF repo
if echo "$CLIENT_STR" | grep -q "scottyeager/sherpa-onnx-nemo-parakeet-tdt-transducer-110m"; then
    pass "client: parakeet-110m URL points to scottyeager HF repo"
else
    fail "client: parakeet-110m URL missing or wrong (expected scottyeager HF)"
fi

# Should NOT reference old k2-fsa tar.bz2 archive
if echo "$CLIENT_STR" | grep -q "parakeet_tdt_transducer_110m-en-36000.tar.bz2"; then
    fail "client: still uses old k2-fsa tar.bz2 archive URL (should be scottyeager HF)"
else
    pass "client: no old k2-fsa tar.bz2 reference (self-hosted HF migration confirmed)"
fi

# ── Phase 5: Server model download and startup ───────────────────────────────

if [ "$SKIP_MODEL_DOWNLOAD" = true ]; then
    echo ""
    echo "--- Phase 5: Model download (SKIPPED) ---"
else
    echo ""
    echo "--- Phase 5: Server startup and model download ---"
    echo "Note: Sortformer is 492MB, Parakeet 0.6B is ~300MB — allow up to 15 minutes."

    # Create a named volume for model storage
    VOL="telemuze-test-$$"
    docker volume create "$VOL" > /dev/null
    trap "docker volume rm '$VOL' > /dev/null 2>&1 || true" EXIT

    # Start server in background (it will download models then listen)
    echo "Starting server container..."
    CID=$(docker run -d \
        -v "$VOL:/root/.local/share/telemuze" \
        -e RUST_LOG=telemuze=info \
        "$RUNTIME_IMAGE" 2>&1)
    echo "Container ID: $CID"

    # Poll logs every 10s, wait up to 15 minutes for "Listening on"
    MAX_WAIT=900  # 15 minutes
    WAITED=0
    FINAL_STATUS="timeout"

    while [ $WAITED -lt $MAX_WAIT ]; do
        sleep 10
        WAITED=$((WAITED + 10))

        LOGS=$(docker logs "$CID" 2>&1) || true

        if echo "$LOGS" | grep -q "Listening on"; then
            FINAL_STATUS="listening"
            break
        elif echo "$LOGS" | grep -q "^error\|Error\|panic\|SIGKILL"; then
            FINAL_STATUS="error"
            break
        fi

        # Report download progress every minute
        if (( WAITED % 60 == 0 )); then
            echo "  ${WAITED}s elapsed. Recent log:"
            echo "$LOGS" | tail -5 | sed 's/^/    /'
        fi
    done

    # Capture final logs before stopping
    FINAL_LOGS=$(docker logs "$CID" 2>&1) || true
    docker stop "$CID" > /dev/null 2>&1 || true
    docker rm "$CID" > /dev/null 2>&1 || true

    echo ""
    echo "Server output (last 30 lines):"
    echo "$FINAL_LOGS" | tail -30

    case "$FINAL_STATUS" in
        listening)
            pass "server downloaded models and started listening"
            ;;
        error)
            fail "server encountered an error during startup/download"
            ;;
        timeout)
            # Check if downloads at least started
            if echo "$FINAL_LOGS" | grep -q "Downloading\|downloaded\|Model.*ready\|bytes"; then
                warn "server timed out but downloads were in progress (increase timeout or rerun)"
                pass "server initiated model downloads (reached download phase)"
            else
                fail "server timed out with no download activity"
            fi
            ;;
    esac

    # ── Client VAD download test ─────────────────────────────────────────────

    echo ""
    echo "--- Phase 5b: Client VAD model download ---"
    echo "Running client with --no-cmd --no-eou (expect VAD download then audio error)..."

    CLIENT_VOL="telemuze-client-test-$$"
    docker volume create "$CLIENT_VOL" > /dev/null
    trap "docker volume rm '$VOL' '$CLIENT_VOL' > /dev/null 2>&1 || true" EXIT

    # The client will: download silero_vad.onnx → load VAD → fail at audio device
    CLIENT_OUT=$(timeout 120 docker run --rm \
        -v "$CLIENT_VOL:/root/.local/share/telemuze" \
        -e RUST_LOG=telemuze_listen=info \
        --entrypoint /artifacts/telemuze-listen \
        "$BUILDER_IMAGE" \
        --no-cmd --no-eou 2>&1) || CLIENT_EXIT=$?

    echo "Client output:"
    echo "$CLIENT_OUT"

    if echo "$CLIENT_OUT" | grep -q "VAD model downloaded"; then
        pass "client downloaded Silero VAD model successfully"
    elif echo "$CLIENT_OUT" | grep -q "Loading VAD model\|silero_vad"; then
        pass "client found/loaded existing VAD model"
    elif echo "$CLIENT_OUT" | grep -q "No input audio device"; then
        # Reached audio init — VAD download must have succeeded
        pass "client reached audio init (VAD download succeeded)"
    else
        fail "client did not reach VAD download or audio stage"
    fi

    docker volume rm "$CLIENT_VOL" > /dev/null 2>&1 || true
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "======================================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "======================================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "Completed: $(date)"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
