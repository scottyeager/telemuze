#!/bin/bash
# smart_dictation_client.sh
#
# Toggle-style dictation client for Telemuze.
# Bind this script to a global hotkey (e.g., Super+Space in GNOME/KDE).
#
# First press:  Starts recording audio from the microphone.
# Second press: Stops recording, sends audio to the server, types the result.
#
# Dependencies: arecord, curl, wtype (Wayland) or xdotool (X11)
#
# Configuration via environment variables:
#   TELEMUZE_URL  - Server URL (default: http://127.0.0.1:7313)
#   TELEMUZE_TYPE - "wayland" or "x11" (auto-detected if unset)

set -euo pipefail
exec 2>/tmp/dictation.log
set -x

PID_FILE="/tmp/dictation.pid"
AUDIO_FILE="/tmp/dictation.wav"
NOTIFY_ID_FILE="/tmp/dictation_notify.id"
SERVER_URL="${TELEMUZE_URL:-http://127.0.0.1:7313}/v1/dictate/smart"

# Show a persistent notification via D-Bus, print its numeric ID to stdout
show_notification() {
    local summary="$1"
    local body="${2:-}"
    local replaces_id="${3:-0}"
    local result
    result=$(gdbus call --session \
        --dest org.freedesktop.Notifications \
        --object-path /org/freedesktop/Notifications \
        --method org.freedesktop.Notifications.Notify \
        "Telemuze" "$replaces_id" "audio-input-microphone" \
        "$summary" "$body" '[]' '{"urgency": <byte 1>}' 0 \
        2>/dev/null) || true
    echo "$result" | sed 's/.* \([0-9][0-9]*\).*/\1/'
}

# Dismiss a notification by ID
dismiss_notification() {
    local notify_id="$1"
    gdbus call --session \
        --dest org.freedesktop.Notifications \
        --object-path /org/freedesktop/Notifications \
        --method org.freedesktop.Notifications.CloseNotification \
        "$notify_id" >/dev/null 2>&1 || true
}

# Auto-detect display server
detect_display_server() {
    if [ -n "${TELEMUZE_TYPE:-}" ]; then
        echo "$TELEMUZE_TYPE"
    elif [ -n "${WAYLAND_DISPLAY:-}" ]; then
        echo "wayland"
    else
        echo "x11"
    fi
}

# Type text into the currently focused window
type_text() {
    local text="$1"
    local display_type
    display_type=$(detect_display_server)

    if [ "$display_type" = "wayland" ]; then
        wtype "$text "
    else
        xdotool type --clearmodifiers "$text "
    fi
}

# Play a notification sound (optional, fails silently)
play_sound() {
    local sound="$1"
    paplay "/usr/share/sounds/freedesktop/stereo/${sound}.oga" 2>/dev/null || true
}

if [ -f "$PID_FILE" ]; then
    # === STOP RECORDING & TRANSCRIBE ===

    # Read the notification ID for updates
    NOTIFY_ID=0
    if [ -f "$NOTIFY_ID_FILE" ]; then
        NOTIFY_ID=$(cat "$NOTIFY_ID_FILE")
    fi

    # Stop recording and wait for arecord to finalize the WAV header
    REC_PID=$(cat "$PID_FILE")
    rm -f "$PID_FILE"
    kill "$REC_PID" 2>/dev/null || true
    # Poll until process exits (max 2s), then force-kill if stuck
    for _ in $(seq 1 40); do
        kill -0 "$REC_PID" 2>/dev/null || break
        sleep 0.05
    done
    kill -9 "$REC_PID" 2>/dev/null || true

    # Update notification to show processing state
    show_notification "Processing..." "Transcribing audio" "$NOTIFY_ID" >/dev/null

    play_sound "message-sent-instant"

    # Send to Telemuze server
    # The /v1/dictate/smart endpoint returns plain text (no JSON)
    FINAL_TEXT=$(curl -s -X POST "$SERVER_URL" \
        -F file="@${AUDIO_FILE}" \
        --max-time 30) || true

    # Clean up audio file
    rm -f "$AUDIO_FILE"

    # Dismiss the notification
    dismiss_notification "$NOTIFY_ID"
    rm -f "$NOTIFY_ID_FILE"

    if [ -n "$FINAL_TEXT" ]; then
        type_text "$FINAL_TEXT"
        play_sound "message-new-instant"
    else
        play_sound "dialog-error"
    fi
else
    # === START RECORDING ===

    # Record in 16kHz mono WAV to skip server-side resampling
    arecord -f S16_LE -c 1 -r 16000 "$AUDIO_FILE" -q &
    echo $! > "$PID_FILE"

    # Show persistent recording indicator
    NOTIFY_ID=$(show_notification "Recording..." "Press hotkey again to stop and transcribe")
    echo "$NOTIFY_ID" > "$NOTIFY_ID_FILE"

    play_sound "bell"
fi
