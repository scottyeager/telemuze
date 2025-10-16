#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid

LOG_DIR = "/job/logs"
OUT_ROOT = "/job/output"
IN_ROOT = "/job/input"
MODEL_DIR = "/models"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_to_log(cmd, log_fh, timeout=None):
    """Run a command, streaming stdout+stderr to log file (no memory buffering)."""
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")


def ffprobe_duration_seconds(path):
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        return float(out.decode().strip())
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_path", required=True, help="Input media file path on remote"
    )
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--ffmpeg-timeout-sec", type=int, default=20 * 60)
    ap.add_argument(
        "--job-id",
        default=os.environ.get("JOB_ID")
        or str(int(time.time() * 1000)) + "-" + uuid.uuid4().hex[:6],
    )
    args = ap.parse_args()

    in_path = args.in_path
    if not os.path.isfile(in_path):
        print(json.dumps({"ok": False, "error": f"E_INPUT: file not found: {in_path}"}))
        sys.exit(2)

    job_id = args.job_id
    # Derive dirs
    out_dir = os.path.join(OUT_ROOT, job_id)
    in_dir = os.path.join(IN_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(in_dir, exist_ok=True)

    basename = pathlib.Path(in_path).name
    wav_path = os.path.join(in_dir, f"{basename.rsplit('.', 1)[0]}.wav")
    txt_path = os.path.join(out_dir, "transcript.txt")
    log_path = os.path.join(LOG_DIR, f"{job_id}.log")

    try:
        with open(log_path, "a", encoding="utf-8") as log_fh:
            log_fh.write(
                f"[info] job_id={job_id} model={args.model} language={args.language}\n"
            )
            log_fh.flush()

            # Convert to WAV 16k mono s16le
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                in_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                wav_path,
            ]
            run_to_log(ffmpeg_cmd, log_fh, timeout=args.ffmpeg_timeout_sec)

            # Transcribe with WhisperX
            # Import here to get logs in file if import fails
            import whisperx

            device = "cpu"
            compute_type = "int8"  # CPU-friendly
            log_fh.write(
                f"[info] loading whisperx model={args.model} device={device} compute_type={compute_type}\n"
            )
            log_fh.flush()
            model = whisperx.load_model(
                args.model,
                device=device,
                compute_type=compute_type,
                download_root=MODEL_DIR,
            )

            audio = whisperx.load_audio(wav_path)
            lang = None if args.language == "auto" else args.language
            result = model.transcribe(audio, language=lang)

            # TODO: we can get start/end timestamps and other goodies from segments
            # result: {'segments': [{'text': "...", 'start': 0.031, 'end': 2.107}], 'language': 'en'}
            text = "\n".join(
                s["text"].strip() for s in result.get("segments", []) if "text" in s
            )

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

        # Build final JSON result
        dur = ffprobe_duration_seconds(wav_path)
        chars = 0
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                chars = len(f.read())
        except Exception:
            pass

        print(
            json.dumps(
                {
                    "ok": True,
                    "language": args.language,
                    "duration_sec": dur,
                    "text_path": txt_path,
                    "chars": chars,
                }
            )
        )
        sys.exit(0)

    except subprocess.TimeoutExpired:
        print(
            json.dumps({"ok": False, "error": "E_FFMPEG_TIMEOUT: conversion timed out"})
        )
        sys.exit(124)
    except Exception as e:
        eprint(f"E_WHISPERX: {e}")
        print(json.dumps({"ok": False, "error": f"E_WHISPERX: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
