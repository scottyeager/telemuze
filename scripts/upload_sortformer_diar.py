#!/usr/bin/env -S uv run
# /// script
# dependencies = ["huggingface_hub>=0.24"]
# ///
"""Upload the streaming Sortformer 4-speaker diarization ONNX to Hugging Face."""
from __future__ import annotations

import getpass
from pathlib import Path

from huggingface_hub import HfApi, login

MODEL_FILE = Path.home() / ".local/share/telemuze/models/diar_streaming_sortformer_4spk-v2.1.onnx"
REPO_ID = "scottyeager/diar_streaming_sortformer_4spk-v2.1"
REPO_TYPE = "model"


def main() -> None:
    if not MODEL_FILE.is_file():
        raise SystemExit(f"Model file not found: {MODEL_FILE}")

    size_mb = MODEL_FILE.stat().st_size / (1024 * 1024)
    print(f"Uploading from: {MODEL_FILE.parent}")
    print(f"           to: https://huggingface.co/{REPO_ID}")
    print("Files:")
    print(f"  {MODEL_FILE.name} ({size_mb:.1f} MB)")

    token = getpass.getpass("Hugging Face token (write access): ").strip()
    if not token:
        raise SystemExit("No token provided.")
    login(token=token, add_to_git_credential=False)

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    api.upload_file(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_or_fileobj=str(MODEL_FILE),
        path_in_repo=MODEL_FILE.name,
        commit_message="Upload streaming Sortformer 4spk v2.1 diarization ONNX",
    )
    print(f"Done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
