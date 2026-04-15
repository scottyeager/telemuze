#!/usr/bin/env -S uv run
# /// script
# dependencies = ["huggingface_hub>=0.24"]
# ///
"""Upload the int8 parakeet-unified-en-0.6b ONNX files to Hugging Face.

Pushes only the int8 variant (encoder/decoder/joiner) plus tokens.txt
from the directory produced by `scripts/export_parakeet_unified.py`.
"""
from __future__ import annotations

import getpass
from pathlib import Path

from huggingface_hub import HfApi, login

MODEL_DIR = Path.home() / ".local/share/telemuze/models/parakeet-unified-en-0.6b-int8"
REPO_ID = "scottyeager/parakeet-unified-en-0.6b-int8"
REPO_TYPE = "model"

FILES_TO_UPLOAD = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
]


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"Model directory not found: {MODEL_DIR}")

    missing = [f for f in FILES_TO_UPLOAD if not (MODEL_DIR / f).is_file()]
    if missing:
        raise SystemExit(
            f"Missing files in {MODEL_DIR}: {', '.join(missing)}"
        )

    print(f"Uploading from: {MODEL_DIR}")
    print(f"           to: https://huggingface.co/{REPO_ID}")
    print("Files:")
    for f in FILES_TO_UPLOAD:
        size_mb = (MODEL_DIR / f).stat().st_size / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")

    token = getpass.getpass("Hugging Face token (write access): ").strip()
    if not token:
        raise SystemExit("No token provided.")
    login(token=token, add_to_git_credential=False)

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    for filename in FILES_TO_UPLOAD:
        print(f"Uploading {filename}...")
        api.upload_file(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_or_fileobj=str(MODEL_DIR / filename),
            path_in_repo=filename,
            commit_message=f"Upload {filename}",
        )
    print(f"Done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
