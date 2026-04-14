#!/usr/bin/env -S uv run
# /// script
# dependencies = ["huggingface_hub>=0.24"]
# ///
"""Upload the int8 parakeet-tdt-transducer-110m model to Hugging Face."""
from __future__ import annotations

import getpass
from pathlib import Path

from huggingface_hub import HfApi, login

MODEL_DIR = Path.home() / ".local/share/telemuze/models/sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8"
REPO_ID = "scottyeager/sherpa-onnx-nemo-parakeet-tdt-transducer-110m-en-int8"
REPO_TYPE = "model"


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"Model directory not found: {MODEL_DIR}")

    print(f"Uploading: {MODEL_DIR}")
    print(f"      to: https://huggingface.co/{REPO_ID}")
    token = getpass.getpass("Hugging Face token (write access): ").strip()
    if not token:
        raise SystemExit("No token provided.")
    login(token=token, add_to_git_credential=False)

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(MODEL_DIR),
        commit_message="Upload sherpa-onnx int8 parakeet-tdt-transducer-110m",
    )
    print(f"Done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
