# We want to cache the model(s) inside the Docker image, so that we can cache
# them on the node. See here for available models and stats: https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
# This script can also be used to warm the cache after deployment, by forcing
# the node to load the model and code. We only load the tiny model for cache
# warming, since the listener performs cache warming but might not have enough
# RAM to load turbo. We warm the turbo model just by reading the first byte,
# since this will cause the node to download the full file.

import argparse
import os

import whisperx

model_dir = "/models"
device = "cpu"
compute_type = "int8"  # CPU-friendly

models = ["tiny", "turbo"]

parser = argparse.ArgumentParser()
parser.add_argument("--warm", action="store_true", help="Warm the cache only")
args = parser.parse_args()

if not args.warm:
    for model in models:
        model = whisperx.load_model(
            model, device, compute_type=compute_type, download_root=model_dir
        )
else:
    model = whisperx.load_model(
        "tiny", device, compute_type=compute_type, download_root=model_dir
    )

    # Just check every file, to make sure we hit the turbo model
    for root, _, files in os.walk(model_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                with open(path, "rb") as f:
                    f.read(1)
            except Exception:
                pass
