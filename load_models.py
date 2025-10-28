# We want to cache the model(s) inside the Docker image, so that we can cache
# them on the node. See here for available models and stats: https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
# This script can also be used to warm the cache after deployment, by forcing
# the node to load the model and code. We only load the tiny model for cache
# warming, since the listener performs cache warming but might not have enough
# RAM to load turbo. We warm the turbo model just by reading the first byte,
# since this will cause the node to download the full file.


import whisperx

model_dir = "/models"
device = "cpu"
compute_type = "int8"  # CPU-friendly

models = ["tiny", "turbo"]

for model in models:
    model = whisperx.load_model(
        model, device, compute_type=compute_type, download_root=model_dir
    )
