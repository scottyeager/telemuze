# We want to cache the model(s) inside the Docker image, so that we can cache
# them on the node. See here for available models and stats: https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages


import whisperx

model_dir = "/models"

models = ["tiny", "turbo"]
device = "cpu"
compute_type = "int8"  # CPU-friendly

for model in models:
    model = whisperx.load_model(
        model, device, compute_type=compute_type, download_root=model_dir
    )
