# We want to cache the model(s) inside the Docker image, so that they're there
# immediately when we deploy the VM
# See here for available models and stats:
# https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages

import whisperx

models = ["tiny"]
device = "cpu"
compute_type = "int8"  # CPU-friendly
model_dir = "/models"

for model in models:
    model = whisperx.load_model(
        model, device, compute_type=compute_type, download_root=model_dir
    )
