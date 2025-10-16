# Composer Dockerfile
# Replace this with your actual zinit+sshd Ubuntu base image.
FROM ghcr.io/scottyeager/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    XDG_CACHE_HOME=/models

# System deps: python, pip, ffmpeg, libsndfile for torchaudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv venv /.venv

# Python deps: PyTorch CPU wheels, WhisperX, and friends
# Important: install torch/torchaudio from the PyTorch CPU index first
RUN uv pip install \
    --index https://download.pytorch.org/whl/cpu \
    torchaudio~=2.8.0 whisperx

# Runtime directories
RUN mkdir -p /job/input /job/output /job/logs /models /opt/telemuze

COPY composer/load_models.py /opt/telemuze/

# Cache models so they are there immediately on deployment
RUN uv run /opt/telemuze/load_models.py

# Do this last so we don't have to cache models again if we change it
COPY composer/composer.py /opt/telemuze/

# No ENTRYPOINT/CMD. Listener will SSH in and run:
# /usr/local/bin/transcribe.sh --in <remote_input_path> --model <model> --language <lang>
