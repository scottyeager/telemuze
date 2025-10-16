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

# Python deps: PyTorch CPU wheels, WhisperX, and friends
# Important: install torch/torchaudio from the PyTorch CPU index first
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.* torchvision==0.18.* torchaudio==2.3.*

RUN pip3 install --no-cache-dir \
    whisperx==3.1.* \
    numpy \
    orjson==3.* \
    tqdm

# Runtime directories
RUN mkdir -p /job/input /job/output /job/logs /models /opt/telemuze

# Copy SSH-invoked scripts
COPY composer/transcribe.sh /usr/local/bin/transcribe.sh
COPY composer/run_transcribe.py /opt/telemuze/run_transcribe.py
RUN chmod +x /usr/local/bin/transcribe.sh && chown -R transcriber:transcriber /opt/telemuze

# No ENTRYPOINT/CMD. Listener will SSH in and run:
# /usr/local/bin/transcribe.sh --in <remote_input_path> --model <model> --language <lang>
