# Combined Dockerfile for Telemuze Composer and Listener
FROM ghcr.io/scottyeager/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    XDG_CACHE_HOME=/models

# System deps: python, pip, ffmpeg, libsndfile for torchaudio, ssh client, certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    ffmpeg \
    libsndfile1 \
    openssh-client \
    ca-certificates \
    libc++1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Download and install tfcmd
RUN  wget https://github.com/threefoldtech/tfgrid-sdk-go/releases/download/v0.16.11/tfgrid-sdk-go_Linux_x86_64.tar.gz && \
    tar -xzf tfgrid-sdk-go_Linux_x86_64.tar.gz -C /usr/local/bin tfcmd && \
    chmod +x /usr/local/bin/tfcmd && \
    rm tfgrid-sdk-go_Linux_x86_64.tar.gz

# Download and install Telegram Bot API binary
RUN wget -O /usr/local/bin/telegram-bot-api https://github.com/scottyeager/Telegram-Bot-API-Builder/releases/latest/download/telegram-bot-api && \
    chmod +x /usr/local/bin/telegram-bot-api

# Create a single virtual environment
RUN uv venv /.venv

# Python deps: PyTorch CPU wheels, WhisperX, and friends for the composer
# Important: install torch/torchaudio from the PyTorch CPU index first
RUN uv pip install \
    --index https://download.pytorch.org/whl/cpu \
    torchaudio~=2.8.0 whisperx

# Python deps for the listener
COPY requirements.txt /opt/telemuze/requirements.txt
RUN uv pip install --no-cache-dir -r /opt/telemuze/requirements.txt

# Runtime directories
RUN mkdir -p /job/input /job/output /job/logs /models /opt/telemuze

# Copy composer files
COPY load_models.py /opt/telemuze/

# Cache models so they are there immediately on deployment
RUN uv run /opt/telemuze/load_models.py

# Copy listener code and zinit configs
COPY composer.py /opt/telemuze/
COPY listener.py /opt/telemuze/
COPY zinit/* /etc/zinit/
COPY scripts/start-bot-api.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-bot-api.sh

# Create state dir for SSH keys/db
RUN mkdir -p /root/.telemuze /tmp/telemuze && chmod 700 /root/.telemuze
