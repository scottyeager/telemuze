# In this stage we download python code and models then put them into a tar
# archive to copy to the second stage. This supports our caching strategy by
# having a single file for the cache warmer to touch
FROM ghcr.io/scottyeager/ubuntu:24.04 AS dependencies

# We don't actually need python in this stage, but installing it will ensure
# that uv targets the right version and bin paths
RUN apt update && apt install -y python3

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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

COPY load_models.py /

# Download the models
RUN mkdir -p /models
RUN uv run /load_models.py

# We don't compress because it's a lot of data and we want to be able to extract
# it fast when we start a composer
RUN tar -cf dependencies.tar .venv /models

# Combined Dockerfile for Telemuze Composer and Listener
FROM ghcr.io/scottyeager/ubuntu:24.04

COPY --from=dependencies dependencies.tar /

# System deps: python, pip, ffmpeg, libsndfile for torchaudio, ssh client, certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
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

# Copy python code and zinit configs
COPY composer.py /opt/telemuze/
COPY listener.py /opt/telemuze/
COPY zinit/* /etc/zinit/

# Copy and make scripts executable
COPY scripts/start-bot-api.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-bot-api.sh

# Create remaining dirs here, to avoid busting cache if we need to add some
RUN mkdir -p /job/input \
    /job/output \
    /job/logs \
    /opt/telegram-bot-api \
    /root/.telemuze \
    /tmp/telemuze
RUN chmod 700 /root/.telemuze
