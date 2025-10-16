# Listener Dockerfile
# Replace this with your actual zinit+sshd Ubuntu base image.
FROM ghcr.io/scottyeager/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# System deps: python, pip, ssh client, certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY listener/requirements.txt /opt/telemuze/

WORKDIR /opt/telemuze

RUN uv venv /.venv

# Python deps (listener)
RUN uv pip install --no-cache-dir --break-system-packages -r requirements.txt

RUN  wget https://github.com/threefoldtech/tfgrid-sdk-go/releases/download/v0.16.11/tfgrid-sdk-go_Linux_x86_64.tar.gz && \
    tar -xzf tfgrid-sdk-go_Linux_x86_64.tar.gz -C /usr/local/bin tfcmd && \
    chmod +x /usr/local/bin/tfcmd && \
    rm tfgrid-sdk-go_Linux_x86_64.tar.gz

# Copy listener code
COPY main.py /opt/telemuze/

COPY listener/zinit/* /etc/zinit

# Create state dir for SSH keys/db (owned by root by default; adjust if you use a non-root user)
RUN mkdir -p /root/.telemuze /tmp/telemuze && chmod 700 /root/.telemuze
