# Portable build environment for telemuze.
#
# Builds on Ubuntu 22.04 to target glibc 2.35 (supports Ubuntu 22.04+,
# Debian 12+, RHEL 9+, and most modern distros from ~2022 onward).
#
# Usage:
#   # Build everything and extract binaries
#   docker build -t telemuze:build --target builder .
#   docker run --rm -v "$PWD/dist-out:/out" telemuze:build cp -r /artifacts/. /out/
#
#   # Build a runnable image
#   docker build -t telemuze:latest .
#   docker run -p 7313:7313 telemuze:latest
#
# sherpa-onnx binaries come from the prebuilt release archives of our fork
# (scottyeager/sherpa-onnx); sherpa-onnx-sys's build.rs downloads them
# automatically. Version is pinned via the git tag in Cargo.toml.

ARG UBUNTU_VERSION=22.04

# =====================================================================
# Stage 1: Builder — compiles the telemuze binaries
# =====================================================================
FROM ubuntu:${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        pkg-config \
        libssl-dev \
        libasound2-dev \
        libxcb1-dev \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup (minimal stable toolchain)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal

# ---------------------------------------------------------------------
# Build telemuze (server, dist launcher, and client). sherpa-onnx-sys's
# build.rs downloads prebuilt shared/static archives from the fork's
# GitHub release and copies the runtime .so files into target/release/.
# ---------------------------------------------------------------------
WORKDIR /src
COPY . /src/

RUN cargo build --release --bin telemuze

# Self-contained launcher that embeds the server + .so files
RUN TELEMUZE_SERVER_BIN=/src/target/release/telemuze \
    TELEMUZE_SHERPA_SO=/src/target/release/libsherpa-onnx-c-api.so \
    TELEMUZE_ONNXRUNTIME_SO=/src/target/release/libonnxruntime.so \
    cargo build --release -p telemuze-dist

# Client (separate workspace, static sherpa-onnx via the crate's build.rs)
RUN cd /src/client && cargo build --release --bin telemuze-listen

# Collect output artifacts in /artifacts for easy extraction
RUN mkdir -p /artifacts \
    && cp /src/target/release/telemuze-dist /artifacts/telemuze \
    && cp /src/target/release/telemuze /artifacts/telemuze-server \
    && cp /src/target/release/libsherpa-onnx-c-api.so /artifacts/ \
    && cp /src/target/release/libonnxruntime.so /artifacts/ \
    && cp /src/client/target/release/telemuze-listen /artifacts/ \
    && ls -lh /artifacts/

# =====================================================================
# Stage 2: Runtime — minimal image that runs the self-contained binary
# =====================================================================
FROM ubuntu:${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime libraries: ffmpeg (audio transcoding), openssl3 (reqwest),
# libgomp1 (onnxruntime OpenMP), libstdc++6 (sherpa-onnx C++ stdlib).
# These exist by default on most modern distros; for a true single-file
# distribution, use the telemuze binary directly on the host.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        libssl3 \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /artifacts/telemuze /usr/local/bin/telemuze

ENV TELEMUZE_PORT=7313

# Models and runtime state extracted by the launcher
VOLUME ["/root/.local/share/telemuze"]

EXPOSE 7313

ENTRYPOINT ["/usr/local/bin/telemuze"]
