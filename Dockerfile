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
# Arguments:
#   SHERPA_ONNX_VERSION  Git tag/branch of k2-fsa/sherpa-onnx to build
#                        (default: v1.12.34, matching the Cargo.toml dep)
#   SHERPA_ONNX_REPO     Override the sherpa-onnx repo URL (e.g. to use a fork
#                        with local patches)

ARG UBUNTU_VERSION=22.04

# =====================================================================
# Stage 1: Builder — compiles sherpa-onnx and the telemuze binaries
# =====================================================================
FROM ubuntu:${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        pkg-config \
        libssl-dev \
        libasound2-dev \
        libxcb1-dev \
        libclang-dev \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup (minimal stable toolchain)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal

# ---------------------------------------------------------------------
# Build sherpa-onnx (shared libraries) from source so the .so files
# are linked against the older glibc in this image.
# ---------------------------------------------------------------------
ARG SHERPA_ONNX_VERSION=v1.12.34
ARG SHERPA_ONNX_REPO=https://github.com/k2-fsa/sherpa-onnx.git

RUN git clone --depth 1 --branch ${SHERPA_ONNX_VERSION} \
        ${SHERPA_ONNX_REPO} /tmp/sherpa-onnx

RUN cmake -S /tmp/sherpa-onnx -B /tmp/sherpa-onnx/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DSHERPA_ONNX_ENABLE_C_API=ON \
        -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=ON \
        -DSHERPA_ONNX_ENABLE_BINARY=OFF \
        -DSHERPA_ONNX_ENABLE_TESTS=OFF \
        -DSHERPA_ONNX_ENABLE_TTS=OFF \
        -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
        -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    && cmake --build /tmp/sherpa-onnx/build --parallel "$(nproc)"

# Collect the runtime .so files into a stable location.
# sherpa-onnx downloads onnxruntime during its cmake configure step and places
# it under _deps/onnxruntime-src/lib/. Copy both it and the C-API .so we just
# built into a single directory.
RUN mkdir -p /opt/sherpa-onnx/lib \
    && cp /tmp/sherpa-onnx/build/lib/libsherpa-onnx-c-api.so /opt/sherpa-onnx/lib/ \
    && cp /tmp/sherpa-onnx/build/_deps/onnxruntime-src/lib/libonnxruntime.so /opt/sherpa-onnx/lib/

# sherpa-onnx-sys's build.rs looks for the .so files on LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/sherpa-onnx/lib

# ---------------------------------------------------------------------
# Build telemuze (server, dist launcher, and client)
# ---------------------------------------------------------------------
WORKDIR /src
COPY . /src/

# Server — links against /opt/sherpa-onnx/lib via LD_LIBRARY_PATH above.
# The checked-in .cargo/config.toml sets rpath=$ORIGIN, so the resulting
# binary finds its .so files next to itself at runtime (which matches the
# dist-extracted layout).
RUN cargo build --release --bin telemuze

# Self-contained launcher that embeds the server + .so files
RUN TELEMUZE_SERVER_BIN=/src/target/release/telemuze \
    TELEMUZE_SHERPA_SO=/opt/sherpa-onnx/lib/libsherpa-onnx-c-api.so \
    TELEMUZE_ONNXRUNTIME_SO=/opt/sherpa-onnx/lib/libonnxruntime.so \
    cargo build --release -p telemuze-dist

# Client (separate workspace, static sherpa-onnx via the crate's build.rs)
RUN cd /src/client && cargo build --release --no-default-features \
        --features sherpa-onnx/static --bin telemuze-listen \
        || cd /src/client && cargo build --release --bin telemuze-listen

# Collect output artifacts in /artifacts for easy extraction
RUN mkdir -p /artifacts \
    && cp /src/target/release/telemuze-dist /artifacts/telemuze \
    && cp /src/target/release/telemuze /artifacts/telemuze-server \
    && cp /opt/sherpa-onnx/lib/libsherpa-onnx-c-api.so /artifacts/ \
    && cp /opt/sherpa-onnx/lib/libonnxruntime.so /artifacts/ \
    && (cp /src/client/target/release/telemuze-listen /artifacts/ 2>/dev/null || true) \
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

ENV TELEMUZE_HOST=0.0.0.0 \
    TELEMUZE_PORT=7313

# Models and runtime state extracted by the launcher
VOLUME ["/root/.local/share/telemuze"]

EXPOSE 7313

ENTRYPOINT ["/usr/local/bin/telemuze"]
