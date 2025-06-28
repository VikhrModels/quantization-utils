# Metal Dockerfile for GGUF quantization
# Optimized for Apple Silicon with Metal acceleration

FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Build arguments
ARG THREADS=8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    pkg-config \
    libopenblas-dev \
    libgomp1 \
    jq \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Function to detect architecture and download latest llama.cpp release
# Note: For Metal/Apple Silicon, we'll try macOS binaries first, then fall back to Ubuntu
RUN set -e && \
    ARCH=$(uname -m) && \
    case $ARCH in \
        x86_64) LLAMA_ARCH="x64"; OS_TYPE="ubuntu" ;; \
        aarch64|arm64) LLAMA_ARCH="arm64"; OS_TYPE="macos" ;; \
        armv7l) LLAMA_ARCH="armv7l"; OS_TYPE="ubuntu" ;; \
        *) echo "Unsupported architecture: $ARCH" && exit 1 ;; \
    esac && \
    echo "Detected architecture: $ARCH -> llama.cpp: $LLAMA_ARCH" && \
    LATEST_RELEASE=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | jq -r .tag_name) && \
    echo "Latest llama.cpp release: $LATEST_RELEASE" && \
    if [ "$OS_TYPE" = "macos" ]; then \
        DOWNLOAD_URL="https://github.com/ggml-org/llama.cpp/releases/download/$LATEST_RELEASE/llama-$LATEST_RELEASE-bin-macos-$LLAMA_ARCH.zip"; \
    else \
        DOWNLOAD_URL="https://github.com/ggml-org/llama.cpp/releases/download/$LATEST_RELEASE/llama-$LATEST_RELEASE-bin-ubuntu-$LLAMA_ARCH.zip"; \
    fi && \
    echo "Downloading from: $DOWNLOAD_URL" && \
    curl -L -o llama.cpp.zip "$DOWNLOAD_URL" && \
    unzip llama.cpp.zip && \
    rm llama.cpp.zip && \
    find . -name "llama-*-bin-*" -type d | head -1 | xargs -I{} mv {} /opt/llama.cpp && \
    chmod +x /opt/llama.cpp/bin/* && \
    ln -sf /opt/llama.cpp/bin/* /usr/local/bin/ || \
    echo "Pre-built binaries not available, will build from source when needed"

# Add llama.cpp binaries to PATH
ENV PATH="/opt/llama.cpp/bin:$PATH"

# Copy requirements from parent directory and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the GGUF directory
COPY . ./GGUF/

# Set working directory to GGUF
WORKDIR /app/GGUF

# Set build environment for Metal (Apple Silicon)
ENV CMAKE_ARGS="-DGGML_METAL=ON -DGGML_METAL_NDEBUG=ON"
ENV LLAMA_CPP_CMAKE_ARGS="-DGGML_METAL=ON -DGGML_METAL_NDEBUG=ON"
ENV THREADS=${THREADS}

# Create necessary directories
RUN mkdir -p models imatrix wikitext-2-raw

# Default command
CMD ["/bin/bash"] 