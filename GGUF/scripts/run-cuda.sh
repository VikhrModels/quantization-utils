#!/bin/bash

# Simple one-off CUDA runner for GGUF quantization
# Usage: ./run-cuda.sh [pipeline_args...]

set -e

IMAGE_NAME="gguf-quantization-cuda"
CONTAINER_NAME="gguf-cuda"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to detect CUDA architecture
detect_cuda_arch() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [[ -n "$gpu_info" ]]; then
            local major=$(echo $gpu_info | cut -d. -f1)
            local minor=$(echo $gpu_info | cut -d. -f2)
            echo "${major}${minor}"
        else
            echo "75;80;86;89;90"
        fi
    else
        echo "75;80;86;89;90"
    fi
}

# Check if image exists, build if not
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    echo -e "${YELLOW}Building GGUF CUDA image...${NC}"
    local cuda_arch="${CUDA_ARCHITECTURES:-$(detect_cuda_arch)}"
    echo "CUDA architectures: $cuda_arch"
    docker build -f Dockerfile.cuda -t $IMAGE_NAME \
        --build-arg CUDA_ARCHITECTURES="$cuda_arch" \
        --build-arg THREADS=$(nproc 2>/dev/null || echo 8) \
        .
    echo -e "${GREEN}Build completed!${NC}"
fi

# Create directories and run
echo -e "${YELLOW}Running GGUF quantization (CUDA)...${NC}"
mkdir -p models output .cache

# Show GPU info
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
fi

docker run --rm \
    --gpus all \
    --name $CONTAINER_NAME \
    -v "$(pwd)/models:/app/GGUF/models" \
    -v "$(pwd)/output:/app/GGUF/output" \
    -v "$(pwd)/.cache:/root/.cache" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e THREADS=$(nproc 2>/dev/null || echo 8) \
    $IMAGE_NAME \
    python3 pipeline.py "$@" 