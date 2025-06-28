#!/bin/bash

# Simple one-off CPU runner for GGUF quantization
# Usage: ./run-cpu.sh [pipeline_args...]

set -e

IMAGE_NAME="gguf-quantization-cpu"
CONTAINER_NAME="gguf-cpu"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if image exists, build if not
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    echo -e "${YELLOW}Building GGUF CPU image...${NC}"
    docker build -f Dockerfile.cpu -t $IMAGE_NAME \
        --build-arg THREADS=$(nproc 2>/dev/null || echo 8) \
        .
    echo -e "${GREEN}Build completed!${NC}"
fi

# Create directories and run
echo -e "${YELLOW}Running GGUF quantization (CPU)...${NC}"
mkdir -p models output .cache

docker run --rm \
    --name $CONTAINER_NAME \
    -v "$(pwd)/models:/app/GGUF/models" \
    -v "$(pwd)/output:/app/GGUF/output" \
    -v "$(pwd)/.cache:/root/.cache" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e THREADS=$(nproc 2>/dev/null || echo 8) \
    $IMAGE_NAME \
    python3 pipeline.py "$@" 