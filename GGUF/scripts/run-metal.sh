#!/bin/bash

# Simple one-off Metal runner for GGUF quantization (Apple Silicon)
# Usage: ./run-metal.sh [pipeline_args...]

set -e

IMAGE_NAME="gguf-quantization-metal"
CONTAINER_NAME="gguf-metal"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if image exists, build if not
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    echo -e "${YELLOW}Building GGUF Metal image...${NC}"
    docker build -f Dockerfile.metal -t $IMAGE_NAME \
        --build-arg THREADS=$(sysctl -n hw.ncpu 2>/dev/null || echo 8) \
        .
    echo -e "${GREEN}Build completed!${NC}"
fi

# Create directories and run
echo -e "${YELLOW}Running GGUF quantization (Metal)...${NC}"
mkdir -p models output .cache

# Show system info
echo "System info:"
echo "Architecture: $(uname -m)"
echo "CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown')"
echo "Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024) " GB"}' || echo 'Unknown')"

docker run --rm \
    --name $CONTAINER_NAME \
    -v "$(pwd)/models:/app/GGUF/models" \
    -v "$(pwd)/output:/app/GGUF/output" \
    -v "$(pwd)/.cache:/root/.cache" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e THREADS=$(sysctl -n hw.ncpu 2>/dev/null || echo 8) \
    $IMAGE_NAME \
    python3 pipeline.py "$@" 