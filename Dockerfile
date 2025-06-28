# Multi-stage Dockerfile for quantization-utils
# Supports CPU, CUDA, and Apple Silicon environments

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

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
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make sure GGUF directory has proper permissions
RUN chmod -R 755 GGUF/

# CPU-only build target
FROM base as cpu
ENV CMAKE_ARGS=""
ENV LLAMA_CPP_CMAKE_ARGS=""
WORKDIR /app/GGUF
CMD ["python3", "pipeline.py", "--help"]

# CUDA build target
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as cuda-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# CUDA target
FROM cuda-base as cuda
ENV CMAKE_ARGS="-DGGML_CUDA=ON"
ENV LLAMA_CPP_CMAKE_ARGS="-DGGML_CUDA=ON"
ENV CUDA_VISIBLE_DEVICES=0
WORKDIR /app/GGUF
CMD ["python3", "pipeline.py", "--help"]

# Apple Silicon / Metal target (for local builds)
FROM base as metal
ENV CMAKE_ARGS="-DGGML_METAL=ON"
ENV LLAMA_CPP_CMAKE_ARGS="-DGGML_METAL=ON"
WORKDIR /app/GGUF
CMD ["python3", "pipeline.py", "--help"] 