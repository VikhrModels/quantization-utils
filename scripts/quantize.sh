#!/bin/bash
# Quick Quantization Script
# Usage: ./scripts/quantize.sh MODEL_ID [OPTIONS]

set -e

# Default values
MODEL_ID=""
QUANTS=""
FORCE=""
HF_TOKEN=""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Helper functions
usage() {
    echo "Usage: $0 MODEL_ID [OPTIONS]"
    echo ""
    echo "Quick quantization script for GGUF models"
    echo ""
    echo "Arguments:"
    echo "  MODEL_ID          HuggingFace model ID (required)"
    echo ""
    echo "Options:"
    echo "  -q, --quants      Comma-separated quantization levels"
    echo "  -f, --force       Force re-quantization of existing models"
    echo "  --token           HuggingFace token for gated models"
    echo "  -h, --help        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 microsoft/DialoGPT-medium"
    echo "  $0 Vikhrmodels/Vikhr-Gemma-2B-instruct -q Q4_K_M,Q8_0 -f"
    echo "  $0 meta-llama/Llama-2-7b-hf --token \$HF_TOKEN"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -q|--quants)
            QUANTS="$2"
            shift 2
            ;;
        -f|--force)
            FORCE="--force"
            shift
            ;;
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            if [[ -z "$MODEL_ID" ]]; then
                MODEL_ID="$1"
            else
                echo "Error: Multiple model IDs specified"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_ID" ]]; then
    echo "Error: MODEL_ID is required"
    usage
    exit 1
fi

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "quantization-utils" ]]; then
    echo -e "${YELLOW}Warning: quantization-utils conda environment not activated${NC}"
    echo "Run: conda activate quantization-utils"
    echo "Or use the full setup script: ./scripts/setup.sh"
    exit 1
fi

# Change to GGUF directory
cd GGUF

echo -e "${BLUE}🚀 Starting quantization for: ${MODEL_ID}${NC}"
echo -e "${BLUE}📊 Quantization levels: ${QUANTS}${NC}"

# Convert comma-separated quants to array
IFS=',' read -ra QUANT_ARRAY <<< "$QUANTS"

# Build command
CMD="python pipeline.py --model_id \"$MODEL_ID\""

# Add quantization levels
if [[ ! -z "$QUANTS" ]]; then
    for quant in "${QUANT_ARRAY[@]}"; do
        CMD="$CMD -q $quant"
    done
fi

# Add optional parameters
if [[ ! -z "$FORCE" ]]; then
    CMD="$CMD $FORCE"
fi

if [[ ! -z "$HF_TOKEN" ]]; then
    CMD="$CMD --hf_token \"$HF_TOKEN\""
fi

echo -e "${BLUE}🔧 Running: $CMD${NC}"
echo ""

# Execute the command
eval $CMD

echo ""
echo -e "${GREEN}✅ Quantization completed!${NC}"
echo -e "${GREEN}📁 Check the output in: GGUF/models/${MODEL_ID}-GGUF/${NC}" 