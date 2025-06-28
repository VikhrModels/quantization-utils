#!/bin/bash
# Perplexity Testing Script
# Usage: ./scripts/perplexity.sh MODEL_ID [OPTIONS]

set -e

# Default values
MODEL_ID=""
FORCE=""
DATASET="wikitext"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Helper functions
usage() {
    echo "Usage: $0 MODEL_ID [OPTIONS]"
    echo ""
    echo "Run perplexity tests on quantized GGUF models"
    echo ""
    echo "Arguments:"
    echo "  MODEL_ID          HuggingFace model ID (required)"
    echo ""
    echo "Options:"
    echo "  -f, --force       Force re-calculation of perplexity"
    echo "  -d, --dataset     Dataset to use (default: wikitext)"
    echo "  -h, --help        Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 microsoft/DialoGPT-medium"
    echo "  $0 Vikhrmodels/Vikhr-Gemma-2B-instruct --force"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -f|--force)
            FORCE="--force-perplexity"
            shift
            ;;
        -d|--dataset)
            DATASET="$2"
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
    exit 1
fi

# Check if model directory exists
MODEL_DIR="GGUF/models/${MODEL_ID}-GGUF"
if [[ ! -d "$MODEL_DIR" ]]; then
    echo -e "${YELLOW}Warning: Model directory not found: $MODEL_DIR${NC}"
    echo "Please quantize the model first:"
    echo "  ./scripts/quantize.sh $MODEL_ID"
    exit 1
fi

# Count available GGUF files
GGUF_COUNT=$(find "$MODEL_DIR" -name "*.gguf" | wc -l)
if [[ $GGUF_COUNT -eq 0 ]]; then
    echo -e "${YELLOW}Warning: No GGUF files found in $MODEL_DIR${NC}"
    echo "Please quantize the model first:"
    echo "  ./scripts/quantize.sh $MODEL_ID"
    exit 1
fi

echo -e "${BLUE}🧪 Running perplexity tests for: ${MODEL_ID}${NC}"
echo -e "${BLUE}📊 Found ${GGUF_COUNT} GGUF files to test${NC}"
echo -e "${BLUE}📖 Using dataset: ${DATASET}${NC}"

# Change to GGUF directory
cd GGUF

# Build command
CMD="python pipeline.py --model_id \"$MODEL_ID\" --perplexity"

# Add optional parameters
if [[ ! -z "$FORCE" ]]; then
    CMD="$CMD $FORCE"
fi

echo -e "${BLUE}🔧 Running: $CMD${NC}"
echo ""

# Execute the command
eval $CMD

echo ""
echo -e "${GREEN}✅ Perplexity testing completed!${NC}"
echo -e "${GREEN}📁 Check results in: GGUF/models/${MODEL_ID}-GGUF/perplexity/${NC}"

# Show summary if results exist
RESULTS_DIR="models/${MODEL_ID}-GGUF/perplexity"
if [[ -d "$RESULTS_DIR" ]]; then
    echo ""
    echo -e "${BLUE}📈 Perplexity Summary:${NC}"
    find "$RESULTS_DIR" -name "*.txt" | while read -r file; do
        if [[ -f "$file" ]]; then
            echo "  $(basename "$file")"
        fi
    done
fi 