#!/bin/bash
# Environment Validation Script
# Checks if the quantization utils environment is properly set up

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0

# Helper functions
check_passed() {
    echo -e "${GREEN}✅ $1${NC}"
    ((CHECKS_PASSED++))
}

check_failed() {
    echo -e "${RED}❌ $1${NC}"
    ((CHECKS_FAILED++))
}

check_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

check_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Validation functions
check_conda() {
    echo "🔍 Checking Conda installation..."
    if command -v conda &> /dev/null; then
        check_passed "Conda found: $(conda --version)"
    else
        check_failed "Conda not found. Please install Miniconda or Anaconda."
        return 1
    fi
}

check_environment() {
    echo ""
    echo "🔍 Checking conda environment..."
    
    if conda env list | grep -q "quantization-utils"; then
        check_passed "Environment 'quantization-utils' exists"
        
        if [[ "$CONDA_DEFAULT_ENV" == "quantization-utils" ]]; then
            check_passed "Environment 'quantization-utils' is activated"
        else
            check_warning "Environment 'quantization-utils' not activated"
            echo "   Run: conda activate quantization-utils"
        fi
    else
        check_failed "Environment 'quantization-utils' not found"
        echo "   Run: conda env create -f environment.yml"
        return 1
    fi
}

check_python_packages() {
    echo ""
    echo "🔍 Checking Python packages..."
    
    packages=(
        "torch"
        "transformers"
        "datasets"
        "huggingface_hub"
        "numpy"
        "tqdm"
    )
    
    for package in "${packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            check_passed "Python package '$package' available"
        else
            check_failed "Python package '$package' missing"
        fi
    done
}

check_llama_binaries() {
    echo ""
    echo "🔍 Checking llama.cpp binaries..."
    
    binaries=(
        "llama-quantize"
        "llama-imatrix" 
        "llama-perplexity"
        "llama-cli"
    )
    
    for binary in "${binaries[@]}"; do
        if command -v "$binary" &> /dev/null; then
            check_passed "Binary '$binary' found in PATH"
        elif [[ -f "$HOME/.local/bin/$binary" ]]; then
            check_passed "Binary '$binary' found in ~/.local/bin"
        else
            check_failed "Binary '$binary' not found"
        fi
    done
}

check_gpu() {
    echo ""
    echo "🔍 Checking GPU capabilities..."
    
    # Check NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        check_passed "NVIDIA GPU detected: $gpu_name"
        
        # Check CUDA
        if nvidia-smi > /dev/null 2>&1; then
            cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
            check_info "CUDA Version: $cuda_version"
        fi
    elif [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        check_passed "Apple Silicon detected (Metal acceleration available)"
    else
        check_info "No GPU acceleration detected (CPU mode)"
    fi
}

check_directories() {
    echo ""
    echo "🔍 Checking directory structure..."
    
    directories=(
        "GGUF"
        "GGUF/models"
        "GGUF/imatrix"
        "GGUF/output"
        "GGUF/resources/standard_cal_data"
    )
    
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            check_passed "Directory '$dir' exists"
        else
            check_failed "Directory '$dir' missing"
        fi
    done
}

check_calibration_data() {
    echo ""
    echo "🔍 Checking calibration data..."
    
    cal_dir="GGUF/resources/standard_cal_data"
    if [[ -d "$cal_dir" ]]; then
        file_count=$(find "$cal_dir" -type f | wc -l)
        if [[ $file_count -gt 0 ]]; then
            check_passed "Calibration data available ($file_count files)"
        else
            check_warning "Calibration data directory empty"
            echo "   Run: python setup.py to download calibration data"
        fi
    else
        check_failed "Calibration data directory missing"
    fi
}

run_quick_test() {
    echo ""
    echo "🔍 Running quick functionality test..."
    
    if [[ "$CONDA_DEFAULT_ENV" == "quantization-utils" ]]; then
        cd GGUF
        if python -c "from shared import validate_environment, SetupLogger; logger = SetupLogger(); validate_environment(logger)" 2>/dev/null; then
            check_passed "Python modules load correctly"
        else
            check_failed "Python modules have issues"
        fi
        cd ..
    else
        check_warning "Skipping Python test (environment not activated)"
    fi
}

# Main validation
main() {
    echo "🔬 Quantization Utils Environment Validation"
    echo "==========================================="
    
    check_conda
    check_environment  
    check_python_packages
    check_llama_binaries
    check_gpu
    check_directories
    check_calibration_data
    run_quick_test
    
    echo ""
    echo "📊 Validation Summary"
    echo "===================="
    echo -e "✅ Checks passed: ${GREEN}$CHECKS_PASSED${NC}"
    echo -e "❌ Checks failed: ${RED}$CHECKS_FAILED${NC}"
    
    if [[ $CHECKS_FAILED -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}🎉 Environment is ready for quantization!${NC}"
        echo ""
        echo "Quick start:"
        echo "  conda activate quantization-utils"
        echo "  ./scripts/quantize.sh microsoft/DialoGPT-medium"
        exit 0
    else
        echo ""
        echo -e "${RED}⚠️  Some checks failed. Please fix the issues above.${NC}"
        echo ""
        echo "To fix most issues, run:"
        echo "  ./scripts/setup.sh"
        exit 1
    fi
}

# Run validation
main "$@" 