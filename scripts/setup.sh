#!/bin/bash
# Quantization Utils Setup Script for Unix-like systems
# Supports Linux and macOS with automatic environment detection

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "Conda not found. Please install Miniconda or Anaconda first."
        echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    log_success "Conda found: $(conda --version)"
}

# Initialize conda for shell
init_conda() {
    log_info "Initializing conda for current shell..."
    eval "$(conda shell.bash hook)"
}

# Create conda environment
create_environment() {
    log_info "Creating conda environment from environment.yml..."
    
    if conda env list | grep -q "quantization-utils"; then
        log_warning "Environment 'quantization-utils' already exists."
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n quantization-utils
        else
            log_info "Using existing environment."
            return 0
        fi
    fi
    
    conda env create -f environment.yml
    log_success "Conda environment created successfully!"
}

# Activate environment and run setup
run_setup() {
    log_info "Activating environment and running setup..."
    
    # Activate the environment
    conda activate quantization-utils
    
    # Run the Python setup script
    python setup.py
    
    log_success "Setup completed!"
}

# Add PATH to shell profile
update_path() {
    local shell_profile=""
    local shell_name=$(basename "$SHELL")
    
    case $shell_name in
        bash)
            shell_profile="$HOME/.bashrc"
            ;;
        zsh)
            shell_profile="$HOME/.zshrc"
            ;;
        fish)
            shell_profile="$HOME/.config/fish/config.fish"
            ;;
        *)
            log_warning "Unknown shell: $shell_name. Please manually add ~/.local/bin to your PATH."
            return
            ;;
    esac
    
    if [[ ! -f "$shell_profile" ]]; then
        touch "$shell_profile"
    fi
    
    # Check if PATH is already updated
    if grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$shell_profile"; then
        log_info "PATH already includes ~/.local/bin"
    else
        echo '' >> "$shell_profile"
        echo '# Added by quantization-utils setup' >> "$shell_profile"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$shell_profile"
        log_success "Added ~/.local/bin to PATH in $shell_profile"
        log_warning "Please restart your shell or run: source $shell_profile"
    fi
}

# Detect GPU capabilities
detect_gpu() {
    log_info "Detecting GPU capabilities..."
    
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU detected!"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    elif [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
        log_success "Apple Silicon detected! Metal acceleration available."
    else
        log_info "No GPU acceleration detected. Using CPU mode."
    fi
}

# Main execution
main() {
    echo "🚀 Quantization Utils Bare Metal Setup"
    echo "======================================"
    
    # Check prerequisites
    check_conda
    detect_gpu
    
    # Setup process
    init_conda
    create_environment
    run_setup
    update_path
    
    echo ""
    echo "✅ Setup completed successfully!"
    echo ""
    echo "📖 Quick Start:"
    echo "   conda activate quantization-utils"
    echo "   cd GGUF"
    echo "   python pipeline.py --model_id microsoft/DialoGPT-medium"
    echo ""
    echo "📚 For more information, check README.md"
}

# Run main function
main "$@" 