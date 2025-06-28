# Quantization Utils - Bare Metal Setup

A comprehensive toolkit for quantizing large language models to GGUF format with support for multiple acceleration backends (CUDA, Metal, CPU).

## 🚀 Features

| Feature             | Status | Description                                       |
| ------------------- | ------ | ------------------------------------------------- |
| 🖥️ **Bare Metal**    | ✅      | Native installation without Docker                |
| 🔧 **Auto Setup**    | ✅      | Automatic environment detection and configuration |
| 🎯 **Multi-Backend** | ✅      | CUDA, Metal (Apple Silicon), and CPU support      |
| 📦 **Conda Ready**   | ✅      | Complete conda environment with all dependencies  |
| ⚡ **Quick Scripts** | ✅      | Convenient scripts for common tasks               |
| 📊 **Perplexity**    | ✅      | Automated quality testing of quantized models     |
| 🔍 **Validation**    | ✅      | Environment health checks and troubleshooting     |

## 📋 Prerequisites

| Requirement | Minimum Version | Notes                     |
| ----------- | --------------- | ------------------------- |
| **Conda**   | Latest          | Miniconda or Anaconda     |
| **Python**  | 3.11+           | Installed via conda       |
| **Git**     | 2.0+            | For repository operations |
| **CMake**   | 3.14+           | For building llama.cpp    |

### GPU Support (Optional)

| Platform          | Requirements     | Acceleration               |
| ----------------- | ---------------- | -------------------------- |
| **NVIDIA**        | CUDA 11.8+       | ✅ CUDA acceleration        |
| **Apple Silicon** | macOS + M1/M2/M3 | ✅ Metal acceleration       |
| **Others**        | Any CPU          | ✅ Optimized CPU processing |

## 🛠️ Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/Vikhrmodels/quantization-utils.git
cd quantization-utils

# Run the automated setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Option 2: Manual Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate quantization-utils

# Run setup to install llama.cpp and prepare directories
python setup.py

# Add to PATH (if needed)
export PATH="$HOME/.local/bin:$PATH"
```

## 🔍 Validation

Verify your installation:

```bash
# Check environment health
./scripts/validate.sh

# Quick test
conda activate quantization-utils
cd GGUF
python -c "from shared import validate_environment; validate_environment()"
```

## 📊 Usage Examples

### Basic Model Quantization

```bash
# Activate environment
conda activate quantization-utils

# Quantize a model with default settings
./scripts/quantize.sh microsoft/DialoGPT-medium

# Custom quantization levels
./scripts/quantize.sh Vikhrmodels/Vikhr-Gemma-2B-instruct -q Q4_K_M,Q5_K_M,Q8_0

# Force re-quantization
./scripts/quantize.sh microsoft/DialoGPT-medium --force
```

### Advanced Pipeline Usage

```bash
cd GGUF

# Full pipeline with all quantization levels
python pipeline.py --model_id microsoft/DialoGPT-medium

# Specific quantization levels only
python pipeline.py --model_id microsoft/DialoGPT-medium -q Q4_K_M -q Q8_0

# With perplexity testing
python pipeline.py --model_id microsoft/DialoGPT-medium --perplexity

# For gated models (requires HF token)
python pipeline.py --model_id meta-llama/Llama-2-7b-hf --hf_token $HF_TOKEN
```

### Perplexity Testing

```bash
# Test all quantized versions
./scripts/perplexity.sh microsoft/DialoGPT-medium

# Force recalculation
./scripts/perplexity.sh microsoft/DialoGPT-medium --force
```

## 📁 Directory Structure

```
quantization-utils/
├── 📄 environment.yml          # Conda environment definition
├── 🐍 setup.py                 # Environment setup script
├── 📖 README.md                # This file
│
├── 🔧 scripts/                 # Convenience scripts
│   ├── setup.sh               # Automated setup
│   ├── validate.sh             # Environment validation
│   ├── quantize.sh             # Quick quantization
│   └── perplexity.sh           # Perplexity testing
│
└── 📦 GGUF/                    # Main processing directory
    ├── 🐍 pipeline.py          # Main pipeline script
    ├── 🐍 shared.py            # Shared utilities
    ├── 📁 models/              # Downloaded models
    ├── 📁 imatrix/             # Importance matrices
    ├── 📁 output/              # Final quantized models
    ├── 📁 resources/           # Calibration data
    │   └── standard_cal_data/
    └── 📁 modules/             # Processing modules
        ├── convert.py
        ├── quantize.py
        ├── imatrix.py
        └── perplexity.py
```

## ⚙️ Configuration Options

### Environment Variables

| Variable               | Description           | Example  |
| ---------------------- | --------------------- | -------- |
| `HF_TOKEN`             | HuggingFace API token | `hf_...` |
| `CUDA_VISIBLE_DEVICES` | GPU selection         | `0,1`    |
| `OMP_NUM_THREADS`      | CPU threads           | `8`      |

### Pipeline Parameters

| Parameter      | Description          | Default    |
| -------------- | -------------------- | ---------- |
| `--model_id`   | HuggingFace model ID | Required   |
| `--quants`     | Quantization levels  | All levels |
| `--force`      | Force reprocessing   | False      |
| `--perplexity` | Run quality tests    | False      |
| `--threads`    | Processing threads   | CPU count  |

### Quantization Levels

| Level    | Description        | Size     | Quality       |
| -------- | ------------------ | -------- | ------------- |
| `Q2_K`   | 2-bit quantization | Smallest | Good          |
| `Q4_K_M` | 4-bit mixed        | Balanced | Very Good     |
| `Q5_K_M` | 5-bit mixed        | Larger   | Excellent     |
| `Q6_K`   | 6-bit              | Large    | Near Original |
| `Q8_0`   | 8-bit              | Largest  | Original      |

## 🐛 Troubleshooting

### Common Issues

| Issue                       | Solution                               |
| --------------------------- | -------------------------------------- |
| `conda: command not found`  | Install Miniconda/Anaconda             |
| `llama-quantize: not found` | Run `python setup.py`                  |
| `CUDA out of memory`        | Reduce batch size or use CPU           |
| `Permission denied`         | Check file permissions with `chmod +x` |

### Environment Problems

```bash
# Reset environment
conda env remove -n quantization-utils
conda env create -f environment.yml

# Reinstall llama.cpp
rm -rf ~/.local/bin/llama-*
python setup.py

# Check installation
./scripts/validate.sh
```

### Binary Issues

```bash
# Manual llama.cpp installation
cd /tmp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc)
make install
```

## 🔧 Development

### Adding New Quantization Methods

1. Update `shared.py` with new `Quant` enum values
2. Modify `modules/quantize.py` to handle new methods
3. Update pipeline default quantization list
4. Test with validation scripts

### Custom Calibration Data

```python
# Add to GGUF/resources/standard_cal_data/
# Files should be UTF-8 text with one sample per line
```

## 📈 Performance Tips

| Tip              | Description                                       |
| ---------------- | ------------------------------------------------- |
| 🚀 **GPU Usage**  | Use CUDA/Metal for 5-10x speedup                  |
| 💾 **Memory**     | Monitor RAM usage with large models               |
| 🔄 **Batch Size** | Adjust based on available memory                  |
| 📊 **Threads**    | Set to CPU core count for optimal CPU performance |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `./scripts/validate.sh`
4. Submit a pull request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🔗 Links

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **HuggingFace**: https://huggingface.co/
- **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

---

<div align="center">

**Ready to quantize? Start with `./scripts/setup.sh`** 🚀

</div>