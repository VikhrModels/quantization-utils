# GGUF Autoconverter

## Overview
The GGUF package provides a pipeline for converting and quantizing Huggingface models to GGUF format for [llama.cpp](https://github.com/ggerganov/llama.cpp) inference engine.


## Running the Pipeline
To run the pipeline, execute the `pipeline.py` script. Below are the steps to set up and run the pipeline:

0. **Consider setting up virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

1. **Install Dependencies**:
```bash
pip install -r GGUF/requirements.txt
```

2. **Run the Pipeline**:
```bash
# Simple example
python GGUF/pipeline.py 
```
```bash
# Advanced example
python GGUF/pipeline.py --model_id Vikhrmodels/it-5.2-fp16-cp --hf_token hf_token_for_gated_and_private_models --quants Q4_K_M,Q6_K,Q8_0
 --quants-skip BF16 --force-imatrix-dataset --force-imatrix-calculation --force-model-convert --force-model-quantization
```
3. **Deactivate virtual environment**:
```bash
deactivate
```

### CLI Help output
 <pre>
usage: pipeline.py [-h] [--model_id MODEL_ID] [--hf_token HF_TOKEN] [--quants QUANTS] [--quants-skip QUANTS_SKIP] [--force] [--force-imatrix-dataset] [--force-imatrix-calculation] [--force-model-convert] [--force-model-quantization]

Run the GGUF Pipeline

options:
  -h, --help            show this help message and exit
  --model_id MODEL_ID   Huggingface model ID
  --hf_token HF_TOKEN   Huggingface token, in case gated model is used
  --quants QUANTS       Quantization levels to run the pipeline on, default is every quantization level
  --quants-skip QUANTS_SKIP
                        Quantization levels to skip for the pipeline
  --force               Force the whole pipeline to run
  --force-imatrix-dataset
                        Force recreation of the imatrix dataset
  --force-imatrix-calculation
                        Force calculation of the imatrix
  --force-model-convert
                        Force conversion of the model
  --force-model-quantization
                        Force quantization of the model
</pre>

## Limitations
- Current working directory is pinned to script's path and is used for model download and `llama.cpp` build.
- The script is attempting to pull and build local copy of `llama.cpp`, while no option to provide pre-built binary is available at the moment.
- There is no option to use pre-downloaded Huggingface model due to sandbox'y nature of the script, yet.

You are welcome to contribute to fix these issues.

## Contributing
Contributions are welcome. By submitting a contribution, you agree to license your contribution under the terms of the MIT License.

## Disclaimer
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

## Contact
For any queries or issues, please contact the maintainers.