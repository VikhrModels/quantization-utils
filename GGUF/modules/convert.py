import json
import os
import shutil

from huggingface_hub import snapshot_download
from modules.hf_model_helper import HFModelDownloader
from shared import (
    LLAMA_CPP_DIR,
    DType,
    LoggerMixin,
    ModelMixin,
    ensure_dir_exists,
    run_command,
)


def find_convert_script() -> str:
    """
    Find the convert_hf_to_gguf.py script in PATH or fallback to local copy.

    Returns:
        Full path to the conversion script

    Raises:
        FileNotFoundError: If script is not found
    """
    # Try to find globally installed script
    global_script = shutil.which("convert_hf_to_gguf.py")
    if global_script:
        return global_script

    # Check if it's in the same directory as llama.cpp binaries
    try:
        llama_cli = shutil.which("llama-cli")
        if llama_cli:
            # Get the directory of llama-cli and look for the script
            llama_dir = os.path.dirname(llama_cli)
            script_path = os.path.join(llama_dir, "convert_hf_to_gguf.py")
            if os.path.exists(script_path):
                return script_path

            # Also check one directory up (common structure)
            parent_dir = os.path.dirname(llama_dir)
            script_path = os.path.join(parent_dir, "convert_hf_to_gguf.py")
            if os.path.exists(script_path):
                return script_path
    except:
        pass

    # Fallback to local copy
    local_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    if os.path.exists(local_script):
        return local_script

    raise FileNotFoundError(
        "convert_hf_to_gguf.py script not found. "
        "Please ensure llama.cpp is properly installed or build locally."
    )


class Convert(LoggerMixin, ModelMixin):
    _dtype = DType.FP32

    def __init__(self, token=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token = token

        model_dir = self.get_model_dir()
        snapshot_download(
            self.model_id,
            local_dir=model_dir,
            token=token,
            allow_patterns=["config.json"],
        )
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.dtype = (
                    DType.FP16 if config.get("torch_dtype") == "float16" else DType.FP32
                )
        else:
            self.dtype = DType.FP32
            self.warning(
                f"Config file not found at {config_path}, defaulting dtype to {DType.FP32}"
            )

    @property
    def dtype(self) -> DType:
        return self._dtype

    @dtype.setter
    def dtype(self, value: DType):
        if value not in [DType.FP16, DType.FP32, DType.BF16]:
            raise ValueError(f"Invalid dtype: {value}")
        self._dtype = value

    def get_converted_model_filepath(self):
        return self.get_quantized_filepath(self.dtype.to_quant())

    def convert_model(self):
        try:
            model_dir = self.get_model_dir()
            config_file = os.path.join(model_dir, "config.json")

            HFModelDownloader(self.model_id, self.token).download_model(self._cwd)

            with open(config_file, "r") as f:
                config = json.load(f)
                self.dtype = DType(config.get("torch_dtype", DType.FP32))

            if not os.path.exists(self.get_converted_model_filepath()):
                convert_script = find_convert_script()
                self.info(f"Using conversion script: {convert_script}")

                ensure_dir_exists(f"{model_dir}-GGUF")
                self.info(f"Converting model {self.model_id}")
                run_command(
                    self.logger,
                    [
                        "python",
                        convert_script,
                        model_dir,
                        "--outtype",
                        "f16" if self.dtype == DType.FP16 else "f32",
                        "--outfile",
                        self.get_converted_model_filepath(),
                    ],
                    ".",
                )
        except Exception as e:
            self.exception(f"Error converting model: {e}")

    def download_model(self):
        model_dir = self.get_model_dir()
        ensure_dir_exists(model_dir)
        self.info(f"Downloading model {self.model_id}")
        snapshot_download(self.model_id, local_dir=model_dir, token=self.token)
        self.info(f"Model downloaded to {model_dir}")
