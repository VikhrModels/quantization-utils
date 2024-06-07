import json
import os

from huggingface_hub import snapshot_download
from modules.quantize import (
    DTYPE_BF16,
    DTYPE_FP16,
    DTYPE_FP32,
)
from shared import (
    LoggerMixin,
    ModelMixin,
    Quant,
    ensure_dir_exists,
    run_command,
)
from modules.hf_model_helper import HFModelDownloader
from modules.llama_cpp import get_llamacpp

CONVERT_CMD = "convert-hf-to-gguf.py"


class Convert(LoggerMixin, ModelMixin):
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
                self.dtype = config.get("torch_dtype", DTYPE_FP32)
        else:
            self.dtype = DTYPE_FP32
            self.warning(
                f"Config file not found at {config_path}, defaulting dtype to {DTYPE_FP32}"
            )

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if value not in [DTYPE_FP16, DTYPE_FP32, DTYPE_BF16]:
            raise ValueError(f"Invalid dtype: {value}")
        self._dtype = value

    def get_converted_model_filepath(self):
        return self.get_quantized_filepath(
            Quant.F16 if self.dtype == DTYPE_FP16 else Quant.F32
        )

    def convert_model(self):
        try:
            model_dir = self.get_model_dir()
            config_file = os.path.join(model_dir, "config.json")

            HFModelDownloader(self.model_id, self.token).download_model(self._cwd)

            with open(config_file, "r") as f:
                config = json.load(f)
                self.dtype = config.get("torch_dtype", DTYPE_FP32)

            if not os.path.exists(self.get_converted_model_filepath()):
                get_llamacpp(self.logger)
                ensure_dir_exists(f"{model_dir}-GGUF")
                self.info(f"Converting model {self.model_id}")
                run_command(
                    self.logger,
                    [
                        "python",
                        CONVERT_CMD,
                        model_dir,
                        "--outtype",
                        "f16" if self.dtype == DTYPE_FP16 else "f32",
                        "--outfile",
                        self.get_converted_model_filepath(),
                    ],
                    "llama.cpp",
                )
        except Exception as e:
            self.exception(f"Error converting model: {e}")

    def download_model(self):
        model_dir = self.get_model_dir()
        ensure_dir_exists(model_dir)
        self.info(f"Downloading model {self.model_id}")
        snapshot_download(self.model_id, local_dir=model_dir, token=self.token)
        self.info(f"Model downloaded to {model_dir}")
