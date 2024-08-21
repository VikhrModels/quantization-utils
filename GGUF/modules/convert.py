import json
import os

from huggingface_hub import snapshot_download
from shared import (
    LoggerMixin,
    ModelMixin,
    ensure_dir_exists,
    run_command,
    DType,
)
from modules.llama_cpp import LLAMA_CPP_DIR
from modules.hf_model_helper import HFModelDownloader
from modules.llama_cpp import get_llamacpp

CONVERT_CMD = "convert_hf_to_gguf.py"


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
                    DType.F16 if config.get("torch_dtype") == "float16" else DType.FP32
                )
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
                        "f16" if self.dtype == DType.FP16 else "f32",
                        "--outfile",
                        self.get_converted_model_filepath(),
                    ],
                    LLAMA_CPP_DIR,
                )
        except Exception as e:
            self.exception(f"Error converting model: {e}")

    def download_model(self):
        model_dir = self.get_model_dir()
        ensure_dir_exists(model_dir)
        self.info(f"Downloading model {self.model_id}")
        snapshot_download(self.model_id, local_dir=model_dir, token=self.token)
        self.info(f"Model downloaded to {model_dir}")
