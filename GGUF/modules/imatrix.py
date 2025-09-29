import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from datasets import load_dataset
from exceptions import IMatrixCalculationError
from huggingface_hub.utils import HfHubHTTPError
from retry import retry_on_network_error
from shared import (
    LoggerMixin,
    ModelMixin,
    Quant,
    ensure_dir_exists,
    get_imatrix_command,
    run_command,
)
from tqdm import tqdm
from transformers import AutoTokenizer

GGUF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Imatrix(LoggerMixin, ModelMixin):
    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id=model_id, *args, **kwargs)
        self.imatrix_dir = os.path.join(self.cwd, "imatrix")
        self.config = get_config().imatrix

    def get_imatrix_dir(self):
        return os.path.join(self.cwd, "imatrix")

    def get_imatrix_dataset_filepath(self, filename_template="imatrix.txt"):
        return os.path.join(
            self.get_imatrix_dir(), f"{self.model_id}.{filename_template}"
        )

    def get_imatrix_filepath(self, filename_template="imatrix.dat"):
        return os.path.join(
            self.get_imatrix_dir(), f"{self.model_id}.{filename_template}"
        )

    def prepare_imatrix_samples(self, qty=None, filepath=None):
        dataset = self.load_dataset()
        slice = self.select_data_slice(dataset, qty)
        tokenizer = self.load_tokenizer()
        self.write_imatrix_dataset(slice, tokenizer, filepath)
        self.append_calibration_data()

    @retry_on_network_error(
        max_retries=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(HfHubHTTPError, OSError, TimeoutError),
    )
    def load_dataset(self):
        dataset_id = self.config.default_dataset
        self.logger.info(f"Loading dataset {dataset_id}")
        return load_dataset(dataset_id)

    def select_data_slice(self, dataset, qty=None):
        qty = qty or self.config.sample_qty
        offset = random.randint(0, len(dataset["train"]) - qty)
        self.logger.info(f"Selecting data slice with offset {offset}")
        return (
            dataset["train"]
            .shuffle(seed=int(time.time()))
            .select(range(offset, offset + qty))
        )

    @retry_on_network_error(
        max_retries=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(HfHubHTTPError, OSError, TimeoutError),
    )
    def load_tokenizer(self):
        self.logger.info(f"Loading tokenizer for model {self.model_id}")
        return AutoTokenizer.from_pretrained(self.model_id)

    def write_imatrix_dataset(self, slice, tokenizer, filepath=None):
        filepath = filepath or self.get_imatrix_dataset_filepath()
        ensure_dir_exists(os.path.dirname(filepath))
        is_nonsystem = "gemma" in self.model_id.lower()
        with open(filepath, "w") as f:
            for i, example in tqdm(enumerate(slice), total=len(slice)):
                if not example["conversations"]:
                    continue
                messages = [
                    {
                        "role": conv["from"]
                        .replace("human", "user")
                        .replace("gpt", "assistant"),
                        "content": conv["value"] or conv["original"],
                    }
                    for conv in example["conversations"]
                ]
                if not is_nonsystem:
                    continue

                if len(messages) > 1:
                    first, messages = messages[0], messages[1:]
                    if first["role"] == "system":
                        messages[0]["content"] = (
                            first["content"] + "\n\n" + messages[0]["content"]
                        )
                    else:
                        messages = [first, *messages]
                else:
                    messages[0]["role"] = "user"
                try:
                    chat = tokenizer.apply_chat_template(messages, tokenize=False)
                except Exception as e:
                    self.logger.error(f"Error processing example {i}: {e}")
                    continue
                f.write(chat + "\n")

    def append_calibration_data(self):
        calibration_dir = os.path.join(GGUF_DIR, self.config.calibration_data_dir)
        self.logger.info(f"Reading all files from directory {calibration_dir}")
        with open(self.get_imatrix_dataset_filepath(), "a") as f:
            for filename in os.listdir(calibration_dir):
                file_path = os.path.join(calibration_dir, filename)
                if os.path.isfile(file_path):
                    self.logger.info(f"Reading file {file_path}")
                    with open(file_path, "r") as cal_file:
                        f.write(cal_file.read() + "\n")
        self.logger.info(
            f"Importance matrix dataset created: {self.get_imatrix_dataset_filepath()}"
        )

    def calculate_imatrix(self, base_quant: Quant):
        try:
            ensure_dir_exists(f"{self.get_model_dir()}-GGUF")
            self.logger.info(f"Calculating imatrix for model {self.model_id}")

            imatrix_cmd = get_imatrix_command()
            self.logger.info(f"Using imatrix binary: {imatrix_cmd}")
            is_cpu = self.is_cpu_environment()
            command = [
                imatrix_cmd,
                "-m",
                self.get_quantized_filepath(quant=base_quant),
                "-f",
                self.get_imatrix_dataset_filepath(),
                "-o",
                self.get_imatrix_filepath(),
            ]

            if not is_cpu:
                command.extend(["-ngl", str(self.config.gpu_layers)])

            tail_args = [
                "-c",
                str(self.config.context_size),
                "-b",
                str(self.config.batch_size),
                "--chunks",
                str(self.config.chunks),
            ]

            if not is_cpu:
                tail_args.extend(["--flash-attn", "on"])

            tail_args.extend(["--temp", str(self.config.temperature)])
            command.extend(tail_args)

            run_command(
                self.logger,
                command,
                ".",
            )
        except Exception as e:
            raise IMatrixCalculationError(self.model_id, reason=str(e)) from e
