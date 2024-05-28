import os
import random
import time

from datasets import load_dataset
from shared import LoggerMixin, ModelMixin, Quant, ensure_dir_exists, run_command
from tqdm import tqdm
from transformers import AutoTokenizer

IMATRIX_CMD = "./imatrix"
STANDARD_CAL_DATA_DIR = os.path.join("resources", "standard_cal_data")


class Imatrix(LoggerMixin, ModelMixin):
    def __init__(self, model_id: str, *args, **kwargs):
        super().__init__(model_id=model_id, *args, **kwargs)
        self.imatrix_dir = os.path.join(os.getcwd(), "imatrix")

    def get_imatrix_dir(self):
        return os.path.join(self.cwd, "imatrix")

    def get_imatrix_dataset_filepath(self):
        return os.path.join(self.get_imatrix_dir(), f"{self.model_id}.imatrix.txt")

    def get_imatrix_filepath(self):
        return os.path.join(self.get_imatrix_dir(), f"{self.model_id}.imatrix.dat")

    def prepare_imatrix_samples(self):
        dataset = self.load_dataset()
        slice = self.select_data_slice(dataset)
        tokenizer = self.load_tokenizer()
        self.write_imatrix_dataset(slice, tokenizer)
        self.append_calibration_data()

    def load_dataset(self):
        dataset_id = "Vikhrmodels/Veles-2.5"
        self.logger.info(f"Loading dataset {dataset_id}")
        return load_dataset(dataset_id)

    def select_data_slice(self, dataset):
        offset = random.randint(0, len(dataset["train"]) - 10000)
        self.logger.info(f"Selecting data slice with offset {offset}")
        return (
            dataset["train"]
            .shuffle(seed=int(time.time()))
            .select(range(offset, offset + 10000))
        )

    def load_tokenizer(self):
        self.logger.info(f"Loading tokenizer for model {self.model_id}")
        return AutoTokenizer.from_pretrained(self.model_id)

    def write_imatrix_dataset(self, slice, tokenizer):
        ensure_dir_exists(os.path.dirname(self.get_imatrix_dataset_filepath()))
        with open(self.get_imatrix_dataset_filepath(), "w") as f:
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
                try:
                    chat = tokenizer.apply_chat_template(messages, tokenize=False)
                except Exception as e:
                    self.logger.error(f"Error processing example {i}: {e}")
                    continue
                f.write(chat + "\n")

    def append_calibration_data(self):
        self.logger.info(f"Reading all files from directory {STANDARD_CAL_DATA_DIR}")
        with open(self.get_imatrix_dataset_filepath(), "a") as f:
            for filename in os.listdir(STANDARD_CAL_DATA_DIR):
                file_path = os.path.join(STANDARD_CAL_DATA_DIR, filename)
                if os.path.isfile(file_path):
                    self.logger.info(f"Reading file {file_path}")
                    with open(file_path, "r") as cal_file:
                        f.write(cal_file.read() + "\n")
        self.logger.info(
            f"Importance matrix dataset created: {self.get_imatrix_dataset_filepath()}"
        )

    def calculate_imatrix(self):
        try:
            ensure_dir_exists(f"{self.get_model_dir()}-GGUF")
            self.logger.info(f"Calculating imatrix for model {self.model_id}")
            run_command(
                self.logger,
                [
                    IMATRIX_CMD,
                    "-m",
                    self.get_quantized_filepath(Quant.F16),
                    "-f",
                    self.get_imatrix_dataset_filepath(),
                    "-o",
                    self.get_imatrix_filepath(),
                    "-cml",
                    "-ngl",
                    "32",
                    "-c",
                    "512",
                    "-b",
                    "512",
                    "--chunks",
                    "1024",
                    "--mlock",
                    "-fa",
                    "-cb",
                    "--numa",
                    "distribute",
                    "-sm",
                    "row",
                    "--temp",
                    "0.5",
                ],
                "llama.cpp",
            )
        except Exception as e:
            self.logger.error(f"Error calculating imatrix: {e}")
            raise
