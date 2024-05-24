from getpass import getpass
import os
import time
import logging
import random
import platform
from util import ensure_dir_exists, run_command
from typing import List, AnyStr
from huggingface_hub import snapshot_download
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import git
from enum import Enum


class Quant(Enum):
    IQ1_S = "IQ1_S"
    IQ1_M = "IQ1_M"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ2_S = "IQ2_S"
    IQ2_M = "IQ2_M"
    Q2_K = "Q2_K"
    Q2_K_S = "Q2_K_S"
    IQ3_XXS = "IQ3_XXS"
    IQ3_S = "IQ3_S"
    IQ3_M = "IQ3_M"
    Q3_K = "Q3_K"
    IQ3_XS = "IQ3_XS"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"
    IQ4_NL = "IQ4_NL"
    IQ4_XS = "IQ4_XS"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K = "Q4_K"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K = "Q5_K"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"
    BF16 = "BF16"
    F32 = "F32"


class ImatrixPipeline:
    def __init__(self, logger: logging.Logger, token: str, model_id: str):
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.logger = logger
        self.logger.info(f"Working directory: {self.cwd}")
        os.chdir(self.cwd)
        self.init_llamacpp()
        self.token = token
        self.model_id = model_id

    def get_model_name(self):
        return self.model_id.split("/")[-1]

    def get_model_dir(self):
        return os.path.join(self.cwd, "models", self.model_id)

    def get_imatrix_dir(self):
        return os.path.join(self.cwd, "imatrix")

    def get_imatrix_dataset_filepath(self):
        return os.path.join(self.get_imatrix_dir(), f"{self.model_id}.imatrix.txt")

    def get_imatrix_filepath(self):
        return os.path.join(self.get_imatrix_dir(), f"{self.model_id}.imatrix.dat")

    def get_quantized_filepath(self, quant: Quant):
        return os.path.join(
            self.get_model_dir() + "-GGUF",
            f"{self.get_model_name()}-{quant.value}.gguf",
        )

    def prepare_imatrix_samples(self):
        dataset_id = "Vikhrmodels/Veles-2.5"
        self.logger.info(f"Loading dataset {dataset_id}")
        dataset = load_dataset(dataset_id)

        offset = random.randint(0, len(dataset["train"]) - 10000)
        self.logger.info(f"Selecting data slice with offset {offset}")
        slice = (
            dataset["train"]
            .shuffle(seed=int(time.time()))
            .select(range(offset, offset + 10000))
        )

        self.logger.info(f"Loading tokenizer for model {self.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.chat_template = (
            tokenizer.chat_template
            if tokenizer.chat_template
            else "<s>{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )

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

            standard_cal_data_dir = "standard_cal_data"
            self.logger.info(
                f"Reading all files from directory {standard_cal_data_dir}"
            )

            for filename in os.listdir(standard_cal_data_dir):
                file_path = os.path.join(standard_cal_data_dir, filename)
                if os.path.isfile(file_path):
                    self.logger.info(f"Reading file {file_path}")
                    with open(file_path, "r") as cal_file:
                        f.write(cal_file.read() + "\n")

            self.logger.info(
                f"Importance matrix dataset created: {self.get_imatrix_dataset_filepath()}"
            )

    def calculate_imatrix(self):
        ensure_dir_exists(f"{self.get_model_dir()}-GGUF")
        self.logger.info(f"Calculating imatrix for model {self.model_id}")
        run_command(
            self.logger,
            [
                "./imatrix",
                "-m",
                self.get_quantized_filepath(Quant.F16),
                "-f",
                self.get_imatrix_dataset_filepath(),
                "-o",
                self.get_imatrix_filepath(),
                "-ngl",
                "32",
                "-c",
                "512",
                "-b",
                "512",
                "--chunks",
                "1000",
                "--mlock",
            ],
            "llama.cpp",
        )

    def convert_model(self):
        model_dir = self.get_model_dir()
        ensure_dir_exists(model_dir)
        ensure_dir_exists(f"{model_dir}-GGUF")
        self.logger.info(f"Downloading model {self.model_id}")
        snapshot_download(self.model_id, local_dir=model_dir)
        self.logger.info(f"Model downloaded to {model_dir}")

        self.logger.info(f"Converting model {self.model_id}")
        run_command(
            self.logger,
            [
                "python",
                "convert.py",
                model_dir,
                "--outtype",
                "f16",
                "--outfile",
                self.get_quantized_filepath(Quant.F16),
            ],
            "llama.cpp",
        )

    def init_llamacpp(self):
        current_os = platform.system()
        make_args = ["LLAMA_BLAS=ON", "LLAMA_BLAS_VENDOR=OpenBLAS"]

        if current_os == "Darwin":  # macOS
            self.logger.info("Using macOS")
            make_args.append("LLAMA_METAL=on")
        elif current_os == "Linux":
            self.logger.info("Using Linux")
            make_args.append("LLAMA_CUDA=1")
        else:
            raise OSError(f"Unsupported operating system: {current_os}")

        llama_cpp_dir = "llama.cpp"
        self.logger.info(f"Checking and updating repository: {llama_cpp_dir}")
        is_updated = (
            self.git_clone_if_not_exist(
                llama_cpp_dir, "https://github.com/ggerganov/llama.cpp"
            )
            or self.git_pull_and_check(llama_cpp_dir)
            or not os.path.exists(f"{llama_cpp_dir}/imatrix")
        )

        if is_updated:
            self.build_llama_cpp(make_args)

    def git_pull_and_check(self, repo_path) -> bool:
        self.logger.info(f"Executing git pull for repository: {repo_path}")
        repo = git.Repo(repo_path)
        pull_info = repo.remotes.origin.pull()
        for info in pull_info:
            if info.flags & (info.NEW_HEAD | info.FAST_FORWARD | info.FORCED_UPDATE):
                self.logger.info(f"Repository {repo_path} updated with new changes.")
                return True
        self.logger.info(f"No new changes in repository {repo_path}.")
        return False

    def git_clone_if_not_exist(self, repo_path, repo_url):
        if not os.path.exists(repo_path):
            self.logger.info(
                f"Repository {repo_path} does not exist, cloning from {repo_url}"
            )
            git.Repo.clone_from(repo_url, repo_path)
            self.logger.info(f"Cloned repository {repo_url} to {repo_path}.")
            return True
        self.logger.info(f"Repository {repo_path} already exists.")
        return False

    def build_llama_cpp(self, flags: List[AnyStr] = []):
        self.logger.info(f"Running make with flags: {flags}")

        make_args = ["make", *flags, "-j8"]
        run_command(self.logger, make_args, "llama.cpp")
        run_command(self.logger, ["chmod", "+x", "imatrix", "quantize"], "llama.cpp")

    def quantize_model(self, quants: List[Quant]):
        imatrix_attrs = [
            "--imatrix",
            self.get_imatrix_filepath(),
        ]
        for quant in quants:
            self.logger.info(f"Quantizing model {self.model_id} for quant {quant}")
            run_command(
                self.logger,
                [
                    "./quantize",
                    *(  # do not use imatrix for upper quants, may lead to lower quality
                        imatrix_attrs
                        if quant
                        not in [
                            Quant.Q4_0,
                            Quant.Q4_1,
                            Quant.Q4_K,
                            Quant.Q4_K_S,
                            Quant.Q4_K_M,
                            Quant.Q5_0,
                            Quant.Q5_1,
                            Quant.Q5_K,
                            Quant.Q5_K_S,
                            Quant.Q5_K_M,
                            Quant.Q6_K,
                            Quant.Q8_0,
                            Quant.F16,
                            Quant.BF16,
                            Quant.F32,
                        ]
                        else []
                    ),
                    self.get_quantized_filepath(Quant.F16),
                    self.get_quantized_filepath(quant),
                    quant.value,
                ],
                "llama.cpp",
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
    )
    logger = logging.getLogger(__name__)
    default_model_id = "Vikhrmodels/it-5.2-fp16-cp"
    pipeline = ImatrixPipeline(
        logger=logger,
        token=os.getenv("HF_TOKEN") or getpass("Provide your HF_TOKEN: "),
        model_id=input(f"Provide HF model ID ({default_model_id}): ") or default_model_id,
    )
    if not os.path.exists(pipeline.get_imatrix_filepath()):
        pipeline.prepare_imatrix_samples()
        pipeline.convert_model()
        pipeline.calculate_imatrix()

    if not os.path.exists(pipeline.get_quantized_filepath(Quant.F16)):
        pipeline.convert_model()

    if not os.path.exists(pipeline.get_imatrix_filepath()):
        pipeline.calculate_imatrix()

    pipeline.quantize_model(
        [
            Quant.IQ1_S,
            Quant.IQ1_M,
            Quant.IQ2_XXS,
            Quant.IQ2_XS,
            Quant.IQ2_S,
            Quant.IQ2_M,
            Quant.Q2_K,
            Quant.Q2_K_S,
            Quant.IQ3_XXS,
            Quant.IQ3_XS,
            Quant.IQ3_S,
            Quant.IQ3_M,
            Quant.Q3_K,
            Quant.Q3_K_S,
            Quant.Q3_K_M,
            Quant.Q3_K_L,
            Quant.IQ4_NL,
            Quant.IQ4_XS,
            Quant.Q4_0,
            Quant.Q4_1,
            Quant.Q4_K,
            Quant.Q4_K_S,
            Quant.Q4_K_M,
            Quant.Q5_0,
            Quant.Q5_1,
            Quant.Q5_K,
            Quant.Q5_K_S,
            Quant.Q5_K_M,
            Quant.Q6_K,
            Quant.Q8_0,
            # Quant.F16, # uncomment for f32 base
            # Quant.BF16, # uncomment for f32 base
            # Quant.F32, # uncomment for f32 base
        ]
    )
