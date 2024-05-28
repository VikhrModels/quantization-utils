import io
import logging
import os
import platform
import subprocess
import sys
import functools
from enum import Enum
import time
from typing import AnyStr, List

import git

LLAMA_CPP_DIR = "llama.cpp"
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"


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


class StreamWrapper:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        if isinstance(self.stream, io.TextIOWrapper) and isinstance(data, bytes):
            self.stream.write(data.decode('ascii'))
        elif isinstance(self.stream, io.BytesIO) and isinstance(data, str):
            self.stream.write(data.encode('utf-8'))
        else:
            self.stream.write(data)

class ModelMixin:
    def __init__(self, model_id, cwd=None, *args, **kwargs):
        self._model_id = model_id
        self._cwd = cwd or os.getcwd()
        super().__init__(*args, **kwargs)

    @property
    def cwd(self):
        return self._cwd

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, value):
        self._model_id = value

    def get_model_name(self):
        return self._model_id.split("/")[-1]

    def get_model_dir(self):
        return os.path.join(self._cwd, "models", self._model_id)

    def get_quantized_filepath(self, quant: Quant):
        return os.path.join(
            self.get_model_dir() + "-GGUF",
            f"{self.get_model_name()}-{quant.value}.gguf",
        )


class LoggerMixin:
    def __init__(self, logger_name: str = __name__, *args, **kwargs):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(logging.INFO)
        super().__init__(*args, **kwargs)

    @property
    def logger(self):
        return self.__logger

    def info(self, msg, *args, **kwargs):
        self.__logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.__logger.debug(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.__logger.error(msg, *args, **kwargs)
        sys.exit(127)

    def warning(self, msg, *args, **kwargs):
        self.__logger.warning(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.__logger.critical(msg, *args, **kwargs)
        sys.exit(127)

    def exception(self, msg, stack_info=True, *args, **kwargs):
        self.__logger.exception(msg, stack_info=stack_info, *args, **kwargs)
        sys.exit(127)

    def log(self, level, msg, *args, **kwargs):
        self.__logger.log(level, msg, *args, **kwargs)


def ensure_dir_exists(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)


def run_command(logger: logging.Logger, command: List[str], cwd: str = "."):
    logger.debug(
        f"Running command: '{' '.join(command)}' in {os.path.join(os.getcwd(), cwd)}"
    )
    process = subprocess.Popen(
        ["stdbuf", "-oL"] + command,
        cwd=os.path.join(os.getcwd(), cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        text=True,
        bufsize=0,
    )

    wrapped_stdout = StreamWrapper(sys.__stdout__)
    wrapped_stderr = StreamWrapper(sys.__stderr__)
    while True:
        wrapped_stdout.write(process.stdout.read())
        wrapped_stderr.write(process.stderr.read())

        if process.poll() is not None:
            break
        time.sleep(0.05)

    code = process.wait()

    if code != 0:
        raise OSError(f"Command {command} failed with error code: {process.returncode}")
    else:
        logger.info(f"Command {command} completed successfully")
    return True


def get_llamacpp(logger: logging.Logger):
    current_os = platform.system()
    make_args = ["LLAMA_BLAS=ON", "LLAMA_BLAS_VENDOR=OpenBLAS"]

    if current_os == "Darwin":  # macOS
        logger.info("Using macOS")
        make_args.append("LLAMA_METAL=on")
    elif current_os == "Linux":
        logger.info("Using Linux")
        make_args.append("LLAMA_CUDA=1")
    else:
        raise OSError(f"Unsupported operating system: {current_os}")

    llama_cpp_dir = LLAMA_CPP_DIR
    logger.info(f"Checking and updating repository: {llama_cpp_dir}")
    is_updated = (
        git_clone_if_not_exist(logger, llama_cpp_dir, LLAMA_CPP_REPO)
        or git_pull_and_check(logger, llama_cpp_dir)
        or not os.path.exists(os.path.join(llama_cpp_dir, "imatrix"))
    )

    if is_updated:
        build_llamacpp(logger, make_args)


def build_llamacpp(logger, flags: List[AnyStr] = []):
    logger.info(f"Running make with flags: {flags}")
    make_args = [
        "make",
        f"-j{os.cpu_count()}",
        *flags,
    ]
    run_command(logger, make_args, LLAMA_CPP_DIR)
    run_command(logger, ["chmod", "+x", "imatrix", "quantize"], LLAMA_CPP_DIR)


def git_pull_and_check(logger, repo_path) -> bool:
    logger.info(f"Executing git pull for repository: {repo_path}")
    repo = git.Repo(repo_path)
    pull_info = repo.remotes.origin.pull()
    for info in pull_info:
        if info.flags & (info.NEW_HEAD | info.FAST_FORWARD | info.FORCED_UPDATE):
            logger.info(f"Repository {repo_path} updated with new changes.")
            return True
    logger.info(f"No new changes in repository {repo_path}.")
    return False


def git_clone_if_not_exist(logger, repo_path, repo_url):
    if not os.path.exists(repo_path):
        logger.info(f"Repository {repo_path} does not exist, cloning from {repo_url}")
        git.Repo.clone_from(repo_url, repo_path)
        logger.info(f"Cloned repository {repo_url} to {repo_path}.")
        return True
