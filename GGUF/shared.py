import io
import logging
import os
import platform
import select
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import AnyStr, List

import git

LLAMA_CPP_DIR = "llama.cpp"
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"


def find_llama_binary(binary_name: str) -> str:
    """
    Find llama.cpp binary in PATH or local installations.

    Args:
        binary_name: Name of the binary (e.g., 'llama-quantize', 'llama-imatrix')

    Returns:
        Full path to the binary

    Raises:
        FileNotFoundError: If binary is not found
    """
    # Search order: PATH -> ~/.local/bin -> local build
    search_paths = [
        None,  # Use PATH
        Path.home() / ".local" / "bin",
        Path(LLAMA_CPP_DIR) / "build" / "bin",
        Path(LLAMA_CPP_DIR),
    ]

    for path in search_paths:
        if path is None:
            # Check global PATH
            global_binary = shutil.which(binary_name)
            if global_binary:
                return global_binary
        else:
            # Check specific directory
            binary_path = path / binary_name
            if binary_path.exists() and binary_path.is_file():
                # Check if executable
                if os.access(binary_path, os.X_OK):
                    return str(binary_path)

    # If not found, provide helpful error message
    raise FileNotFoundError(
        f"llama.cpp binary '{binary_name}' not found. "
        f"Please run 'python setup.py' to install llama.cpp or ensure it's in your PATH. "
        f"Searched in: PATH, ~/.local/bin, {LLAMA_CPP_DIR}/build/bin, {LLAMA_CPP_DIR}"
    )


def get_quantize_command() -> str:
    """Get the llama-quantize command path."""
    return find_llama_binary("llama-quantize")


def get_imatrix_command() -> str:
    """Get the llama-imatrix command path."""
    return find_llama_binary("llama-imatrix")


def get_perplexity_command() -> str:
    """Get the llama-perplexity command path."""
    return find_llama_binary("llama-perplexity")


def get_cli_command() -> str:
    """Get the llama-cli command path."""
    return find_llama_binary("llama-cli")


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


class DType(Enum):
    FP16 = "float16"
    FP32 = "float32"
    BF16 = "bfloat16"

    def to_quant(self) -> Quant:
        if self.value == "float16":
            return Quant.F16
        elif self.value == "float32":
            return Quant.F32
        elif self.value == "bfloat16":
            return Quant.BF16
        else:
            raise ValueError(f"Invalid dtype: {self}")


class StreamWrapper:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        if isinstance(self.stream, io.TextIOWrapper) and isinstance(data, bytes):
            self.stream.write(data.decode("ascii"))
        elif isinstance(self.stream, io.BytesIO) and isinstance(data, str):
            self.stream.write(data.encode("utf-8"))
        else:
            self.stream.write(data)

    def flush(self):
        if hasattr(self.stream, "flush"):
            self.stream.flush()


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
    """Ensure directory exists, create if it doesn't"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def run_command(logger: logging.Logger, command: List[str], cwd: str = "."):
    """Run command with proper logging and error handling"""
    logger.debug(
        f"Running command: '{' '.join(command)}' in {os.path.join(os.getcwd(), cwd)}"
    )
    process = subprocess.Popen(
        command,
        cwd=os.path.join(os.getcwd(), cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        text=True,
        bufsize=1,
    )

    output = []
    while True:
        # Use select to check if stdout or stderr is ready to be read
        reads = [process.stdout.fileno(), process.stderr.fileno()]
        ret = select.select(reads, [], [], 0.1)

        for fd in ret[0]:
            if fd == process.stdout.fileno():
                line = process.stdout.readline()
                if line:
                    logger.info(line.strip())
                    output.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
            if fd == process.stderr.fileno():
                line = process.stderr.readline()
                if line:
                    logger.warning(line.strip())
                    sys.stderr.write(line)
                    sys.stderr.flush()

        # Check if the process has finished
        if process.poll() is not None:
            break

    # Read any remaining output
    for line in process.stdout.readlines():
        logger.info(line.strip())
        output.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    for line in process.stderr.readlines():
        logger.warning(line.strip())
        sys.stderr.write(line)
        sys.stderr.flush()

    code = process.wait()

    if code != 0:
        logger.exception(
            f"Running command: '{' '.join(command)}' in {os.path.join(os.getcwd(), cwd)}: Error code: {code}",
            exc_info=True,
            stack_info=True,
        )
        sys.exit(127)
    else:
        logger.info(f"Command '{' '.join(command)}' completed successfully")

    return "".join(output)


def detect_environment():
    """Detect current environment and capabilities"""
    info = {
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "has_nvidia": False,
        "has_metal": False,
        "acceleration": "cpu",
    }

    # Detect NVIDIA
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        info["has_nvidia"] = result.returncode == 0
    except FileNotFoundError:
        pass

    # Detect Metal (Apple Silicon)
    if info["os"] == "darwin" and "arm" in info["arch"]:
        info["has_metal"] = True

    # Determine acceleration type
    if info["has_nvidia"]:
        info["acceleration"] = "cuda"
    elif info["has_metal"]:
        info["acceleration"] = "metal"

    return info


def validate_environment(logger: logging.Logger):
    """Validate that the environment is properly set up"""
    try:
        # Check if llama.cpp binaries are available
        binaries = ["llama-quantize", "llama-imatrix", "llama-perplexity"]
        missing_binaries = []

        for binary in binaries:
            try:
                find_llama_binary(binary)
                logger.info(f"✅ Found {binary}")
            except FileNotFoundError:
                missing_binaries.append(binary)
                logger.warning(f"❌ Missing {binary}")

        if missing_binaries:
            logger.error(
                f"Missing required binaries: {missing_binaries}. "
                f"Please run 'python setup.py' to install them."
            )
            return False

        # Check environment info
        env_info = detect_environment()
        logger.info(f"🖥️  Environment: {env_info['os']} {env_info['arch']}")
        logger.info(f"🚀 Acceleration: {env_info['acceleration']}")

        return True

    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


# Legacy functions for backward compatibility (deprecated)
def get_llamacpp(logger: logging.Logger):
    """Deprecated: Use setup.py instead"""
    logger.warning(
        "get_llamacpp() is deprecated. Please run 'python setup.py' instead."
    )
    return validate_environment(logger)


def build_llamacpp(logger, flags: List[AnyStr] = []):
    """Deprecated: Use setup.py instead"""
    logger.warning(
        "build_llamacpp() is deprecated. Please run 'python setup.py' instead."
    )


def git_pull_and_check(logger, repo_path) -> bool:
    """Check if git repository has updates"""
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
    """Clone repository if it doesn't exist"""
    if not os.path.exists(repo_path):
        logger.info(f"Repository {repo_path} does not exist, cloning from {repo_url}")
        git.Repo.clone_from(repo_url, repo_path)
        logger.info(f"Cloned repository {repo_url} to {repo_path}.")
        return True
