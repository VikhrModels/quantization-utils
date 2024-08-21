import os
import platform
import logging
from shared import run_command, git_clone_if_not_exist, git_pull_and_check

LLAMA_CPP_DIR = "llama.cpp"
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"


def get_llamacpp(logger: logging.Logger):
    current_os = platform.system()
    make_args = ["LLAMA_BLAS=ON", "LLAMA_BLAS_VENDOR=OpenBLAS"]

    if current_os == "Darwin":
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


def build_llamacpp(logger, flags=[]):
    logger.info(f"Running make with flags: {flags}")
    make_args = ["make", f"-j{os.cpu_count()}", *flags]
    run_command(logger, make_args, LLAMA_CPP_DIR)
    run_command(
        logger, ["chmod", "+x", "llama-imatrix", "llama-quantize"], LLAMA_CPP_DIR
    )
