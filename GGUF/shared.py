import logging

from command_runner import run_command
from environment import detect_environment, get_environment_info
from exceptions import (
    BinaryNotFoundError,
)
from filesystem import ensure_dir_exists
from llama_binaries import (
    find_conversion_script,
    find_llama_binary,
    get_cli_command,
    get_imatrix_command,
    get_perplexity_command,
    get_quantize_command,
)
from mixins import LoggerMixin, ModelMixin

from models import DType, Quant

__all__ = [
    "DType",
    "Quant",
    "LoggerMixin",
    "ModelMixin",
    "detect_environment",
    "get_environment_info",
    "ensure_dir_exists",
    "find_llama_binary",
    "find_conversion_script",
    "get_quantize_command",
    "get_imatrix_command",
    "get_perplexity_command",
    "get_cli_command",
    "run_command",
    "validate_environment",
]


def validate_environment(logger: logging.Logger) -> bool:
    """
    Validate that the environment is properly set up

    Args:
            logger: Logger instance for output

    Returns:
            True if environment is valid, False otherwise

    Raises:
            EnvironmentValidationError: If validation fails critically
    """
    try:
        binaries = ["llama-quantize", "llama-imatrix", "llama-perplexity"]
        missing_binaries = []

        for binary in binaries:
            try:
                find_llama_binary(binary)
                logger.info(f"✅ Found {binary}")
            except BinaryNotFoundError:
                missing_binaries.append(binary)
                logger.warning(f"❌ Missing {binary}")

        if missing_binaries:
            logger.error(
                f"Missing required binaries: {missing_binaries}. "
                f"Please run 'python setup.py' to install them."
            )
            return False

        env_info = detect_environment()
        logger.info(f"🖥️  Environment: {env_info['os']} {env_info['arch']}")
        logger.info(f"🚀 Acceleration: {env_info['acceleration']}")

        return True

    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False
