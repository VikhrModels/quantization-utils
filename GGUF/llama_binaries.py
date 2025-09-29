import os
import shutil
from pathlib import Path
from typing import List

from exceptions import BinaryNotFoundError, ConversionScriptNotFoundError

LLAMA_CPP_DIR = "llama.cpp"


def find_llama_binary(binary_name: str) -> str:
    """
    Find llama.cpp binary in PATH or local installations

    Args:
            binary_name: Name of the binary (e.g., 'llama-quantize', 'llama-imatrix')

    Returns:
            Full path to the binary

    Raises:
            BinaryNotFoundError: If binary is not found
    """
    search_paths = [
        None,
        Path.home() / ".local" / "bin",
        Path(LLAMA_CPP_DIR) / "build" / "bin",
        Path(LLAMA_CPP_DIR),
    ]

    searched: List[str] = []

    for path in search_paths:
        if path is None:
            global_binary = shutil.which(binary_name)
            if global_binary:
                return global_binary
            searched.append("PATH")
        else:
            binary_path = path / binary_name
            searched.append(str(path))
            if binary_path.exists() and binary_path.is_file():
                if os.access(binary_path, os.X_OK):
                    return str(binary_path)

    raise BinaryNotFoundError(binary_name, searched)


def find_conversion_script(script_name: str = "convert_hf_to_gguf.py") -> str:
    """
    Find conversion scripts in the system

    Args:
            script_name: Name of the conversion script to find

    Returns:
            Full path to the script

    Raises:
            ConversionScriptNotFoundError: If script is not found
    """
    global_script = shutil.which(script_name)
    if global_script:
        return global_script

    try:
        llama_cli = find_llama_binary("llama-cli")
        llama_dir = os.path.dirname(llama_cli)
        script_path = os.path.join(llama_dir, script_name)
        if os.path.exists(script_path):
            return script_path
    except BinaryNotFoundError:
        pass

    common_paths = [
        os.path.expanduser("~/.local/bin"),
        "/usr/local/bin",
        "/opt/homebrew/bin",
        LLAMA_CPP_DIR,
    ]

    for path in common_paths:
        script_path = os.path.join(path, script_name)
        if os.path.exists(script_path):
            return script_path

    raise ConversionScriptNotFoundError(script_name)


def get_quantize_command() -> str:
    """Get the llama-quantize command path"""
    return find_llama_binary("llama-quantize")


def get_imatrix_command() -> str:
    """Get the llama-imatrix command path"""
    return find_llama_binary("llama-imatrix")


def get_perplexity_command() -> str:
    """Get the llama-perplexity command path"""
    return find_llama_binary("llama-perplexity")


def get_cli_command() -> str:
    """Get the llama-cli command path"""
    return find_llama_binary("llama-cli")
