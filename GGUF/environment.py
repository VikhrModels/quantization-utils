import platform
import subprocess
from functools import lru_cache
from typing import Dict


def detect_environment() -> Dict[str, any]:
    """Detect current environment and hardware capabilities"""
    info = {
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "has_nvidia": False,
        "has_metal": False,
        "acceleration": "cpu",
    }

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        info["has_nvidia"] = result.returncode == 0
    except FileNotFoundError:
        pass

    if info["os"] == "darwin" and "arm" in info["arch"]:
        info["has_metal"] = True

    if info["has_nvidia"]:
        info["acceleration"] = "cuda"
    elif info["has_metal"]:
        info["acceleration"] = "metal"

    return info


@lru_cache(maxsize=1)
def get_environment_info() -> Dict[str, any]:
    """Get cached environment information"""
    return detect_environment()
