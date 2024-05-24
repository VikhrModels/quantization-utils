import logging
import os
import subprocess
from typing import List


def ensure_dir_exists(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)


def run_command(logger: logging.Logger, command: List[str], cwd: str = "."):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            logger.info(output.strip())

    stderr_output = process.stderr.read()
    if stderr_output:
        logger.error(stderr_output.strip())

    process.wait()

    if process.returncode != 0:
        raise OSError(f"Command {command} failed with error code: {process.returncode}")
    else:
        logger.info(f"Command {command} completed successfully")
    return True
