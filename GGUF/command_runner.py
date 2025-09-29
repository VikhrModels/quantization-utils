import logging
import os
import select
import subprocess
import sys
from typing import List

from exceptions import CommandExecutionError


def run_command(logger: logging.Logger, command: List[str], cwd: str = ".") -> str:
    """
    Run external command with proper logging and error handling

    Args:
            logger: Logger instance for output
            command: Command and arguments as list
            cwd: Working directory for command execution

    Returns:
            Combined stdout output as string

    Raises:
            CommandExecutionError: If command fails
    """
    full_cwd = os.path.join(os.getcwd(), cwd)
    command_str = " ".join(command)
    logger.debug(f"Running command: '{command_str}' in {full_cwd}")

    try:
        process = subprocess.Popen(
            command,
            cwd=full_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as e:
        raise CommandExecutionError(
            command=command_str, return_code=-1, cwd=full_cwd
        ) from e

    output = []

    while True:
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

        if process.poll() is not None:
            break

    for line in process.stdout.readlines():
        logger.info(line.strip())
        output.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    for line in process.stderr.readlines():
        logger.warning(line.strip())
        sys.stderr.write(line)
        sys.stderr.flush()

    return_code = process.wait()

    if return_code != 0:
        raise CommandExecutionError(
            command=command_str, return_code=return_code, cwd=full_cwd
        )

    logger.info(f"Command '{command_str}' completed successfully")
    return "".join(output)
