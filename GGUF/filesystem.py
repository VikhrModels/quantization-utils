from pathlib import Path


def ensure_dir_exists(dir_path: str) -> None:
    """
    Ensure directory exists, create if it doesn't

    Args:
            dir_path: Path to directory (string or Path object)
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)
