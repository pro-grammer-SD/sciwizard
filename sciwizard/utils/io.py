"""File I/O utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def safe_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV with sensible defaults and a clean error message.

    Args:
        path: File path.
        **kwargs: Extra arguments forwarded to ``pd.read_csv``.

    Returns:
        Parsed DataFrame.

    Raises:
        ValueError: On parse failure.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    try:
        df = pd.read_csv(path, **kwargs)
        logger.debug("Read %d rows × %d cols from %s", *df.shape, path.name)
        return df
    except Exception as exc:
        raise ValueError(f"Cannot parse '{path.name}' as CSV: {exc}") from exc


def ensure_dir(path: str | Path) -> Path:
    """Create *path* as a directory (including parents) if it does not exist.

    Args:
        path: Target directory path.

    Returns:
        The resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def unique_filename(directory: Path, stem: str, suffix: str) -> Path:
    """Return a path that does not yet exist by appending a counter if needed.

    Args:
        directory: Parent directory.
        stem: Base filename without extension.
        suffix: File extension including the dot (e.g. ``".csv"``).

    Returns:
        A Path guaranteed not to exist yet.
    """
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
