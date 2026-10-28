"""Logging setup shared by all CLIs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = "INFO", file: str | os.PathLike[str] | None = None
) -> logging.Logger:
    """Configure the root logger (console, plus an optional file). Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(level.upper() if isinstance(level, str) else level)
    for handler in list(root.handlers):
        if getattr(handler, "_screwsynth", False):
            root.removeHandler(handler)
            handler.close()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if file is not None:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file, encoding="utf-8"))
    formatter = logging.Formatter(FORMAT, DATEFMT)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler._screwsynth = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
