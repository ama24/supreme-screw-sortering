"""Repository-relative path resolution."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT_ENV = "SCREWSYNTH_ROOT"
_MARKER = "pyproject.toml"


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository root.

    Uses $SCREWSYNTH_ROOT if set, else walks up from this file (editable install) and then from the
    current directory until a directory containing pyproject.toml is found.
    """
    env = os.environ.get(ROOT_ENV)
    if env:
        return Path(env).resolve()
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / _MARKER).is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not locate the repository root ({_MARKER} not found); set ${ROOT_ENV}."
    )


def resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve a path: absolute paths are kept, relative paths are taken from the repo root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else repo_root() / p
