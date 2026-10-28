"""JSON and YAML read/write helpers. Writes are atomic (temp file + rename)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

PathLike = str | os.PathLike[str]


def _json_default(obj: Any) -> Any:
    # numpy arrays/scalars and similar objects expose tolist()/item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_text(path: PathLike, text: str) -> None:
    """Write a text file atomically (UTF-8, LF line endings), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: PathLike) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, data: Any, indent: int = 2) -> None:
    text = json.dumps(data, indent=indent, default=_json_default, ensure_ascii=False)
    write_text(path, text + "\n")


def read_yaml(path: PathLike) -> Any:
    import yaml

    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: PathLike, data: Any) -> None:
    """Write plain YAML (comments are lost; hand-edited files use ruamel.yaml, see tool 01)."""
    import yaml

    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    write_text(path, text)
