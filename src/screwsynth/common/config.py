"""Configuration loading: a single cached Config object built from configs/config.yaml."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from .io import read_yaml
from .paths import resolve

CONFIG_ENV = "SCREWSYNTH_CONFIG"
DEFAULT_CONFIG = "configs/config.yaml"
_MISSING = object()


class Config(Mapping[str, Any]):
    """Read-only view of a config dict with attribute and dotted-path access.

    cfg.scenegen.objects == cfg["scenegen"]["objects"] == cfg.get("scenegen.objects")
    """

    def __init__(self, data: Mapping[str, Any], source: Path | None = None) -> None:
        self._data = dict(data)
        self._source = source

    @property
    def source(self) -> Path | None:
        return self._source

    @staticmethod
    def _wrap(value: Any) -> Any:
        return Config(value) if isinstance(value, Mapping) else value

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"config has no key {key!r}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return self._wrap(node)

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise KeyError(f"missing config key {dotted!r} in {self._source}")
        return value

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


def config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Explicit path, else $SCREWSYNTH_CONFIG, else configs/config.yaml (repo-relative)."""
    return resolve(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG).resolve()


@cache
def _load(path: Path) -> Config:
    data = read_yaml(path) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top level of the config must be a mapping")
    return Config(data, source=path)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load (once) and return the config; repeated calls for the same file return one object."""
    return _load(config_path(path))


def clear_config_cache() -> None:
    _load.cache_clear()
