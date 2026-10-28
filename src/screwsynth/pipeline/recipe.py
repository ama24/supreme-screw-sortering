"""Dataset recipes: everything needed to produce one dataset, in one small YAML file.

A recipe holds the pipeline settings (camera, number of scenes, steps, export formats) and only the
config values that differ from configs/config.yaml (`config:`, same structure as config.yaml). The
pipeline deep-merges them onto config.yaml and writes the result to output/<name>/config.yaml,
which every step then reads, so a run is fully described by that one file.

    name: run01
    description: first full dataset
    pipeline: {camera: nominal, scenes: 3000, chunk: 200, preview: false}
    config:
      catalogue: {extra_rules: [{match: DIN7991, enabled: false}]}
      scenegen: {objects: [20, 45]}
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from screwsynth.common.config import config_path
from screwsynth.common.io import read_yaml, write_yaml
from screwsynth.common.paths import repo_root, resolve

STEPS = ("catalogue", "parts", "restposes", "render", "postprocess", "labels")
CAMERAS = ("nominal", "calibration")
LABEL_FORMATS = ("seg", "pose", "det", "obb", "coco", "crops")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RecipeError(ValueError):
    pass


@dataclass
class PipelineSettings:
    camera: str = "nominal"  # nominal Pi Camera 3 values, or data/calibration (tool 03)
    scenes: int = 1000  # layouts to render (each gives scenegen.variants_per_layout images)
    chunk: int = 200  # scenes per Blender process (restarted between chunks)
    preview: bool = False  # low resolution and samples (scenegen.preview): a quick look
    steps: list[str] = field(default_factory=lambda: list(STEPS))
    formats: list[str] = field(default_factory=lambda: ["seg", "pose", "coco", "crops"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "scenes": self.scenes,
            "chunk": self.chunk,
            "preview": self.preview,
            "steps": list(self.steps),
            "formats": list(self.formats),
        }


@dataclass
class Recipe:
    name: str
    description: str = ""
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    config: dict[str, Any] = field(default_factory=dict)  # overrides of configs/config.yaml
    path: Path | None = None

    # ------------------------------------------------------------------ files

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: Path | None = None) -> Recipe:
        if not isinstance(data, Mapping):
            raise RecipeError("a recipe must be a mapping")
        unknown = set(data) - {"name", "description", "pipeline", "config"}
        if unknown:
            raise RecipeError(f"unknown recipe keys: {sorted(unknown)}")
        p = dict(data.get("pipeline") or {})
        defaults = PipelineSettings()
        unknown = set(p) - set(defaults.to_dict())
        if unknown:
            raise RecipeError(f"unknown pipeline keys: {sorted(unknown)}")
        r = cls(
            name=str(data.get("name") or (path.stem if path else "")),
            description=str(data.get("description") or ""),
            pipeline=PipelineSettings(
                camera=str(p.get("camera", defaults.camera)),
                scenes=int(p.get("scenes", defaults.scenes)),
                chunk=int(p.get("chunk", defaults.chunk)),
                preview=bool(p.get("preview", defaults.preview)),
                steps=list(p.get("steps", defaults.steps)),
                formats=list(p.get("formats", defaults.formats)),
            ),
            config=copy.deepcopy(dict(data.get("config") or {})),
            path=path,
        )
        r.validate()
        return r

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self.description:
            out["description"] = self.description
        out["pipeline"] = self.pipeline.to_dict()
        out["config"] = copy.deepcopy(self.config)
        return out

    def validate(self) -> None:
        if not NAME_RE.match(self.name):
            raise RecipeError(
                f"recipe name {self.name!r}: use letters, digits, '_', '-' or '.' (it names folders)"
            )
        p = self.pipeline
        if p.camera not in CAMERAS:
            raise RecipeError(f"pipeline.camera must be one of {CAMERAS}, not {p.camera!r}")
        if p.scenes < 1 or p.chunk < 1:
            raise RecipeError("pipeline.scenes and pipeline.chunk must be >= 1")
        bad = set(p.steps) - set(STEPS)
        if bad:
            raise RecipeError(f"unknown steps {sorted(bad)}; choose from {STEPS}")
        bad = set(p.formats) - set(LABEL_FORMATS)
        if bad:
            raise RecipeError(f"unknown label formats {sorted(bad)}; choose from {LABEL_FORMATS}")
        if not isinstance(self.config, Mapping):
            raise RecipeError("config: must be a mapping of config.yaml overrides")

    def save(self, path: Path | None = None) -> Path:
        self.validate()
        path = Path(path or self.path or recipes_dir() / f"{self.name}.yaml")
        write_yaml(path, self.to_dict())
        self.path = path
        return path

    # ------------------------------------------------------------- run layout

    def run_dir(self, base: Mapping[str, Any] | None = None) -> Path:
        base = base if base is not None else load_base_config()
        return resolve(_get(base, "paths.output_dir", "output")) / self.name

    def dataset_dir(self, base: Mapping[str, Any] | None = None) -> Path:
        base = base if base is not None else load_base_config()
        return resolve(_get(base, "paths.datasets_dir", "datasets")) / self.name

    def merged_config(self, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """config.yaml + this recipe's overrides, with a per-dataset classes.yaml."""
        base = base if base is not None else load_base_config()
        cfg = deep_merge(base, self.config)
        run_rel = Path(_get(cfg, "paths.output_dir", "output")) / self.name
        # the class list depends on the recipe's part selection, so each run gets its own
        cfg.setdefault("paths", {})["classes"] = (run_rel / "classes.yaml").as_posix()
        return cfg

    def write_run_config(self, base: Mapping[str, Any] | None = None) -> Path:
        base = base if base is not None else load_base_config()
        path = self.run_dir(base) / "config.yaml"
        write_yaml(path, self.merged_config(base))
        return path


# ---------------------------------------------------------------------- helpers


def recipes_dir() -> Path:
    return repo_root() / "recipes"


def list_recipes() -> list[Path]:
    d = recipes_dir()
    return sorted(d.glob("*.yaml")) if d.is_dir() else []


def load_recipe(path: str | Path) -> Recipe:
    path = Path(path)
    if not path.is_file() and not path.suffix:
        path = recipes_dir() / f"{path}.yaml"  # allow `run01` for recipes/run01.yaml
    path = resolve(path)
    if not path.is_file():
        raise RecipeError(f"recipe not found: {path}")
    return Recipe.from_dict(read_yaml(path) or {}, path)


def load_base_config(path: str | Path | None = None) -> dict[str, Any]:
    """configs/config.yaml (or $SCREWSYNTH_CONFIG) as a plain dict."""
    return dict(read_yaml(config_path(path)) or {})


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of `base` with `over` merged in; mappings merge key by key, anything else replaces."""
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def diff(base: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, Any]:
    """Smallest overrides that turn `base` into `full` (deep_merge(base, diff(base, full)) == full
    for every key of `full`)."""
    out: dict[str, Any] = {}
    for k, v in full.items():
        b = base.get(k, _MISSING)
        if isinstance(v, Mapping) and isinstance(b, Mapping):
            sub = diff(b, v)
            if sub:
                out[k] = sub
        elif b is _MISSING or b != v:
            out[k] = copy.deepcopy(v)
    return out


def get_path(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    return _get(data, dotted, default)


def set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    node = data
    *parents, leaf = dotted.split(".")
    for p in parents:
        if not isinstance(node.get(p), dict):
            node[p] = {}
        node = node[p]
    node[leaf] = value


def remove_path(data: dict[str, Any], dotted: str) -> None:
    """Delete a key and prune the mappings it leaves empty."""
    *parents, leaf = dotted.split(".")
    chain = [data]
    for p in parents:
        node = chain[-1].get(p)
        if not isinstance(node, dict):
            return
        chain.append(node)
    chain[-1].pop(leaf, None)
    for parent, key in zip(reversed(chain[:-1]), reversed(parents), strict=True):
        if parent.get(key) == {}:
            del parent[key]


_MISSING = object()


def _get(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node
