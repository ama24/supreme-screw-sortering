"""Scene composition: class-balanced sampling of parts, lengths and finishes (pure Python)."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PartInfo:
    id: str
    class_name: str
    kind: str
    finishes: tuple[str, ...]
    length: float | None = None


@dataclass
class BalanceState:
    """Instances produced so far per class (persisted so resumed runs stay balanced)."""

    counts: dict[str, int] = field(default_factory=dict)
    scenes: int = 0

    def save(self, path: str | os.PathLike[str]) -> None:
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(
            json.dumps({"counts": self.counts, "scenes": self.scenes}, indent=1), encoding="utf-8"
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BalanceState:
        p = Path(path)
        if not p.is_file():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            counts={k: int(v) for k, v in d.get("counts", {}).items()},
            scenes=int(d.get("scenes", 0)),
        )


@dataclass
class ScenePlan:
    kind: str  # normal | few | crowded
    parts: list[tuple[PartInfo, str]]  # (part, finish)
    gap_mm: float
    touch_probability: float


class BalancedSampler:
    """Favour under-represented classes: draw uniformly among the classes whose instance count is
    within `slack` of the current minimum (deficit sampling; keeps the spread across classes <= slack + 1).
    """

    def __init__(
        self,
        parts: Sequence[PartInfo],
        state: BalanceState | None = None,
        slack: int = 1,
        finish_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.by_class: dict[str, list[PartInfo]] = {}
        for p in parts:
            self.by_class.setdefault(p.class_name, []).append(p)
        if not self.by_class:
            raise ValueError("no parts to sample from")
        self.classes = sorted(self.by_class)
        self.state = state or BalanceState()
        for c in self.classes:
            self.state.counts.setdefault(c, 0)
        self.slack = max(0, int(slack))
        self.finish_weights = dict(finish_weights or {})

    def candidates(self) -> list[str]:
        lo = min(self.state.counts[c] for c in self.classes)
        return [c for c in self.classes if self.state.counts[c] <= lo + self.slack]

    def sample_class(self, rng: random.Random) -> str:
        return rng.choice(self.candidates())

    def sample_part(self, class_name: str, rng: random.Random) -> PartInfo:
        """Uniform over the class's enabled parts (i.e. over its lengths)."""
        return rng.choice(self.by_class[class_name])

    def sample_finish(self, part: PartInfo, rng: random.Random) -> str:
        if not part.finishes:
            raise ValueError(f"{part.id} has no finishes")
        w = [self.finish_weights.get(f, 1.0) for f in part.finishes]
        return rng.choices(part.finishes, weights=w, k=1)[0]

    def draw(self, rng: random.Random) -> tuple[PartInfo, str]:
        c = self.sample_class(rng)
        part = self.sample_part(c, rng)
        return part, self.sample_finish(part, rng)

    def record(self, class_names: Sequence[str]) -> None:
        """Count instances that actually ended up in a rendered scene."""
        for c in class_names:
            self.state.counts[c] = self.state.counts.get(c, 0) + 1
        self.state.scenes += 1


def _range(cfg: Mapping[str, Any], key: str, default: Sequence[int]) -> tuple[int, int]:
    lo, hi = cfg.get(key, default)
    return int(lo), int(hi)


def plan_scene(sampler: BalancedSampler, cfg: Mapping[str, Any], rng: random.Random) -> ScenePlan:
    """Choose the scene type, the number of objects and the parts + finishes.

    During one scene the sampler balances against a provisional count so a single crowded scene does
    not pile up the same class.
    """
    u = rng.random()
    few_f = float(cfg.get("few_fraction", 0.1))
    crowded_f = float(cfg.get("crowded_fraction", 0.2))
    if u < few_f:
        kind, (lo, hi) = "few", _range(cfg, "few_objects", (1, 5))
    elif u < few_f + crowded_f:
        kind, (lo, hi) = "crowded", _range(cfg, "crowded_objects", (30, 50))
    else:
        kind, (lo, hi) = "normal", _range(cfg, "objects", (15, 40))
    n = rng.randint(lo, hi)
    provisional = dict(sampler.state.counts)
    saved = sampler.state.counts
    sampler.state.counts = provisional
    parts = []
    try:
        for _ in range(n):
            part, finish = sampler.draw(rng)
            provisional[part.class_name] = provisional.get(part.class_name, 0) + 1
            parts.append((part, finish))
    finally:
        sampler.state.counts = saved
    if kind == "crowded":
        gap, touch = (
            float(cfg.get("crowded_gap_mm", -0.3)),
            float(cfg.get("touch_probability", 0.5)),
        )
    else:
        gap, touch = float(cfg.get("min_gap_mm", 1.0)), 0.0
    return ScenePlan(kind, parts, gap, touch)


def parts_from_assets(
    assets_dir: str | os.PathLike[str],
    classes: Sequence[str],
    enabled_ids: set[str] | None = None,
    rest_poses_dir: str | os.PathLike[str] | None = None,
) -> list[PartInfo]:
    """Enabled parts that have a built asset (and rest poses), from their part.json files."""
    out = []
    wanted = set(classes)
    for pj in sorted(Path(assets_dir).glob("*/part.json")):
        meta = json.loads(pj.read_text(encoding="utf-8"))
        if meta["class_name"] not in wanted or (
            enabled_ids is not None and meta["id"] not in enabled_ids
        ):
            continue
        if (
            rest_poses_dir is not None
            and not (Path(rest_poses_dir) / f"{meta['id']}.json").is_file()
        ):
            continue
        out.append(
            PartInfo(
                meta["id"],
                meta["class_name"],
                meta["kind"],
                tuple(meta["finishes"]),
                meta.get("length"),
            )
        )
    return out
