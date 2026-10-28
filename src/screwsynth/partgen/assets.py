"""Build, write and re-check the asset folder of a part: data/assets/<part_id>/."""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from screwsynth.catalogue.schema import Part
from screwsynth.common.config import Config
from screwsynth.common.io import read_json, write_json

from . import checks, collision
from .builders import BuiltPart, Lod, build
from .keypoints import keypoints_json, nominal_length
from .preview import render_preview

log = logging.getLogger(__name__)

GENERATOR_VERSION = "1"
RENDER = "render.ply"
COLLISION = "collision.obj"
PART_JSON = "part.json"
PREVIEW = "preview.png"


@dataclass(frozen=True)
class Settings:
    lod_name: str
    lod: Lod
    max_triangles: int = 200_000
    collision_segments: int = 16
    tolerance_mm: float = 0.01
    preview_px: int = 360

    @classmethod
    def from_config(cls, cfg: Config, lod_name: str | None = None) -> Settings:
        pg = cfg.get("partgen", {}) or {}
        name = lod_name or pg.get("lod", "med")
        levels = pg.get("lod_levels", {}) or {}
        if name not in levels:
            raise ValueError(f"unknown lod {name!r}; configured: {list(levels)}")
        return cls(
            lod_name=name,
            lod=Lod(**dict(levels[name])),
            max_triangles=int(pg.get("max_triangles", 200_000)),
            collision_segments=int(pg.get("collision_segments", 16)),
            tolerance_mm=float(pg.get("tolerance_mm", 0.01)),
            preview_px=int(pg.get("preview_px", 360)),
        )


@dataclass
class Result:
    part_id: str
    status: str  # built | skipped | failed
    triangles: int = 0
    seconds: float = 0.0
    problems: tuple[str, ...] = ()
    error: str = ""


def asset_dir(assets_dir: Path, part_id: str) -> Path:
    return Path(assets_dir) / part_id


def is_up_to_date(part: Part, folder: Path, settings: Settings) -> bool:
    meta_path = folder / PART_JSON
    if not all((folder / f).is_file() for f in (RENDER, COLLISION, PART_JSON, PREVIEW)):
        return False
    try:
        meta = read_json(meta_path)
    except (OSError, ValueError):
        return False
    return (
        meta.get("catalogue_hash") == part.hash
        and meta.get("generator_version") == GENERATOR_VERSION
        and meta.get("lod") == settings.lod_name
        and bool(meta.get("checks", {}).get("passed"))
    )


def run_checks(part: Part, built: BuiltPart, settings: Settings) -> list[str]:
    problems = checks.check_mesh(built.mesh, settings.max_triangles)
    problems += checks.check_dimensions(part, built.mesh, settings.tolerance_mm)
    problems += checks.check_keypoints(part, built.keypoints, settings.tolerance_mm)
    problems += checks.check_collision(built.mesh, built.collision)
    return problems


def part_json(
    part: Part, built: BuiltPart, settings: Settings, problems: list[str]
) -> dict[str, Any]:
    mesh = built.mesh
    lo, hi = mesh.bounds
    return {
        "id": part.id,
        "class_name": part.class_name,
        "family": part.family,
        "kind": part.kind,
        "shape": part.shape,
        "size": part.size,
        "length": part.length,
        "pitch": part.pitch,
        "fully_threaded": part.fully_threaded,
        "dims": dict(part.dims),
        "finishes": list(part.finishes),
        "nominal_length_mm": nominal_length(part, built.keypoints),
        "nominal_length_mode": part.nominal_length,
        "keypoints": keypoints_json(built.keypoints, built.labelled),
        "axis_height_when_lying": built.info.get("axis_height_when_lying"),
        "frame": "local: axis +z; screws: bearing at z=0, head +z, tip -z; others centred",
        "units": "mm",
        "mesh": {
            "file": RENDER,
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "volume_mm3": float(mesh.volume),
            "area_mm2": float(mesh.area),
            "bbox_min": lo.tolist(),
            "bbox_max": hi.tolist(),
            "center_of_mass": np.asarray(mesh.center_mass).tolist(),
            "material_slots": {k: list(v) for k, v in built.slots.items()},
        },
        "collision": {
            "file": COLLISION,
            "pieces": len(built.collision),
            "triangles": collision.total_triangles(built.collision),
        },
        "checks": {"passed": not problems, "problems": problems},
        "source": part.source,
        "info": {k: v for k, v in built.info.items() if k != "axis_height_when_lying"},
        "generator_version": GENERATOR_VERSION,
        "lod": settings.lod_name,
        "lod_params": asdict(settings.lod),
        "catalogue_hash": part.hash,
    }


def write_preview(folder: Path, mesh: trimesh.Trimesh, meta: dict, height_px: int) -> None:
    kps = (
        {k: v["xyz"] for k, v in meta["keypoints"].items()}
        if any(v["labelled"] for v in meta["keypoints"].values())
        else None
    )
    img = render_preview(
        mesh, meta["id"], height_px, slots=meta["mesh"]["material_slots"], keypoints=kps
    )
    img.save(folder / PREVIEW)


def build_asset(
    part: Part,
    assets_dir: str | os.PathLike[str],
    settings: Settings,
    force: bool = False,
    preview: bool = True,
) -> Result:
    folder = asset_dir(Path(assets_dir), part.id)
    if not force and is_up_to_date(part, folder, settings):
        return Result(part.id, "skipped")
    t0 = time.perf_counter()
    try:
        built = build(part, settings.lod, settings.collision_segments, settings.max_triangles)
        problems = run_checks(part, built, settings)
        folder.mkdir(parents=True, exist_ok=True)
        built.mesh.export(folder / RENDER, file_type="ply", encoding="binary")
        collision.write_obj(built.collision, folder / COLLISION)
        meta = part_json(part, built, settings, problems)
        (folder / PART_JSON).unlink(missing_ok=True)  # written last: marks a complete build
        if preview:
            write_preview(folder, built.mesh, meta, settings.preview_px)
        write_json(folder / PART_JSON, meta)
    except Exception as exc:  # one bad part (or MemoryError) must not stop a batch
        log.error("%s: build failed: %s: %s", part.id, type(exc).__name__, exc)
        return Result(part.id, "failed", seconds=time.perf_counter() - t0, error=str(exc))
    status = "built" if not problems else "failed"
    for p in problems:
        log.error("%s: check failed: %s", part.id, p)
    return Result(
        part.id,
        status,
        triangles=len(built.mesh.faces),
        seconds=time.perf_counter() - t0,
        problems=tuple(problems),
    )


def load_asset(folder: Path) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh], dict]:
    mesh = trimesh.load_mesh(folder / RENDER, process=False)
    return mesh, collision.read_obj(folder / COLLISION), read_json(folder / PART_JSON)


def recheck_asset(part: Part, assets_dir: Path, settings: Settings) -> Result:
    folder = asset_dir(assets_dir, part.id)
    if not (folder / PART_JSON).is_file():
        return Result(part.id, "failed", error="not built")
    mesh, pieces, meta = load_asset(folder)
    kps = {k: v["xyz"] for k, v in meta["keypoints"].items()}
    built = BuiltPart(mesh, {}, kps, meta["keypoints"]["tip"]["labelled"], pieces)
    problems = run_checks(part, built, settings)
    if meta.get("catalogue_hash") != part.hash:
        problems.append("asset is out of date with the catalogue (rebuild)")
    return Result(
        part.id, "built" if not problems else "failed", len(mesh.faces), problems=tuple(problems)
    )


def _job(args: tuple) -> Result:
    part, assets_dir, settings, force = args
    return build_asset(part, assets_dir, settings, force)


def build_many(
    parts: Iterable[Part],
    assets_dir: Path,
    settings: Settings,
    force: bool = False,
    workers: int = 1,
    on_result: Callable[[Result], None] | None = None,
) -> list[Result]:
    parts = list(parts)
    results: list[Result] = []
    if workers <= 1 or len(parts) <= 1:
        for part in parts:
            results.append(build_asset(part, assets_dir, settings, force))
            if on_result:
                on_result(results[-1])
        return results
    # Workers inherit the environment at spawn time. One BLAS thread each: OpenBLAS otherwise
    # reserves buffers for every core in every process and can fail to allocate on Windows.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_job, (p, assets_dir, settings, force)) for p in parts]
            for fut in as_completed(futures):
                results.append(fut.result())
                if on_result:
                    on_result(results[-1])
    except BrokenProcessPool:
        # usually a worker could not start (e.g. Windows commit memory exhausted)
        log.warning("worker processes failed; building the remaining parts in this process")
        done = {r.part_id for r in results}
        for part in parts:
            if part.id not in done:
                results.append(build_asset(part, assets_dir, settings, force))
                if on_result:
                    on_result(results[-1])
    order = {p.id: i for i, p in enumerate(parts)}
    return sorted(results, key=lambda r: order[r.part_id])


def clean(assets_dir: Path, keep: set[str] | None = None) -> list[str]:
    """Delete generated asset folders (those with a part.json). keep: part ids to keep."""
    removed = []
    if not assets_dir.is_dir():
        return removed
    for folder in sorted(assets_dir.iterdir()):
        if (
            folder.is_dir()
            and (folder / PART_JSON).is_file()
            and folder.name not in (keep or set())
        ):
            shutil.rmtree(folder)
            removed.append(folder.name)
    return removed
