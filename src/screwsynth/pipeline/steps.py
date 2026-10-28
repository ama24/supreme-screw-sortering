"""The dataset pipeline as a list of steps, each a list of commands, plus preflight checks.

Every command is an ordinary CLI call of one tool with --config output/<name>/config.yaml (the merged
recipe), run in its own process: a crash in Blender or a worker cannot take the GUI down, and each
step can also be run by hand from a terminal with exactly the same command.
"""

from __future__ import annotations

import ctypes
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from screwsynth.common.paths import resolve

from .recipe import Recipe, get_path

GENERATE = Path("tools") / "scenegen" / "generate.py"

TITLES = {
    "catalogue": "Catalogue: validate, write classes.yaml",
    "parts": "Parts: build meshes (incremental)",
    "restposes": "Rest poses (incremental)",
    "render": "Render scenes (Blender)",
    "postprocess": "Post-process: lens + sensor",
    "labels": "Labels: export dataset",
}


@dataclass
class Step:
    key: str
    title: str
    # built when the step starts: later steps depend on what earlier ones produced
    commands: Callable[[], list[list[str]]]
    # (done, total) while the step runs, e.g. rendered scenes; None = count finished commands
    progress: Callable[[], tuple[int, int]] | None = None
    # removed before the step runs (derived outputs that would otherwise go stale)
    clean: list[Path] = field(default_factory=list)
    # images worth showing for this step (may not exist yet)
    previews: list[Path] = field(default_factory=list)


@dataclass
class Plan:
    recipe: Recipe
    config: dict[str, Any]
    config_path: Path
    run_dir: Path
    dataset_dir: Path
    steps: list[Step]

    def step(self, key: str) -> Step:
        return next(s for s in self.steps if s.key == key)


def py_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def blenderproc_exe() -> str | None:
    """blenderproc next to this Python (the venv), else on PATH."""
    scripts = Path(sys.executable).parent
    for name in ("blenderproc.exe", "blenderproc"):
        if (scripts / name).is_file():
            return str(scripts / name)
    return shutil.which("blenderproc")


def build_plan(recipe: Recipe, base: dict[str, Any] | None = None) -> Plan:
    """Steps of the recipe, in pipeline order. Writes nothing (see write_run_config)."""
    cfg = recipe.merged_config(base)
    run_dir = recipe.run_dir(cfg)
    dataset_dir = recipe.dataset_dir(cfg)
    config_path = run_dir / "config.yaml"
    c = ["--config", str(config_path)]
    p = recipe.pipeline
    name = recipe.name
    previews = run_dir / "previews"

    def catalogue() -> list[list[str]]:
        mod = "screwsynth.catalogue.cli"
        return [py_module(mod, "validate", *c), py_module(mod, "classes", *c)]

    def parts() -> list[list[str]]:
        mod = "screwsynth.partgen.cli"
        sheet = str(previews / "parts.png")
        return [
            py_module(mod, "build", "--all", *c),
            py_module(mod, "contact-sheet", "--out", sheet, *c),
        ]

    def restposes() -> list[list[str]]:
        mod = "screwsynth.restposes.cli"
        sheet = str(previews / "restposes.png")
        return [
            py_module(mod, "compute", "--all", *c),
            py_module(mod, "contact-sheet", "--out", sheet, *c),
        ]

    def render() -> list[list[str]]:
        exe = blenderproc_exe() or "blenderproc"
        cmds = []
        for start in range(0, p.scenes, p.chunk):
            n = min(p.chunk, p.scenes - start)
            cmd = [exe, "run", str(GENERATE), "--run", name, "--start", str(start)]
            cmd += ["--scenes", str(n), "--resume", *c]
            if p.camera == "nominal":
                cmd.append("--nominal-camera")
            if p.preview:
                cmd.append("--preview")
            cmds.append(cmd)
        return cmds

    variants = max(1, int(get_path(cfg, "scenegen.variants_per_layout", 1) or 1))

    def render_progress() -> tuple[int, int]:
        return _count(run_dir / "scenes"), p.scenes * variants

    def postprocess() -> list[list[str]]:
        mod = "screwsynth.postprocess.cli"
        cmds = [py_module(mod, "run", "--run", name, "--resume", *c)]
        first = _first_scene(run_dir / "scenes")
        if first:
            cmds.append(py_module(mod, "show", "--run", name, "--scene", first, *c))
        return cmds

    def post_progress() -> tuple[int, int]:
        return _count(run_dir / "processed"), max(1, _count(run_dir / "scenes"))

    def labels() -> list[list[str]]:
        mod = "screwsynth.labels.cli"
        ds = str(dataset_dir)
        cmds = [
            py_module(
                mod, "export", "--run", name, "--out", ds, "--formats", ",".join(p.formats), *c
            )
        ]
        if any(f in p.formats for f in ("seg", "pose", "det", "obb")):
            cmds.append(py_module(mod, "check", "--dataset", ds, *c))
            for split in ("train", "val"):
                cmds.append(py_module(mod, "preview", "--dataset", ds, "--split", split, *c))
        return cmds

    all_steps = {
        "catalogue": Step("catalogue", TITLES["catalogue"], catalogue),
        "parts": Step("parts", TITLES["parts"], parts, previews=[previews / "parts.png"]),
        "restposes": Step(
            "restposes", TITLES["restposes"], restposes, previews=[previews / "restposes.png"]
        ),
        "render": Step("render", TITLES["render"], render, progress=render_progress),
        "postprocess": Step(
            "postprocess",
            TITLES["postprocess"],
            postprocess,
            progress=post_progress,
            previews=[run_dir / "show_*.png"],  # glob: the scene name is known only later
        ),
        "labels": Step(
            "labels",
            TITLES["labels"],
            labels,
            clean=[dataset_dir],
            previews=[dataset_dir / "preview_train.jpg", dataset_dir / "preview_val.jpg"],
        ),
    }
    steps = [all_steps[k] for k in all_steps if k in p.steps]
    return Plan(recipe, cfg, config_path, run_dir, dataset_dir, steps)


def latest_render(run_dir: Path) -> Path | None:
    """The most recently rendered rgb.png of a run (for a live preview)."""
    scenes = run_dir / "scenes"
    if not scenes.is_dir():
        return None
    done = [
        d for d in scenes.iterdir() if (d / "meta.json").is_file() and (d / "rgb.png").is_file()
    ]
    return max(done, key=lambda d: (d / "meta.json").stat().st_mtime) / "rgb.png" if done else None


def _count(folder: Path) -> int:
    return sum(1 for d in folder.iterdir() if (d / "meta.json").is_file()) if folder.is_dir() else 0


def _first_scene(folder: Path) -> str | None:
    if not folder.is_dir():
        return None
    names = sorted(d.name for d in folder.iterdir() if (d / "meta.json").is_file())
    return names[0] if names else None


# ------------------------------------------------------------------------- preflight


@dataclass
class Check:
    level: str  # ok | warning | error
    message: str


def preflight(plan: Plan, min_free_gb: float = 6.0) -> list[Check]:
    """Problems worth knowing before a long run. Errors block the run, warnings do not."""
    out: list[Check] = []
    keys = {s.key for s in plan.steps}
    p = plan.recipe.pipeline
    cfg = plan.config
    if "render" in keys:
        exe = blenderproc_exe()
        out.append(
            Check("ok", f"BlenderProc: {exe}")
            if exe
            else Check(
                "error", "BlenderProc not found: pip install blenderproc (see README step 4)"
            )
        )
        if p.camera == "calibration":
            calib = resolve(get_path(cfg, "paths.calibration_dir", "data/calibration"))
            if (calib / "calibration.json").is_file():
                out.append(Check("ok", f"Calibration: {calib / 'calibration.json'}"))
            else:
                out.append(
                    Check(
                        "error",
                        f"camera 'calibration' but {calib / 'calibration.json'} does not exist: "
                        "calibrate first (tool 03) or choose the nominal camera",
                    )
                )
    free = free_commit_gb()
    if free is not None:
        msg = f"Free memory (RAM + page file): {free:.1f} GB"
        if free < min_free_gb and keys & {"render", "parts", "postprocess"}:
            out.append(
                Check(
                    "warning",
                    msg
                    + f" — below {min_free_gb:.0f} GB. Close large programs (Fusion 360, KiCad, "
                    "browsers); Blender and the worker processes can fail to allocate memory.",
                )
            )
        else:
            out.append(Check("ok", msg))
    if "catalogue" not in keys and not resolve(get_path(cfg, "paths.classes")).is_file():
        out.append(
            Check(
                "error", "classes.yaml of this run does not exist yet: include the catalogue step"
            )
        )
    stored = plan.run_dir / "config.yaml"
    if stored.is_file() and keys & {"render", "postprocess"}:
        from screwsynth.common.io import read_yaml

        old = read_yaml(stored) or {}
        changed = [
            k
            for k in ("scenegen", "postprocess", "catalogue", "world_scale", "seed")
            if old.get(k) != cfg.get(k)
        ]
        if changed and any((plan.run_dir / d).is_dir() for d in ("scenes", "processed")):
            out.append(
                Check(
                    "warning",
                    f"settings changed since this dataset was last run ({', '.join(changed)}): "
                    "existing scenes are kept and only missing ones are rendered. Tick 'start fresh' "
                    "(or --fresh) to redo everything with the new settings.",
                )
            )
    return out


def free_commit_gb() -> float | None:
    """Memory that can still be committed (Windows: RAM + page file), in GB."""
    if sys.platform == "win32":

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return None
        return st.ullAvailPageFile / 1024**3
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return info["MemAvailable"] / 1024**2
    except (OSError, KeyError, ValueError):
        return None


def describe(plan: Plan) -> str:
    """Human-readable list of every command (screwsynth-dataset plan)."""
    lines = [f"recipe {plan.recipe.name}: config {plan.config_path}", ""]
    for s in plan.steps:
        lines.append(f"[{s.key}] {s.title}")
        for cmd in s.commands():
            lines.append("    " + " ".join(_quote(a) for a in cmd))
    return "\n".join(lines)


def _quote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or not arg) else arg


def status_path(run_dir: Path) -> Path:
    return run_dir / "pipeline_status.json"


def read_status(run_dir: Path) -> dict[str, Any]:
    path = status_path(run_dir)
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
