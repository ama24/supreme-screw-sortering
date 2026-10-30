"""Training runs (Ultralytics), real/synthetic validation, fine-tuning, learning curve, ONNX."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from screwsynth.common.io import read_json, read_yaml, write_json, write_yaml

from .experiments import Experiment

log = logging.getLogger("screwsynth.train")

# Ultralytics metric keys (results_dict) reported, in order
SEG_KEYS = ("metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/mAP50-95(B)", "metrics/mAP50(B)")
POSE_KEYS = ("metrics/mAP50-95(P)", "metrics/mAP50(P)", "metrics/mAP50-95(B)")


def _yolo(weights: str | Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "training needs ultralytics + PyTorch: see README step 5 (pip install -e '.[train]')"
        ) from exc
    return YOLO(str(weights))


def record_run(run_dir: Path, exp: Experiment, extra: dict[str, Any] | None = None) -> None:
    """Experiment config, dataset stats and git commit / package versions next to the weights."""
    from screwsynth.pipeline.runner import write_environment

    run_dir.mkdir(parents=True, exist_ok=True)
    stats_path = exp.dataset / "stats.json"
    write_json(
        run_dir / "screwsynth_run.json",
        {
            "experiment": exp.to_dict(),
            "dataset_stats": read_json(stats_path) if stats_path.is_file() else None,
            **(extra or {}),
        },
    )
    write_environment(run_dir / "environment.txt")


def metrics_of(result: Any, stage: str) -> dict[str, float]:
    keys = SEG_KEYS if stage == "seg" else POSE_KEYS
    rd = getattr(result, "results_dict", {}) or {}
    return {k: round(float(rd[k]), 4) for k in keys if k in rd}


def validate(
    weights: Path, data_yaml: Path, exp: Experiment, split: str = "val"
) -> dict[str, float]:
    model = _yolo(weights)
    res = model.val(data=str(data_yaml), imgsz=exp.imgsz, split=split, plots=False, verbose=False)
    return metrics_of(res, exp.stage)


def real_and_synthetic(weights: Path, exp: Experiment) -> dict[str, dict[str, float]]:
    """Validation on the real set first (when the dataset has one: tool 09), then synthetic."""
    out: dict[str, dict[str, float]] = {}
    if exp.real_yaml.is_file():
        out["real"] = validate(weights, exp.real_yaml, exp)
    out["synthetic"] = validate(weights, exp.data_yaml, exp)
    return out


def report(metrics: dict[str, dict[str, float]]) -> str:
    lines = []
    for which in ("real", "synthetic"):
        if which in metrics:
            vals = ", ".join(f"{k.split('/')[-1]} {v:.3f}" for k, v in metrics[which].items())
            lines.append(f"{which:9s}: {vals}")
    if "real" not in metrics:
        lines.append("(no real validation set yet: build one with tool 09, export with --val-real)")
    return "\n".join(lines)


def train(exp: Experiment, project: Path, device: str | None = None, **overrides: Any) -> Path:
    """Train one experiment; returns the run directory (weights in weights/best.pt)."""
    if not exp.data_yaml.is_file():
        raise SystemExit(f"{exp.data_yaml} not found: export the dataset with tool 07 first")
    args = exp.ultralytics_args(project)
    if device is not None:
        args["device"] = device
    args.update(overrides)
    model = _yolo(exp.model)
    t0 = time.perf_counter()
    model.train(**args)
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    metrics = real_and_synthetic(best, exp) if best.is_file() else {}
    record_run(
        run_dir,
        exp,
        {
            "train_args": args,
            "train_minutes": round((time.perf_counter() - t0) / 60, 1),
            "metrics": metrics,
        },
    )
    log.info("trained %s in %s\n%s", exp.name, run_dir, report(metrics))
    return run_dir


def finetune(
    weights: Path,
    exp: Experiment,
    data_yaml: Path,
    holdout_yaml: Path,
    project: Path,
    freeze: int = 10,
    lr0: float = 0.0005,
    epochs: int = 30,
    device: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a synthetic model on a small real set with frozen early layers and a low
    learning rate; compares before/after on held-out real images (never used for training)."""
    before = validate(weights, holdout_yaml, exp)
    args = exp.ultralytics_args(project, data=data_yaml)
    args.update(
        name=f"{exp.name}_finetune",
        epochs=epochs,
        freeze=freeze,
        lr0=lr0,
        lrf=0.1,
        warmup_epochs=0,
        mosaic=0.5,
        close_mosaic=max(1, epochs // 5),
    )
    if device is not None:
        args["device"] = device
    model = _yolo(weights)
    model.train(**args)
    run_dir = Path(model.trainer.save_dir)
    after = validate(run_dir / "weights" / "best.pt", holdout_yaml, exp)
    out = {"weights_before": str(weights), "run": str(run_dir), "before": before, "after": after}
    record_run(run_dir, exp, {"finetune": out, "train_args": args})
    write_json(run_dir / "finetune_comparison.json", out)
    return out


# ---------------------------------------------------------------------------- learning curve


def subset_yaml(data_yaml: Path, fraction: float, out_dir: Path, seed: int = 0) -> Path:
    """Dataset yaml whose train split is `fraction` of the images, drawn by layout (the photos of
    one layout stay together), deterministic from the seed."""
    cfg = read_yaml(data_yaml)
    root = Path(cfg["path"])
    images = sorted((root / cfg["train"]).glob("*.*"))
    layouts: dict[str, list[Path]] = {}
    for im in images:
        layouts.setdefault(_layout_of(im.stem), []).append(im)
    keys = sorted(layouts)
    random.Random(seed).shuffle(keys)
    n = max(1, round(len(keys) * fraction))
    chosen = sorted(p for k in keys[:n] for p in layouts[k])
    out_dir.mkdir(parents=True, exist_ok=True)
    lst = out_dir / f"train_{int(round(fraction * 100))}.txt"
    lst.write_text("\n".join(p.as_posix() for p in chosen) + "\n", encoding="utf-8")
    sub = dict(cfg)
    sub["train"] = lst.as_posix()
    sub["val"] = (root / cfg["val"]).as_posix()
    sub["path"] = root.as_posix()
    out = out_dir / f"{data_yaml.stem}_{int(round(fraction * 100))}.yaml"
    write_yaml(out, sub)
    return out


def _layout_of(stem: str) -> str:
    """Image stem -> layout: <run>_<layout>[_r<k>][_<variant>] (tool 05 / 07 naming)."""
    parts = stem.split("_")
    while len(parts) > 1 and (
        parts[-1].startswith("r")
        and parts[-1][1:].isdigit()
        or parts[-1].isdigit()
        and len(parts[-1]) < 3
    ):
        parts.pop()
    return "_".join(parts)


def learning_curve(
    exp: Experiment,
    project: Path,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
    epochs: int = 30,
    device: str | None = None,
) -> dict[str, Any]:
    """Train on growing shares of the data (short schedules) and plot mAP against the size."""
    out_dir = project / f"{exp.name}_curve"
    points = []
    for f in fractions:
        data = subset_yaml(exp.data_yaml, f, out_dir / "subsets", exp.seed)
        n_images = sum(
            1 for _ in Path(read_yaml(data)["train"]).read_text(encoding="utf-8").splitlines()
        )
        args = exp.ultralytics_args(out_dir, data=data)
        args.update(name=f"frac_{int(round(f * 100))}", epochs=epochs, patience=epochs)
        if device is not None:
            args["device"] = device
        model = _yolo(exp.model)
        model.train(**args)
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        m = real_and_synthetic(best, exp)
        points.append({"fraction": f, "train_images": n_images, **m})
        log.info("curve %.0f%% (%d images): %s", f * 100, n_images, json.dumps(m))
    result = {"experiment": exp.name, "epochs": epochs, "points": points}
    write_json(out_dir / "curve.json", result)
    plot_curve(result, out_dir / "curve.png", exp.stage)
    return result


def plot_curve(result: dict[str, Any], path: Path, stage: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = (SEG_KEYS if stage == "seg" else POSE_KEYS)[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for which, style in (("real", "o-"), ("synthetic", "s--")):
        xs = [p["train_images"] for p in result["points"] if which in p]
        ys = [p[which].get(key, float("nan")) for p in result["points"] if which in p]
        if xs:
            ax.plot(xs, ys, style, label=f"{which} val")
    ax.set_xscale("log")
    ax.set_xlabel("training images")
    ax.set_ylabel(key.split("/")[-1])
    ax.set_title(f"{result['experiment']}: more renders worth it?")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------------------------- ONNX


def export_onnx(weights: Path, imgsz: int = 1024, runs: int = 10) -> dict[str, Any]:
    """Export to ONNX and time CPU inference (does it fit on the Pi, or must it run on the PC?)."""
    import numpy as np

    model = _yolo(weights)
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False))
    out: dict[str, Any] = {"onnx": str(onnx_path), "imgsz": imgsz}
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        x = np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32)
        sess.run(None, {inp.name: x})  # warm-up
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            sess.run(None, {inp.name: x})
            times.append((time.perf_counter() - t) * 1000)
        out["cpu_ms_mean"] = round(float(np.mean(times)), 1)
        out["cpu_ms_min"] = round(float(np.min(times)), 1)
        # a Raspberry Pi 5 runs such CPU inference roughly 5-10x slower than a desktop i7
        out["pi5_ms_estimate"] = [round(out["cpu_ms_mean"] * 5), round(out["cpu_ms_mean"] * 10)]
    except ImportError:
        out["note"] = "install onnxruntime to time CPU inference (pip install onnxruntime)"
    write_json(onnx_path.with_suffix(".latency.json"), out)
    return out
