"""Evaluation: detection/segmentation mAP, size confusion, length accuracy, worst cases, and the
comparison of the two measurement methods (A keypoints vs B classical) on the true masks."""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screwsynth.common.io import read_json, read_yaml, write_json, write_text

from .geometry import Geometry, camera_from_meta, measure_length, snap
from .infer import Detection, Pipeline, Scene, Settings, measure_screw, run_pose, run_stage1

log = logging.getLogger("screwsynth.eval")


def scene_from_dir(scene_dir: Path) -> tuple[Scene, dict[str, Any], np.ndarray]:
    """A processed scene (tool 06) or a real capture (tool 09): image, camera, meta, instance mask."""
    meta = read_json(scene_dir / "meta.json")
    img = cv2.imread(str(scene_dir / meta.get("image", "image.png")), cv2.IMREAD_COLOR)
    cam = camera_from_meta(meta["camera"])
    roi = dict(meta["camera"].get("roi") or {})
    inst = cv2.imread(
        str(scene_dir / meta.get("instance_mask", "instance.png")), cv2.IMREAD_UNCHANGED
    )
    return Scene(img, cam, roi, meta["camera"].get("source", "scene")), meta, inst


def _gt_detection(obj: Mapping[str, Any], inst: np.ndarray) -> Detection | None:
    m = (inst == obj["instance_id"]).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    x0, y0, w, h = cv2.boundingRect(m)
    return Detection(
        int(obj.get("class_id", -1)), obj["class_name"], 1.0, c, (x0, y0, x0 + w - 1, y0 + h - 1)
    )


def _gt_mask_window(
    obj: Mapping[str, Any], inst: np.ndarray, margin: int = 8
) -> tuple[np.ndarray, int, int]:
    ys, xs = np.nonzero(inst == obj["instance_id"])
    h, w = inst.shape
    x0, y0 = max(0, xs.min() - margin), max(0, ys.min() - margin)
    x1, y1 = min(w, xs.max() + margin + 1), min(h, ys.max() + margin + 1)
    return inst[y0:y1, x0:x1] == obj["instance_id"], int(x0), int(y0)


def _summary(errors: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    vals = [e for e in errors if e.get(key) is not None]
    if not vals:
        return {"n": 0}
    err = np.array([e[key] - e["true_mm"] for e in vals])
    exact = sum(1 for e in vals if e.get(key + "_snapped") == e["true_mm"])
    return {
        "n": len(vals),
        "exact_rate": round(exact / len(vals), 4),
        "mean_abs_mm": round(float(np.abs(err).mean()), 4),
        "max_abs_mm": round(float(np.abs(err).max()), 4),
        "bias_mm": round(float(err.mean()), 4),
    }


def _by(errors: Sequence[Mapping[str, Any]], field: str, key: str) -> dict[str, Any]:
    groups: dict[str, list] = defaultdict(list)
    for e in errors:
        groups[str(e[field])].append(e)
    return {g: _summary(v, key) for g, v in sorted(groups.items())}


# ------------------------------------------------------------------ method A vs B (true masks)


def compare_measurement(
    dataset: Path,
    geo: Geometry,
    split: str = "val",
    stage2: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Measure every measurable screw of the dataset's crops.csv with:
    A_true: the true keypoints (the best a keypoint model can do: the geometry alone),
    A: the stage-2 keypoint model (if weights are given), B: the classical method on the true mask.
    """
    st = settings or Settings()
    model = None
    if stage2 is not None:
        from ultralytics import YOLO

        model = YOLO(str(stage2))
    rows = [
        r
        for r in csv.DictReader((dataset / "crops.csv").open(encoding="utf-8"))
        if split in ("all", r["split"])
    ]
    by_scene: dict[str, list] = defaultdict(list)
    for r in rows:
        by_scene[r["scene_dir"]].append(r)
    errors = []
    for scene_dir, rs in by_scene.items():
        scene, meta, inst = scene_from_dir(Path(scene_dir))
        objs = {o["instance_id"]: o for o in meta["objects"]}
        for r in rs:
            o = objs.get(int(r["instance_id"]))
            if o is None or r["class_name"] not in geo:
                continue
            g = geo[r["class_name"]]
            true_mm = float(r["length_mm"])
            e: dict[str, Any] = {
                "scene": Path(scene_dir).name,
                "instance_id": o["instance_id"],
                "class": g.class_name,
                "family": g.family,
                "size": g.size,
                "true_mm": true_mm,
            }
            kp = o["keypoints"]
            a_true, _, _ = measure_length(scene.cam, g, kp[g.anchor]["uv"], kp["tip"]["uv"])
            e["A_true"] = a_true
            e["A_true_snapped"] = snap(a_true, g.lengths)[0]
            window = _gt_mask_window(o, inst)
            det = _gt_detection(o, inst)
            if det is not None:
                res = measure_screw(scene, det, g, st, mask=window)
                e["B"] = res.get("length_classical_mm")
                e["B_snapped"] = snap(e["B"], g.lengths)[0] if e["B"] is not None else None
                if model is not None:
                    pose = run_pose(model, scene, det, st)
                    if pose is not None:
                        a, _, _ = measure_length(scene.cam, g, pose[0][g.anchor], pose[0]["tip"])
                        e["A"] = a
                        e["A_snapped"] = snap(a, g.lengths)[0]
            errors.append(e)
    methods = ["A_true", "B"] + (["A"] if model is not None else [])
    return {
        "dataset": str(dataset),
        "split": split,
        "screws": len(errors),
        "methods": {m: _summary(errors, m) for m in methods},
        "by_family": {m: _by(errors, "family", m) for m in methods},
        "by_size": {m: _by(errors, "size", m) for m in methods},
        "worst": {
            m: sorted(
                (e for e in errors if e.get(m) is not None), key=lambda e: -abs(e[m] - e["true_mm"])
            )[:10]
            for m in methods
        },
    }


def comparison_markdown(res: Mapping[str, Any]) -> str:
    names = {
        "A_true": "A with the true keypoints (geometry only)",
        "A": "A keypoint model",
        "B": "B classical (true mask)",
    }
    lines = [
        f"# Measurement methods: {res['dataset']} ({res['split']}, {res['screws']} measurable screws)",
        "",
        "| Method | n | exact after snapping | mean abs error | max abs error | bias |",
        "|---|---|---|---|---|---|",
    ]
    for m, s in res["methods"].items():
        if s.get("n"):
            lines.append(
                f"| {names.get(m, m)} | {s['n']} | {s['exact_rate']:.1%} | {s['mean_abs_mm']:.3f} mm | "
                f"{s['max_abs_mm']:.3f} mm | {s['bias_mm']:+.3f} mm |"
            )
    lines += ["", "## By family", ""]
    for m, fams in res["by_family"].items():
        lines.append(
            f"**{names.get(m, m)}**: "
            + ", ".join(
                f"{f} {s['exact_rate']:.0%} ({s['n']})" for f, s in fams.items() if s.get("n")
            )
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- full evaluation


def detection_metrics(stage1: Path, dataset: Path, split: str, imgsz: int) -> dict[str, Any]:
    """mAP per class and the size confusion within each family (Ultralytics val)."""
    from ultralytics import YOLO

    data = dataset / "seg.yaml"
    model = YOLO(str(stage1))
    res = model.val(data=str(data), imgsz=imgsz, split=split, plots=True, verbose=False)
    names = {int(k): v for k, v in read_yaml(data)["names"].items()}
    out: dict[str, Any] = {
        "map50_95_mask": round(float(res.seg.map), 4),
        "map50_mask": round(float(res.seg.map50), 4),
        "map50_95_box": round(float(res.box.map), 4),
        "per_class_map50_95_mask": {
            names[i]: round(float(v), 4) for i, v in enumerate(res.seg.maps) if i in names
        },
    }
    cm = getattr(getattr(res, "confusion_matrix", None), "matrix", None)
    if cm is not None:
        out["size_confusion"] = size_confusion(np.asarray(cm), names)
    return out


def size_confusion(
    matrix: np.ndarray, names: Mapping[int, str]
) -> dict[str, dict[str, dict[str, int]]]:
    """Ultralytics confusion matrix (predicted x true) -> per family: true size -> predicted size."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    nc = len(names)
    for t in range(nc):
        for p in range(nc):
            n = int(round(matrix[p, t]))
            if n == 0:
                continue
            ft, st = _family_size(names[t])
            fp, sp = _family_size(names[p])
            if ft != fp or not ft:
                continue
            out.setdefault(ft, {}).setdefault(st, {})[sp] = (
                out.get(ft, {}).get(st, {}).get(sp, 0) + n
            )
    return out


def _family_size(cls: str) -> tuple[str, str]:
    if "_M" not in cls:
        return "", cls
    fam, size = cls.rsplit("_", 1)
    return fam, size


def length_accuracy(
    pipeline: Pipeline, dataset: Path, split: str, out_dir: Path, iou_min: float = 0.5
) -> dict[str, Any]:
    """Run the full pipeline on the full-resolution scenes; match detections to the true screws
    by mask IoU; compare the measured lengths with the true ones."""
    rows = [
        r
        for r in csv.DictReader((dataset / "crops.csv").open(encoding="utf-8"))
        if split in ("all", r["split"])
    ]
    by_scene: dict[str, list] = defaultdict(list)
    for r in rows:
        by_scene[r["scene_dir"]].append(r)
    errors = []
    worst_dir = out_dir / "worst"
    for scene_dir, rs in by_scene.items():
        scene, meta, inst = scene_from_dir(Path(scene_dir))
        dets = run_stage1(pipeline.stage1, scene, pipeline.names, pipeline.st)
        objs = {o["instance_id"]: o for o in meta["objects"]}
        for r in rs:
            o = objs.get(int(r["instance_id"]))
            if o is None:
                continue
            gt = inst == o["instance_id"]
            best, best_iou = None, 0.0
            for d in dets:
                m, x0, y0 = d.mask_window(inst.shape)
                full = np.zeros_like(gt)
                full[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]] = m
                inter = np.logical_and(full, gt).sum()
                iou = inter / max(1, np.logical_or(full, gt).sum())
                if iou > best_iou:
                    best, best_iou = d, iou
            e: dict[str, Any] = {
                "scene": Path(scene_dir).name,
                "scene_dir": scene_dir,
                "instance_id": o["instance_id"],
                "true_class": r["class_name"],
                "family": r["class_name"].rsplit("_", 1)[0],
                "size": r["size"],
                "true_mm": float(r["length_mm"]),
                "detected": best is not None and best_iou >= iou_min,
            }
            if e["detected"]:
                e["class"] = best.class_name
                g = pipeline.geo.classes.get(best.class_name)
                if g is not None:
                    pose = (
                        run_pose(pipeline.stage2, scene, best, pipeline.st)
                        if pipeline.stage2
                        else None
                    )
                    res = measure_screw(scene, best, g, pipeline.st, pose)
                    if res.get("length"):
                        e["pred"] = res["length"]["length_raw_mm"]
                        e["pred_snapped"] = res["length"]["length_mm"]
                        e["method"] = res["length"]["method"]
                        e["confidence"] = res["length"]["confidence"]
            errors.append(e)
    measured = [e for e in errors if "pred" in e]
    worst = sorted(measured, key=lambda e: -abs(e["pred"] - e["true_mm"]))[:12]
    for k, e in enumerate(worst):
        _save_case(
            Path(e["scene_dir"]), e, worst_dir / f"{k:02d}_{e['scene']}_{e['instance_id']}.png"
        )
    return {
        "screws": len(errors),
        "detected": sum(e["detected"] for e in errors),
        "class_correct": sum(1 for e in errors if e.get("class") == e["true_class"]),
        "overall": _summary(measured, "pred"),
        "by_family": _by(measured, "family", "pred"),
        "by_size": _by(measured, "size", "pred"),
        "worst": worst,
    }


def _save_case(scene_dir: Path, e: Mapping[str, Any], path: Path) -> None:
    try:
        scene, meta, inst = scene_from_dir(scene_dir)
    except (OSError, KeyError):
        return
    ys, xs = np.nonzero(inst == e["instance_id"])
    if len(xs) == 0:
        return
    pad = 40
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    crop = scene.image[y0 : ys.max() + pad, x0 : xs.max() + pad].copy()
    cv2.putText(
        crop,
        f"true {e['true_mm']:g} / measured {e.get('pred', float('nan')):.2f}",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)


def evaluation_markdown(
    name: str, res: Mapping[str, Any], other: Mapping[str, Any] | None = None
) -> str:
    def row(label: str, a: Any, b: Any = None) -> str:
        return f"| {label} | {a} |" + (f" {b} |" if other is not None else "")

    det = res.get("detection", {})
    ln = res.get("length", {}).get("overall", {})
    od = (other or {}).get("detection", {})
    ol = (other or {}).get("length", {}).get("overall", {})
    head = f"| | {name} |" + (f" {other.get('name')} |" if other else "")
    sep = "|---|---|" + ("---|" if other else "")
    lines = [f"# Evaluation: {name}", "", head, sep]
    lines.append(row("mask mAP50-95", det.get("map50_95_mask"), od.get("map50_95_mask")))
    lines.append(row("mask mAP50", det.get("map50_mask"), od.get("map50_mask")))
    lines.append(
        row("length exact after snapping", _pct(ln.get("exact_rate")), _pct(ol.get("exact_rate")))
    )
    lines.append(row("length mean abs error (mm)", ln.get("mean_abs_mm"), ol.get("mean_abs_mm")))
    lines.append(row("length max abs error (mm)", ln.get("max_abs_mm"), ol.get("max_abs_mm")))
    if other is not None:
        lines += ["", f"The gap between {name} and {other.get('name')} is the sim-to-real gap."]
    worst = res.get("length", {}).get("worst", [])
    if worst:
        lines += [
            "",
            "## Worst length errors",
            "",
            "| scene | id | class | true | measured | method |",
            "|---|---|---|---|---|---|",
        ]
        for e in worst:
            lines.append(
                f"| {e['scene']} | {e['instance_id']} | {e.get('class')} | {e['true_mm']:g} | {e['pred']:.2f} | {e.get('method')} |"
            )
    return "\n".join(lines) + "\n"


def _pct(x: Any) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"


def write_report(
    out_dir: Path, name: str, res: dict[str, Any], other: dict[str, Any] | None = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "evaluation.json", {"result": res, "compare_with": other})
    write_text(out_dir / "report.md", evaluation_markdown(name, res, other))
