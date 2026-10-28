"""Preview contact sheets and label checks for exported datasets."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screwsynth.common.io import read_json

from . import geometry as g

KP_COLOURS = [(0, 0, 255), (0, 200, 0), (255, 0, 0)]


def _colour(i: int) -> tuple[int, int, int]:
    rng = random.Random(i * 7919)
    h = rng.random()
    import colorsys

    r, gg, b = colorsys.hsv_to_rgb(h, 0.8, 1.0)
    return int(b * 255), int(gg * 255), int(r * 255)


def parse_line(line: str, fmt: str) -> dict[str, Any]:
    v = line.split()
    c = int(v[0])
    nums = np.array([float(x) for x in v[1:]])
    if fmt == "seg":
        return {"cls": c, "poly": nums.reshape(-1, 2)}
    if fmt == "det":
        return {"cls": c, "box": nums}
    if fmt == "obb":
        return {"cls": c, "obb": nums.reshape(4, 2)}
    if fmt == "pose":
        return {"cls": c, "box": nums[:4], "kps": nums[4:].reshape(-1, 3)}
    raise ValueError(fmt)


def draw(img: np.ndarray, labels: list[dict[str, Any]], names: dict[int, str]) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for i, lab in enumerate(labels):
        col = _colour(i)
        if "poly" in lab:
            pts = np.round(lab["poly"] * [w, h]).astype(np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], col)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            cv2.polylines(out, [pts], True, col, 1, cv2.LINE_AA)
            x0, y0 = pts.min(axis=0)
        if "obb" in lab:
            cv2.polylines(
                out, [np.round(lab["obb"] * [w, h]).astype(np.int32)], True, col, 1, cv2.LINE_AA
            )
        if "box" in lab:
            cx, cy, bw, bh = lab["box"] * [w, h, w, h]
            x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
            cv2.rectangle(out, (x0, y0), (int(cx + bw / 2), int(cy + bh / 2)), col, 1)
        if "kps" in lab:
            for k, (x, y, v) in enumerate(lab["kps"]):
                if v > 0:
                    cv2.circle(
                        out,
                        (int(round(x * w)), int(round(y * h))),
                        3,
                        KP_COLOURS[k % 3],
                        -1 if v == 2 else 1,
                    )
        cv2.putText(
            out,
            names.get(lab["cls"], str(lab["cls"])),
            (int(x0), max(10, int(y0) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            col,
            1,
            cv2.LINE_AA,
        )
    return out


def preview(dataset: Path, n: int, out: Path, split: str = "val", seed: int = 0) -> Path:
    exp = read_json(dataset / "export.json")
    names = {int(k): v for k, v in exp["names"].items()}
    fmt = "seg" if (dataset / "seg").is_dir() else "pose" if (dataset / "pose").is_dir() else "det"
    imgs = sorted((dataset / fmt / "images" / split).glob("*"))
    if not imgs:
        imgs = sorted((dataset / fmt / "images").rglob("*.*"))
    imgs = random.Random(seed).sample(imgs, min(n, len(imgs)))
    tiles = []
    pose_dir = dataset / "pose" / "labels"
    for p in imgs:
        img = cv2.imread(str(p))
        lab_path = dataset / fmt / "labels" / p.parent.name / (p.stem + ".txt")
        labels = [
            parse_line(line, fmt)
            for line in lab_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pose_path = pose_dir / p.parent.name / (p.stem + ".txt")
        if fmt != "pose" and pose_path.is_file():
            for lab, line in zip(
                labels, pose_path.read_text(encoding="utf-8").splitlines(), strict=False
            ):
                lab["kps"] = parse_line(line, "pose")["kps"]
        tile = draw(img, labels, names)
        tiles.append(cv2.resize(tile, (512, int(512 * tile.shape[0] / tile.shape[1]))))
    cols = 4
    th = max(t.shape[0] for t in tiles)
    rows = [
        np.hstack(
            [
                *tiles[r * cols : (r + 1) * cols],
                *[np.zeros((th, 512, 3), np.uint8)]
                * (cols - len(tiles[r * cols : (r + 1) * cols])),
            ]
        )
        for r in range((len(tiles) + cols - 1) // cols)
    ]
    sheet = np.vstack([np.pad(r, ((0, th - r.shape[0]), (0, 0), (0, 0))) for r in rows])
    cv2.imwrite(str(out), sheet)
    return out


def _outer_area(mask: np.ndarray) -> float:
    """Area enclosed by the outer contours of a mask (holes filled), in pixels."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return float(sum(cv2.contourArea(c) for c in contours))


def check(dataset: Path) -> list[str]:
    """Every label parses, class ids exist, coordinates in [0, 1], polygon area ~ mask area."""
    exp = read_json(dataset / "export.json")
    names = {int(k): v for k, v in exp["names"].items()}
    problems: list[str] = []
    expected_len = {"det": 5, "obb": 9, "pose": 14}
    for fmt in ("seg", "det", "obb", "pose"):
        for lab in (
            sorted((dataset / fmt / "labels").rglob("*.txt")) if (dataset / fmt).is_dir() else []
        ):
            for ln, line in enumerate(lab.read_text(encoding="utf-8").splitlines(), 1):
                v = line.split()
                where = f"{fmt}/{lab.parent.name}/{lab.name}:{ln}"
                try:
                    c = int(v[0])
                    nums = np.array([float(x) for x in v[1:]])
                except (ValueError, IndexError):
                    problems.append(f"{where}: unparsable")
                    continue
                if c not in names:
                    problems.append(f"{where}: unknown class {c}")
                if fmt in expected_len and len(v) != expected_len[fmt]:
                    problems.append(f"{where}: {len(v)} fields, expected {expected_len[fmt]}")
                if fmt == "seg" and (len(nums) < 6 or len(nums) % 2):
                    problems.append(f"{where}: polygon needs >= 3 points")
                if fmt == "pose":
                    vis = nums[4:].reshape(-1, 3)[:, 2]
                    if not np.isin(vis, [0, 1, 2]).all():
                        problems.append(f"{where}: keypoint visibility not in 0/1/2")
                    nums = np.r_[nums[:4], nums[4:].reshape(-1, 3)[:, :2].ravel()]
                if (nums < -1e-6).any() or (nums > 1 + 1e-6).any():
                    problems.append(f"{where}: coordinate outside [0, 1]")
    # polygon area vs mask area (full resolution) via sources.csv
    src = dataset / "sources.csv"
    if (dataset / "seg").is_dir() and src.is_file():
        for row in csv.DictReader(src.open(encoding="utf-8")):
            scene = Path(row["scene_dir"])
            if not (scene / "instance.png").is_file():
                continue
            w, h = int(row["width"]), int(row["height"])
            lab = dataset / "seg" / "labels" / (Path(row["image"]).with_suffix(".txt"))
            if not lab.is_file():
                continue
            mask = cv2.imread(str(scene / "instance.png"), cv2.IMREAD_UNCHANGED)
            ids = [i for i in np.unique(mask) if i]
            # filled outer-contour area: YOLO polygons have no holes, so a washer is a full disc
            areas = sorted(_outer_area(mask == i) for i in ids)
            polys = sorted(
                g.polygon_area(parse_line(line, "seg")["poly"] * [w, h])
                for line in lab.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            for pa in polys:
                if pa < 200:
                    continue
                best = min(areas, key=lambda a: abs(a - pa)) if areas else 0
                if best == 0 or not (0.8 <= pa / best <= 1.2):
                    problems.append(
                        f"seg/{lab.name}: polygon area {pa:.0f} px has no mask of similar area"
                    )
    return problems
