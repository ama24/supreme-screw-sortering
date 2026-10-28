"""Fake processed scenes (image + 16-bit instance mask + meta.json) for label/train tests."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from screwsynth.common.io import write_json
from screwsynth.common.png import write_png

PX_PER_MM = 11.7


def rotated_rect(cx, cy, length, width, angle_deg):
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a)])
    n = np.array([-d[1], d[0]])
    c = np.array([cx, cy])
    return np.array(
        [
            c - d * length / 2 - n * width / 2,
            c + d * length / 2 - n * width / 2,
            c + d * length / 2 + n * width / 2,
            c - d * length / 2 + n * width / 2,
        ]
    )


def make_scene(
    root: Path,
    scene_id: str,
    layout_id: str | None = None,
    size=(640, 600),
    seed=0,
    n_objects=4,
    thin=False,
) -> Path:
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w, 3), (160, 90, 40), np.uint8)  # BGR plate
    mask = np.zeros((h, w), np.uint16)
    objects = []
    for k in range(1, n_objects + 1):
        cx, cy = rng.uniform(80, w - 80), rng.uniform(80, h - 80)
        length = rng.uniform(8, 25) * PX_PER_MM
        width = (2.0 if thin else rng.uniform(3, 6)) * PX_PER_MM
        ang = rng.uniform(0, 180)
        poly = rotated_rect(cx, cy, length, width, ang)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], k)
        cv2.fillPoly(img, [np.round(poly).astype(np.int32)], (200, 200, 205))
        a = math.radians(ang)
        d = np.array([math.cos(a), math.sin(a)])
        tip = np.array([cx, cy]) + d * length / 2 * 0.95
        head = np.array([cx, cy]) - d * length / 2 * 0.95
        bearing = head + d * width * 0.8
        is_screw = k % 4 != 0
        kp = lambda p, v: {"xyz": [0, 0, 0], "uv": [float(p[0]), float(p[1])], "visibility": v}  # noqa: E731
        objects.append(
            {
                "instance_id": k,
                "part_id": f"DIN912_M{2 + k}x{10 + k}" if is_screw else f"DIN934_M{2 + k}",
                "class_name": f"DIN912_M{2 + k}" if is_screw else f"DIN934_M{2 + k}",
                "class_id": k - 1,
                "kind": "screw" if is_screw else "nut",
                "family": "DIN912" if is_screw else "DIN934",
                "size": f"M{2 + k}",
                "length": float(10 + k) if is_screw else None,
                "finish": "zinc_clear",
                "keypoints": {
                    "head_top": kp(head, 2 if is_screw else 0),
                    "bearing": kp(bearing, 2 if is_screw else 0),
                    "tip": kp(tip, 2 if is_screw else 0),
                },
                "keypoints_labelled": is_screw,
                "measurable": is_screw,
                "axis_tilt_deg": 4.0,
                "axis_height_head_mm": 2.5,
                "axis_height_tip_mm": 1.5,
                "axis_height_head_top_mm": 2.4,
                "nominal_length_mode": "bearing_to_tip",
                "visible_pixel_fraction": 1.0,
                "alone_area_px": int(length * width),
                "mask_area_px": 0,
            }
        )
    for o in objects:
        o["mask_area_px"] = int((mask == o["instance_id"]).sum())
        o["visible_pixel_fraction"] = round(min(1.0, o["mask_area_px"] / o["alone_area_px"]), 4)
    d = root / scene_id
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "image.png"), img)
    write_png(d / "instance.png", mask)
    K = [[3400.0 * w / 2560, 0, (w - 1) / 2], [0, 3400.0 * w / 2560, (h - 1) / 2], [0, 0, 1]]
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1, -1])
    T[:3, 3] = -T[:3, :3] @ np.array([110.0, 110.0, 300.0])
    write_json(
        d / "meta.json",
        {
            "scene_id": scene_id,
            "layout_id": layout_id or scene_id,
            "image": "image.png",
            "camera": {
                "K": K,
                "dist_coeffs": [0, 0, 0, 0, 0],
                "roi": {"x0": 0, "y0": 0, "w": w, "h": h},
                "T_cam_plate": T.tolist(),
                "resolution": [w, h],
                "distorted": True,
            },
            "objects": objects,
        },
    )
    return d


def make_run(output_dir: Path, run: str, n: int = 10, variants: int = 1, **kw) -> list[Path]:
    proc = output_dir / run / "processed"
    out = []
    for i in range(n):
        for v in range(variants):
            sid = f"{i:06d}" if variants == 1 else f"{i:06d}_{v}"
            out.append(make_scene(proc, sid, f"{i:06d}", seed=i * 10 + v, **kw))
    return out
