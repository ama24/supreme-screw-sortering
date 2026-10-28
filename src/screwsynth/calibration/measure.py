"""Measurement validation and drift recheck."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from .board import BoardSpec
from .extrinsic import board_pose, plate_from_board
from .geometry import Camera


def segment_objects(image: np.ndarray, min_area_px: int = 400) -> list[np.ndarray]:
    """Contours of objects that differ in colour from the (dominant) plate.

    The colour distance to the median (plate) colour is thresholded with Otsu, i.e. roughly at half
    contrast, so blurred edges are split where the true edge is.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    med = np.median(lab.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(lab - med, axis=2)
    d8 = np.clip(dist * 255.0 / max(float(dist.max()), 1e-6), 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(12.0, otsu * max(float(dist.max()), 1e-6) / 255.0)
    mask = (dist > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = mask.shape
    keep = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cv2.contourArea(c) >= min_area_px and x > 0 and y > 0 and x + cw < w and y + ch < h:
            keep.append(c.reshape(-1, 2).astype(float))
    return keep


def measure_contour(contour_px: np.ndarray, cam: Camera, height_mm: float = 0.0) -> dict[str, Any]:
    """Length/width (mm) of an outline mapped to the plane z = height (undistortion + rays)."""
    pts = cam.pixel_to_plate(contour_px, height_mm)[:, :2].astype(np.float32)
    (cx, cy), (a, b), ang = cv2.minAreaRect(pts)
    # the contour runs through the centres of the outermost pixels: add one pixel per dimension
    px_mm = 1.0 / cam.px_per_mm((float(cx), float(cy)))
    length, width = max(a, b) + px_mm, min(a, b) + px_mm
    if a < b:
        ang += 90
    return {
        "centre_mm": [float(cx), float(cy)],
        "length_mm": float(length),
        "width_mm": float(width),
        "angle_deg": float(ang % 180),
    }


def validate(
    image: np.ndarray,
    cam: Camera,
    true_lengths: Sequence[float],
    height_mm: float = 0.0,
    order: str = "size",
) -> dict[str, Any]:
    """Measure every object and compare with caliper lengths (matched by size or left-to-right)."""
    objs = [measure_contour(c, cam, height_mm) for c in segment_objects(image)]
    if len(objs) != len(true_lengths):
        raise ValueError(f"found {len(objs)} objects but {len(true_lengths)} lengths were given")
    if order == "x":
        objs.sort(key=lambda o: o["centre_mm"][0])
        truth = list(true_lengths)
    else:
        objs.sort(key=lambda o: o["length_mm"])
        truth = sorted(true_lengths)
    rows = []
    for o, t in zip(objs, truth, strict=True):
        rows.append({**o, "true_mm": float(t), "error_mm": o["length_mm"] - float(t)})
    errs = np.array([r["error_mm"] for r in rows])
    return {
        "objects": rows,
        "mean_abs_error_mm": float(np.abs(errs).mean()) if len(errs) else 0.0,
        "max_abs_error_mm": float(np.abs(errs).max()) if len(errs) else 0.0,
        "height_mm": height_mm,
    }


def recheck(
    image: np.ndarray,
    spec: BoardSpec,
    calib: Mapping[str, Any],
    max_tilt_deg: float = 0.15,
    max_height_mm: float = 0.3,
) -> dict[str, Any]:
    """Compare the plate plane seen now (board flat anywhere on the plate) with the calibration."""
    cam = Camera.from_calibration(calib)
    thickness = float(calib["extrinsic"].get("board_thickness_mm", 0.0))
    T_cb, rms, n = board_pose(image, spec, cam)
    T_now = plate_from_board(T_cb, thickness)
    n_ref = cam.R[:, 2]
    n_now = T_now[:3, 2]
    tilt = math.degrees(math.acos(float(np.clip(n_ref @ n_now, -1, 1))))
    h_ref = float(cam.centre_in_plate[2])
    h_now = float((-T_now[:3, :3].T @ T_now[:3, 3])[2])
    dh = h_now - h_ref
    ok = tilt <= max_tilt_deg and abs(dh) <= max_height_mm
    return {
        "plane_tilt_change_deg": tilt,
        "camera_height_change_mm": dh,
        "reprojection_rms_px": rms,
        "corners": n,
        "recalibrate_extrinsic": not ok,
        "thresholds": {"max_tilt_deg": max_tilt_deg, "max_height_mm": max_height_mm},
    }
