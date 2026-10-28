"""Intrinsic calibration from ChArUco views (standard 5-coefficient or rational 8-coefficient)."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .board import BoardSpec, Detection, detect

log = logging.getLogger(__name__)

MODELS = {"standard5": 0, "rational8": cv2.CALIB_RATIONAL_MODEL}


@dataclass
class View:
    file: str
    detection: Detection | None
    used: bool = False
    reason: str = ""
    rms: float | None = None


@dataclass
class IntrinsicResult:
    K: np.ndarray
    dist: np.ndarray
    model: str
    rms: float
    holdout_rms: dict[str, float]
    image_size: tuple[int, int]
    views: list[View] = field(default_factory=list)
    residuals: np.ndarray | None = None  # (N, 4): x, y, dx, dy over all used corners

    def to_json(self) -> dict[str, Any]:
        return {
            "camera_matrix": self.K.tolist(),
            "dist_coeffs": self.dist.ravel().tolist(),
            "distortion_model": self.model,
            "rms_px": self.rms,
            "holdout_rms_px": self.holdout_rms,
            "image_size": list(self.image_size),
            "views": [
                {
                    "file": v.file,
                    "used": v.used,
                    "reason": v.reason,
                    "rms_px": v.rms,
                    "corners": 0 if v.detection is None else v.detection.n,
                    "sharpness": None if v.detection is None else round(v.detection.sharpness, 2),
                }
                for v in self.views
            ],
        }


def load_views(
    paths: Sequence[str | os.PathLike[str]], spec: BoardSpec
) -> tuple[list[View], tuple[int, int]]:
    views, size = [], None
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            views.append(View(str(p), None, reason="unreadable"))
            continue
        h, w = img.shape
        if size is None:
            size = (w, h)
        elif size != (w, h):
            views.append(View(str(p), None, reason=f"size {w}x{h} differs from {size}"))
            continue
        views.append(View(Path(p).name, detect(img, spec)))
    if size is None:
        raise ValueError("no readable images")
    return views, size


def select_views(views: list[View], min_corners: int, blur_ratio: float) -> list[View]:
    """Reject views without enough corners or much blurrier than the median view."""
    sharp = [v.detection.sharpness for v in views if v.detection is not None]
    median = float(np.median(sharp)) if sharp else 0.0
    for v in views:
        if v.reason:
            continue
        if v.detection is None:
            v.reason = "board not found"
        elif v.detection.n < min_corners:
            v.reason = f"only {v.detection.n} corners"
        elif v.detection.sharpness < blur_ratio * median:
            v.reason = (
                f"blurry (sharpness {v.detection.sharpness:.0f} < {blur_ratio:.0%} of median)"
            )
        else:
            v.used = True
    return [v for v in views if v.used]


def _calibrate(views: list[View], size: tuple[int, int], flags: int):
    obj = [v.detection.object_points.astype(np.float32) for v in views]
    img = [v.detection.corners.astype(np.float32) for v in views]
    return cv2.calibrateCamera(obj, img, size, None, None, flags=flags)


def _pnp_rms(view: View, K: np.ndarray, dist: np.ndarray) -> float:
    d = view.detection
    ok, rvec, tvec = cv2.solvePnP(d.object_points, d.corners, K, dist)
    proj, _ = cv2.projectPoints(d.object_points, rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - d.corners) ** 2, axis=1))))


def holdout_error(views: list[View], size: tuple[int, int], flags: int, folds: int = 5) -> float:
    """k-fold: calibrate without a fold, measure reprojection RMS on it (pose by solvePnP)."""
    k = min(folds, len(views))
    errs = []
    for f in range(k):
        test = views[f::k]
        train = [v for i, v in enumerate(views) if i % k != f]
        if len(train) < 3:
            continue
        _, K, dist, _, _ = _calibrate(train, size, flags)
        errs += [_pnp_rms(v, K, dist) for v in test]
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def calibrate(
    paths: Sequence[str | os.PathLike[str]],
    spec: BoardSpec,
    model: str = "auto",
    min_corners: int = 12,
    blur_ratio: float = 0.4,
    rational_gain: float = 0.05,
) -> IntrinsicResult:
    views, size = load_views(paths, spec)
    used = select_views(views, min_corners, blur_ratio)
    if len(used) < 4:
        raise ValueError(f"only {len(used)} usable views (need >= 4; aim for 25-40)")
    holdout = {name: holdout_error(used, size, flags) for name, flags in MODELS.items()}
    if model == "auto":
        model = "standard5"
        if holdout["rational8"] < (1 - rational_gain) * holdout["standard5"]:
            model = "rational8"
    rms, K, dist, rvecs, tvecs = _calibrate(used, size, MODELS[model])
    residuals = []
    for v, r, t in zip(used, rvecs, tvecs, strict=True):
        proj, _ = cv2.projectPoints(v.detection.object_points, r, t, K, dist)
        diff = proj.reshape(-1, 2) - v.detection.corners
        v.rms = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        residuals.append(np.c_[v.detection.corners, diff])
    log.info(
        "intrinsics: %d/%d views, model %s, RMS %.3f px (holdout %s)",
        len(used),
        len(views),
        model,
        rms,
        {k: round(x, 3) for k, x in holdout.items()},
    )
    return IntrinsicResult(
        K, dist.ravel(), model, float(rms), holdout, size, views, np.vstack(residuals)
    )
