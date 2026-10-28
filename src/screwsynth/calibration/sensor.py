"""Sensor and lighting profile: noise model, flat-field, black level, empty-plate colour."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import Camera

CHANNELS = ("r", "g", "b")


def noise_model(
    frames: Sequence[np.ndarray], bins: int = 32, saturation: float = 250.0
) -> dict[str, Any]:
    """Fit variance = a * intensity + b per channel from repeated frames of a static scene."""
    stack = np.stack([f.astype(np.float32) for f in frames])  # (N, H, W, 3) BGR
    mean = stack.mean(axis=0)
    var = stack.var(axis=0, ddof=1)
    out = {}
    for ci, name in zip((2, 1, 0), CHANNELS, strict=True):
        m, v = mean[..., ci].ravel(), var[..., ci].ravel()
        keep = (m > 2) & (m < saturation)
        m, v = m[keep], v[keep]
        edges = np.linspace(m.min(), m.max(), bins + 1)
        idx = np.clip(np.digitize(m, edges) - 1, 0, bins - 1)
        xs, ys, ws = [], [], []
        for k in range(bins):
            sel = idx == k
            if sel.sum() >= 50:
                xs.append(float(np.median(m[sel])))
                ys.append(float(np.mean(v[sel])))  # mean: unbiased for the sample variance
                ws.append(float(sel.sum()))
        x, y, w = np.array(xs), np.array(ys), np.sqrt(ws)
        A = np.c_[x, np.ones_like(x)]
        (a, b), *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        pred = A @ np.array([a, b])
        r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        out[name] = {"a": float(a), "b": float(b), "r2": float(r2), "bins": {"mean": xs, "var": ys}}
    return out


def flat_field(white: np.ndarray, degree: int = 4, step: int = 8) -> np.ndarray:
    """Smooth luminance flat-field (low-order 2D polynomial), normalised to max 1."""
    gray = (
        cv2.cvtColor(white, cv2.COLOR_BGR2GRAY).astype(np.float64)
        if white.ndim == 3
        else white.astype(np.float64)
    )
    h, w = gray.shape
    small = cv2.resize(gray, (w // step, h // step), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0 : small.shape[0], 0 : small.shape[1]]
    xn = (xx + 0.5) * step / w * 2 - 1
    yn = (yy + 0.5) * step / h * 2 - 1
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    A = np.stack([xn.ravel() ** i * yn.ravel() ** j for i, j in terms], axis=1)
    coef, *_ = np.linalg.lstsq(A, small.ravel(), rcond=None)
    X, Y = np.meshgrid((np.arange(w) + 0.5) / w * 2 - 1, (np.arange(h) + 0.5) / h * 2 - 1)
    field = sum(c * X**i * Y**j for c, (i, j) in zip(coef, terms, strict=True))
    return np.clip(field / field.max(), 1e-3, 1.0)


def undistorted_roi(image: np.ndarray, cam: Camera, roi: Mapping[str, int]) -> np.ndarray:
    und = cv2.undistort(image, cam.K, cam.dist)
    return und[roi["y0"] : roi["y0"] + roi["h"], roi["x0"] : roi["x0"] + roi["w"]]


def colour_stats(image_bgr: np.ndarray) -> dict[str, Any]:
    px = image_bgr.reshape(-1, 3)[:, ::-1].astype(np.float64)
    return {
        "mean_rgb": px.mean(axis=0).round(3).tolist(),
        "std_rgb": px.std(axis=0).round(3).tolist(),
        "median_rgb": np.median(px, axis=0).tolist(),
    }


def build_profile(
    frames: Sequence[np.ndarray],
    white: np.ndarray,
    empty: np.ndarray,
    cam: Camera | None,
    roi: Mapping[str, int] | None,
    black_level: float | None = None,
    out_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return (profile dict, flat-field float image, undistorted empty-plate ROI crop)."""
    noise = noise_model(frames)
    ff = flat_field(white)
    plate = undistorted_roi(empty, cam, roi) if cam is not None and roi is not None else empty
    if black_level is None:
        black_level = float(np.percentile(np.stack(frames).min(axis=0), 0.1))
    profile = {
        "noise": {k: {kk: v[kk] for kk in ("a", "b", "r2")} for k, v in noise.items()},
        "noise_bins": {k: v["bins"] for k, v in noise.items()},
        "noise_model": "variance = a * intensity + b (8-bit units, per channel, as captured)",
        "black_level": black_level,
        "flatfield": {
            "file": "flatfield.png",
            "model": "2D polynomial degree 4 of the white-sheet luminance, normalised to max 1",
            "min": float(ff.min()),
            "encoding": "16-bit PNG, value / 65535",
        },
        "plate_empty": {
            "file": "plate_empty.png",
            "roi": dict(roi) if roi else None,
            **colour_stats(plate),
        },
        "frames": len(frames),
        "image_size": [int(white.shape[1]), int(white.shape[0])],
    }
    if out_dir is not None:
        cv2.imwrite(str(Path(out_dir) / "flatfield.png"), np.round(ff * 65535).astype(np.uint16))
        cv2.imwrite(str(Path(out_dir) / "plate_empty.png"), plate)
    return profile, ff, plate
