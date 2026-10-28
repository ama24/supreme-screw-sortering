"""Camera and sensor effects applied to undistorted renders (OpenCV + numpy).

Order (see process.py): lens distortion -> vignetting -> blur -> chromatic aberration -> exposure /
colour -> sensor noise -> compression. Images are uint8 RGB (H, W, 3) unless stated otherwise.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

# ------------------------------------------------------------------------------- distortion


@lru_cache(maxsize=8)
def _maps(K_key: tuple, dist_key: tuple, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    K = np.array(K_key, dtype=np.float64).reshape(3, 3)
    dist = np.array(dist_key, dtype=np.float64)
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(
        pts, K, dist, P=K, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9)
    )
    und = und.reshape(h, w, 2).astype(np.float32)
    return np.ascontiguousarray(und[..., 0]), np.ascontiguousarray(und[..., 1])


def distortion_maps(
    K: np.ndarray, dist: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-distortion remap: for every output (distorted) pixel, where to sample the render.

    The render is an ideal pinhole image with intrinsics K. Distortion acts on normalised
    coordinates, so the same coefficients apply to the ROI crop / scaled render with its own K.
    """
    return _maps(
        tuple(np.asarray(K, float).ravel()), tuple(np.asarray(dist, float).ravel()), tuple(size)
    )


def has_distortion(dist: Sequence[float]) -> bool:
    return bool(np.any(np.abs(np.asarray(dist, dtype=float)) > 0))


def distort_image(img: np.ndarray, K, dist, interpolation: str = "linear") -> np.ndarray:
    if not has_distortion(dist):
        return img.copy()
    h, w = img.shape[:2]
    mx, my = distortion_maps(K, dist, (w, h))
    interp = {"linear": cv2.INTER_LINEAR, "lanczos": cv2.INTER_LANCZOS4, "cubic": cv2.INTER_CUBIC}[
        interpolation
    ]
    return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)


def distort_mask(mask: np.ndarray, K, dist) -> np.ndarray:
    """Instance ids must never blend: nearest-neighbour only, outside -> background."""
    if not has_distortion(dist):
        return mask.copy()
    h, w = mask.shape[:2]
    mx, my = distortion_maps(K, dist, (w, h))
    return cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def distort_points(uv: np.ndarray, K, dist) -> np.ndarray:
    """Undistorted render pixels -> distorted pixels (OpenCV projection model)."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if not has_distortion(dist) or len(uv) == 0:
        return uv.copy()
    K = np.asarray(K, dtype=np.float64)
    xn = (uv - K[:2, 2]) / np.diag(K)[:2]
    pts = np.c_[xn, np.ones(len(xn))].reshape(-1, 1, 3)
    out, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, np.asarray(dist, dtype=np.float64))
    return out.reshape(-1, 2)


# ------------------------------------------------------------------------------------ optics


def radial_flatfield(size: tuple[int, int], k: float) -> np.ndarray:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    return np.clip(1.0 - k * r2, 0.05, 1.0)


def flatfield_for_render(
    full_flatfield: np.ndarray | None,
    roi: Mapping[str, int] | None,
    size: tuple[int, int],
    fallback_k: float = 0.25,
) -> np.ndarray:
    """Crop the calibrated full-sensor flat-field to the ROI and resample to the render size."""
    if full_flatfield is None or roi is None:
        return radial_flatfield(size, fallback_k)
    crop = full_flatfield[
        roi["y0"] : roi["y0"] + roi["h"], roi["x0"] : roi["x0"] + roi["w"]
    ].astype(np.float32)
    ff = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return ff / max(float(ff.max()), 1e-6)


def apply_vignetting(img: np.ndarray, flatfield: np.ndarray, strength: float) -> np.ndarray:
    f = 1.0 - strength * (1.0 - flatfield)
    return np.clip(img.astype(np.float32) * f[..., None], 0, 255)


def apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.astype(np.float32)
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)


def apply_chromatic_aberration(
    img: np.ndarray, scales: Sequence[float], centre: Sequence[float]
) -> np.ndarray:
    """Tiny per-channel magnification about the principal point (lateral CA)."""
    out = np.empty_like(img, dtype=np.float32)
    h, w = img.shape[:2]
    cx, cy = centre
    for c, s in enumerate(scales):
        M = np.array([[s, 0, (1 - s) * cx], [0, s, (1 - s) * cy]], dtype=np.float32)
        out[..., c] = cv2.warpAffine(
            img[..., c].astype(np.float32),
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return out


def apply_colour(img: np.ndarray, gains: Sequence[float], gamma: float) -> np.ndarray:
    x = np.clip(img.astype(np.float32) / 255.0 * np.asarray(gains, dtype=np.float32), 0, 1)
    return (x ** (1.0 / gamma)) * 255.0


# ------------------------------------------------------------------------------------- sensor


def apply_noise(
    img: np.ndarray,
    model: Mapping[str, Mapping[str, float]],
    scale: float,
    rng: np.random.Generator,
    fixed_pattern: np.ndarray | None = None,
) -> np.ndarray:
    """Signal-dependent noise, variance = a * I + b per channel (RGB order).

    The profile from tool 03 is measured on captured 8-bit images (after the camera's tone curve), so
    it is applied in those same units. To first order this equals adding the corresponding noise in
    linear space before the tone curve, without needing the (unknown) ISP curve.
    """
    x = img.astype(np.float32)
    out = np.empty_like(x)
    for c, ch in enumerate(("r", "g", "b")):
        a, b = float(model[ch]["a"]), float(model[ch]["b"])
        v = x[..., c]
        if fixed_pattern is not None:
            v = v * fixed_pattern[..., c]
        sd = np.sqrt(np.maximum(a * v + b, 0.0)) * scale
        out[..., c] = v + rng.standard_normal(v.shape).astype(np.float32) * sd
    return np.clip(out, 0, 255)


def fixed_pattern_map(size: tuple[int, int], strength: float, seed: int) -> np.ndarray:
    """Per-pixel gain (PRNU), fixed for a run."""
    w, h = size
    return (
        1.0 + np.random.default_rng(seed).standard_normal((h, w, 3)).astype(np.float32) * strength
    )


def encode(img: np.ndarray, fmt: str, quality: int = 95) -> bytes:
    bgr = cv2.cvtColor(np.clip(np.rint(img), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if fmt == "jpg":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    else:
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("image encoding failed")
    return buf.tobytes()


def uniform(rng: random.Random, r: Any) -> float:
    if isinstance(r, (list, tuple)):
        return rng.uniform(float(r[0]), float(r[1]))
    return float(r)
