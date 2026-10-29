"""Method B: classical keypoints from the mask and sub-pixel edges (no keypoint model).

1. The screw axis is the principal axis of its mask; the wider end is the head.
2. Coarse ends (tip, head top) and the bearing (where the width steps from head to shank) come
   from the mask's width profile along the axis.
3. Each is refined to sub-pixel precision on the full-resolution image: the strongest intensity
   edge along lines parallel to the axis, with a parabolic peak fit.
4. Rim correction: the outermost visible point of an end face is its top rim (end pointing away
   from the camera's nadir) or bottom rim (pointing towards it), not the axis point. That rim
   point is intersected with its own height plane and moved down/up to the axis height, then
   projected back, so the keypoints mean the same as the keypoint model's (points on the axis).

All pixel coordinates are in the image passed in (full-resolution ROI, distorted like the camera).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from screwsynth.calibration.geometry import Camera

from .geometry import ClassGeometry, measure_length


@dataclass
class EdgeKeypoints:
    """2D result: keypoints (px), axis direction (head -> tip) and shank widths (px)."""

    keypoints: dict[str, np.ndarray]
    axis: np.ndarray
    shank_widths_px: list[float]
    head_width_px: float
    notes: list[str] = field(default_factory=list)


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x0 + 1] * fx * (1 - fy)
        + img[y0 + 1, x0] * (1 - fx) * fy
        + img[y0 + 1, x0 + 1] * fx * fy
    )


def _edge_along(
    gray: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    s0: float,
    half: float = 4.0,
    step: float = 0.25,
) -> float | None:
    """Sub-pixel position s (along u from origin) of the strongest edge within s0 +- half."""
    s = np.arange(s0 - half, s0 + half + step / 2, step)
    pts = origin[None, :] + s[:, None] * u[None, :]
    prof = bilinear(gray, pts[:, 0], pts[:, 1])
    grad = np.abs(np.gradient(prof, step))
    i = int(np.argmax(grad))
    if grad[i] <= 1e-6:
        return None
    if 0 < i < len(grad) - 1:  # parabola through the peak and its neighbours
        a, b, c = grad[i - 1], grad[i], grad[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        return float(s[i] + np.clip(off, -1, 1) * step)
    return float(s[i])


def edge_keypoints(
    gray: np.ndarray, mask: np.ndarray, head_shape: str = "flat"
) -> EdgeKeypoints | None:
    """Keypoints of a lying screw in image coordinates from its mask and the image edges.

    head_shape: "flat" (socket cap, hex, countersunk top) or "dome" (button head: the top is a
    point on the axis, so only the centre line is used).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    pts = np.c_[xs, ys].astype(float)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, 1] / np.linalg.norm(evecs[:, 1])
    v = np.array([-u[1], u[0]])
    s = (pts - c) @ u
    t = (pts - c) @ v
    smin, smax = s.min(), s.max()
    n_bins = max(8, int(math.ceil(smax - smin)) + 1)
    edges = np.linspace(smin, smax + 1e-6, n_bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    width = np.zeros(n_bins)
    for b in range(n_bins):
        tb = t[idx == b]
        width[b] = tb.max() - tb.min() + 1 if len(tb) else 0
    k = max(2, n_bins // 4)  # head region: the head is never longer than a quarter of the screw
    if np.percentile(width[:k], 90) < np.percentile(width[-k:], 90):  # u: from the head to the tip
        u, v = -u, -v
        s, t = -s, -t
        smin, smax = -smax, -smin
        width = width[::-1]
    head_w = float(np.percentile(width[:k], 90))
    shank = width[int(0.45 * n_bins) : int(0.85 * n_bins)]
    shank_w = float(np.median(shank)) if len(shank) else float(np.median(width))
    notes = []
    if head_w < 1.15 * shank_w:
        notes.append("head hardly wider than the shank: bearing uncertain")
    # bearing: where the width, having reached the head width, falls below the middle
    mid = (head_w + shank_w) / 2
    s_axis = np.linspace(smin, smax, n_bins)
    i0 = next((i for i in range(n_bins) if width[i] >= 0.85 * head_w), 0)
    j = next((i for i in range(i0 + 1, n_bins) if width[i] < mid), int(0.15 * n_bins))
    if 0 < j < n_bins:
        w0, w1 = width[j - 1], width[j]
        f = (w0 - mid) / (w0 - w1) if w0 != w1 else 0.5
        s_bear = s_axis[j - 1] + f * (s_axis[j] - s_axis[j - 1])
    else:
        s_bear = s_axis[j]

    # sub-pixel refinement on the image
    def refine(s0: float, offsets: np.ndarray) -> float:
        found = [_edge_along(gray, c + o * v, u, s0) for o in offsets]
        found = [f for f in found if f is not None]
        return float(np.median(found)) if found else s0

    s_tip = refine(smax + 0.5, np.array([-0.15, 0.0, 0.15]) * shank_w)
    top_offsets = np.array([0.0]) if head_shape == "dome" else np.linspace(-0.3, 0.3, 5) * head_w
    s_top = refine(smin - 0.5, top_offsets)
    lateral = (shank_w / 2 + (head_w - shank_w) / 4) if head_w > shank_w else shank_w / 2
    s_bear = refine(s_bear, np.array([-lateral, lateral]))
    kps = {
        "head_top": c + s_top * u,
        "bearing": c + s_bear * u,
        "tip": c + s_tip * u,
    }
    return EdgeKeypoints(kps, u, [float(w) for w in shank], head_w, notes)


def end_radii(geo: ClassGeometry) -> dict[str, float]:
    """Radius of the visible end faces (mm): the rim that forms the silhouette extremity."""
    if geo.family == "ISO7380":
        top = 0.35 * geo.head_radius  # domed head: the top is close to the axis
    elif geo.family == "DIN933":
        top = 0.95 * geo.head_radius
    else:
        top = 0.92 * geo.head_radius
    return {"head_top": top, "tip": 0.75 * geo.d / 2, "bearing": 0.0}


def rim_corrected(
    cam: Camera,
    px: np.ndarray,
    axis_height: float,
    radius: float,
    outward_xy: np.ndarray,
    axis_px: np.ndarray | None = None,
) -> np.ndarray:
    """Pixel of the axis point behind a silhouette extremity formed by an end-face rim.

    The end face is a disc around the axis point. Seen from above, its top rim lies further from
    the camera's nadir than the axis point, its bottom rim closer. The extremity in the direction
    `outward_xy` (plate frame) is the top rim if that direction points away from the nadir, else
    the bottom rim; the axis point is straight below/above it. Only the part of that shift along
    the screw axis moves the extremity found on the axis line, so with `axis_px` (unit image
    direction of the axis) the correction is projected onto it.
    """
    if radius <= 0:
        return np.asarray(px, float)
    nadir = cam.centre_in_plate[:2]
    probe = cam.pixel_to_plate(np.asarray([px], float), axis_height)[0]
    away = float(np.dot(outward_xy, probe[:2] - nadir)) > 0
    z_rim = axis_height + radius if away else max(0.0, axis_height - radius)
    rim = cam.pixel_to_plate(np.asarray([px], float), z_rim)[0]
    full = cam.project(np.array([[rim[0], rim[1], axis_height]]))[0]
    if axis_px is None:
        return full
    u = np.asarray(axis_px, float) / np.linalg.norm(axis_px)
    p = np.asarray(px, float)
    return p + float((full - p) @ u) * u


def classical_keypoints(
    gray: np.ndarray,
    mask: np.ndarray,
    cam: Camera,
    geo: ClassGeometry,
    iterations: int = 2,
) -> tuple[dict[str, np.ndarray], EdgeKeypoints] | None:
    """Method B keypoints (axis points, image px) of one lying screw."""
    head_shape = "dome" if geo.family == "ISO7380" else "flat"
    ek = edge_keypoints(gray, mask, head_shape)
    if ek is None:
        return None
    radii = end_radii(geo)
    kps = {k: v.copy() for k, v in ek.keypoints.items()}
    L = geo.lengths[len(geo.lengths) // 2] if geo.lengths else 10.0
    for _ in range(max(1, iterations)):
        h = geo.heights(L)
        a = cam.pixel_to_plate(np.asarray([ek.keypoints["head_top"]]), h["head_top"])[0]
        t = cam.pixel_to_plate(np.asarray([ek.keypoints["tip"]]), h["tip"])[0]
        tip_dir = (t - a)[:2] / max(1e-9, float(np.linalg.norm((t - a)[:2])))
        kps = {
            "tip": rim_corrected(
                cam, ek.keypoints["tip"], h["tip"], radii["tip"], tip_dir, ek.axis
            ),
            "head_top": rim_corrected(
                cam, ek.keypoints["head_top"], h["head_top"], radii["head_top"], -tip_dir, ek.axis
            ),
            "bearing": ek.keypoints["bearing"],
        }
        L, _, _ = measure_length(cam, geo, kps[geo.anchor], kps["tip"], heights=h)
    return kps, ek


def to_gray(img: np.ndarray) -> np.ndarray:
    """Float grey image in [0, 1] from 8-bit RGB/BGR or grey."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a / 255.0 if a.max() > 1.5 else a


def describe(ek: EdgeKeypoints) -> Mapping[str, Any]:
    return {
        "head_width_px": round(ek.head_width_px, 2),
        "shank_width_px": round(float(np.median(ek.shank_widths_px)), 2)
        if ek.shank_widths_px
        else None,
        "notes": list(ek.notes),
    }
