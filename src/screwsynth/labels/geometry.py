"""Mask -> polygon / box / oriented box conversions (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np


def mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Outer contours (N, 2) of a boolean mask, largest first (pixel-centre coordinates)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c.reshape(-1, 2).astype(np.float64) for c in cs if len(c) >= 1]
    return sorted(cs, key=lambda c: -cv2.contourArea(c.astype(np.float32)) - len(c) * 1e-6)


def simplify(contour: np.ndarray, tol_px: float, min_points: int = 4) -> np.ndarray:
    """Douglas-Peucker with tolerance tol_px; relaxes the tolerance down if too few points remain."""
    c = contour.astype(np.float32).reshape(-1, 1, 2)
    t = tol_px
    while True:
        s = cv2.approxPolyDP(c, t, True).reshape(-1, 2)
        if len(s) >= min_points or t < 0.05 or len(contour) <= min_points:
            return s.astype(np.float64) if len(s) >= 3 else contour
        t /= 2


def merge_pieces(pieces: list[np.ndarray]) -> np.ndarray:
    """Join several polygons into one with zero-width bridges (as Ultralytics expects for split masks)."""
    if not pieces:
        return np.zeros((0, 2))
    poly = pieces[0]
    rest = list(pieces[1:])
    while rest:
        # connect the piece closest to the current polygon
        best = None
        for k, p in enumerate(rest):
            d = np.linalg.norm(poly[:, None, :] - p[None, :, :], axis=2)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if best is None or d[i, j] < best[0]:
                best = (d[i, j], k, i, j)
        _, k, i, j = best
        p = rest.pop(k)
        poly = np.vstack([poly[: i + 1], p[j:], p[: j + 1], poly[i:]])
    return poly


def polygon(
    mask: np.ndarray, tol_px: float = 0.7, multi: str = "merge", min_points: int = 4
) -> np.ndarray:
    """Single polygon for a (possibly split) visible mask. multi: merge | largest."""
    cs = [simplify(c, tol_px, min_points) for c in mask_contours(mask)]
    cs = [c for c in cs if len(c) >= 3]
    if not cs:
        return np.zeros((0, 2))
    return cs[0] if multi == "largest" or len(cs) == 1 else merge_pieces(cs)


def bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned box (x0, y0, x1, y1) covering whole pixels of the mask."""
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def obb(mask: np.ndarray) -> np.ndarray:
    """Minimum-area rotated rectangle around the mask pixels (4 corners, pixel-edge coordinates)."""
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys].astype(np.float32)
    # include pixel extents: the 4 corners of every boundary pixel
    offs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    hull = cv2.convexHull(pts)
    pts = (hull.reshape(-1, 1, 2) + offs[None]).reshape(-1, 2)
    return cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float64)


def rasterize(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def rle_encode(mask: np.ndarray) -> dict:
    """Uncompressed COCO RLE (column-major run lengths, starting with background)."""
    flat = mask.astype(np.uint8).ravel(order="F")
    changes = np.flatnonzero(np.diff(np.r_[0, flat, 0]))
    runs = np.diff(np.r_[0, changes])
    counts = runs.tolist()
    if len(flat) - sum(counts) > 0:
        counts.append(len(flat) - sum(counts))
    return {"size": list(mask.shape), "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for n in rle["counts"]:
        flat[pos : pos + n] = val
        pos += n
        val ^= 1
    return flat.reshape((h, w), order="F").astype(bool)
