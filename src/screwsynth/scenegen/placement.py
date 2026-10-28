"""2D placement of parts on the plate: convex footprints, overlap rejection, controlled touching.

Pure numpy (usable inside Blender). All lengths in mm in the plate frame.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain)."""
    pts = np.unique(np.round(np.asarray(points, dtype=float)[:, :2], 9), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def rot2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def separation(a: np.ndarray, b: np.ndarray) -> float:
    """Separating-axis distance between convex polygons (> 0 apart, < 0 penetration depth)."""
    best = -math.inf
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.c_[edges[:, 1], -edges[:, 0]]
        lens = np.linalg.norm(normals, axis=1)
        normals = normals[lens > 1e-12] / lens[lens > 1e-12, None]
        pa, pb = a @ normals.T, b @ normals.T
        gap = np.maximum(pb.min(axis=0) - pa.max(axis=0), pa.min(axis=0) - pb.max(axis=0))
        best = max(best, float(gap.max()))
    return best


@dataclass
class Placed:
    index: int
    footprint: np.ndarray  # (N, 2) plate xy, CCW
    yaw: float
    xy: tuple[float, float]


def footprint(local_hull: np.ndarray, yaw: float, xy: Sequence[float]) -> np.ndarray:
    return local_hull @ rot2(yaw).T + np.asarray(xy, dtype=float)


def inside_area(poly: np.ndarray, area: Sequence[float]) -> bool:
    x0, y0, x1, y1 = area
    return bool(
        poly[:, 0].min() >= x0
        and poly[:, 1].min() >= y0
        and poly[:, 0].max() <= x1
        and poly[:, 1].max() <= y1
    )


def _touching_position(
    hull: np.ndarray, yaw: float, other: Placed, gap: float, rng: random.Random
) -> tuple[float, float]:
    """Slide the new footprint towards a placed one along a random direction until they touch."""
    ang = rng.uniform(0, 2 * math.pi)
    d = np.array([math.cos(ang), math.sin(ang)])
    centre = other.footprint.mean(axis=0)
    radius = float(
        np.linalg.norm(other.footprint - centre, axis=1).max() + np.linalg.norm(hull, axis=1).max()
    )
    lo, hi = 0.0, 2 * radius + 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        sep = separation(footprint(hull, yaw, centre + mid * d), other.footprint)
        if sep < gap:
            lo = mid
        else:
            hi = mid
    p = centre + hi * d
    return float(p[0]), float(p[1])


def place(
    hulls: Sequence[np.ndarray],
    area: Sequence[float],
    rng: random.Random,
    gap_mm: float = 1.0,
    touch_probability: float = 0.0,
    max_tries: int = 60,
) -> list[Placed]:
    """Place footprints (local hulls, yaw 0) at random yaw/position without overlap.

    gap_mm > 0 keeps parts apart, gap_mm < 0 lets footprints overlap by that much (they touch and lean
    after the physics settle). With touch_probability a new part is slid against an existing one.
    Parts that cannot be placed after max_tries are skipped.
    """
    placed: list[Placed] = []
    x0, y0, x1, y1 = area
    for i, hull in enumerate(hulls):
        for _ in range(max_tries):
            yaw = rng.uniform(0, 2 * math.pi)
            if placed and rng.random() < touch_probability:
                other = rng.choice(placed)
                xy = _touching_position(hull, yaw, other, gap_mm, rng)
            else:
                xy = (rng.uniform(x0, x1), rng.uniform(y0, y1))
            poly = footprint(hull, yaw, xy)
            if not inside_area(poly, area):
                continue
            if all(separation(poly, p.footprint) >= gap_mm - 1e-9 for p in placed):
                placed.append(Placed(i, poly, yaw, xy))
                break
    return placed
