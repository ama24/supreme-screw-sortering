"""2D placement of parts on the plate: convex footprints, overlap rejection, controlled touching.

Pure numpy (usable inside Blender). All lengths in mm in the plate frame.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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


def edge_normals(poly: np.ndarray) -> np.ndarray:
    """Unit outward normals of a CCW polygon's edges (the separating-axis candidates)."""
    e = np.empty_like(poly)
    e[:-1] = poly[1:] - poly[:-1]
    e[-1] = poly[0] - poly[-1]
    n = np.stack([e[:, 1], -e[:, 0]], axis=1)
    lens = np.hypot(n[:, 0], n[:, 1])
    ok = lens > 1e-12
    return n[ok] / lens[ok, None]


def separation(
    a: np.ndarray, b: np.ndarray, na: np.ndarray | None = None, nb: np.ndarray | None = None
) -> float:
    """Separating-axis distance between convex polygons (> 0 apart, < 0 penetration depth).
    na / nb: precomputed edge_normals of a / b."""
    normals = np.vstack(
        [edge_normals(a) if na is None else na, edge_normals(b) if nb is None else nb]
    )
    pa, pb = a @ normals.T, b @ normals.T
    gap = np.maximum(pb.min(axis=0) - pa.max(axis=0), pa.min(axis=0) - pb.max(axis=0))
    return float(gap.max())


@dataclass
class Placed:
    index: int
    footprint: np.ndarray  # (N, 2) plate xy, CCW
    yaw: float
    xy: tuple[float, float]
    layer: int = 1  # 1 = on the plate in its rest pose, 2 = dropped onto parts of layer 1


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


# --------------------------------------------------------------------------- clumped layouts


@dataclass
class ClumpSettings:
    """How a handful of parts spread by hand looks: clumps, contact, some parts on top of others."""

    clumps: tuple[int, int] = (1, 3)  # number of clumps per scene
    spread_mm: tuple[float, float] = (4.0, 15.0)  # radius scale of a clump (Gaussian sigma)
    loose_fraction: float = 0.1  # parts placed anywhere, away from the clumps
    edge_probability: float = 0.3  # chance a clump sits against the edge (lip) of the frame
    overlap_probability: tuple[float, float] = (0.1, 0.35)  # per scene: share of parts on top
    contact_gap_mm: tuple[float, float] = (0.0, 1.0)  # gap left when a part is slid into a clump
    give_up_after: int = 5  # consecutive parts that do not fit: the plate is full, stop
    max_support_mm: float = 12.0  # parts on top only land on parts at most this tall

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> ClumpSettings:
        cfg = dict(cfg or {})
        d = cls()

        def pair(key: str, cast: type) -> tuple:
            lo, hi = cfg.get(key, getattr(d, key))
            return cast(lo), cast(hi)

        return cls(
            clumps=pair("clumps", int),
            spread_mm=pair("spread_mm", float),
            loose_fraction=float(cfg.get("loose_fraction", d.loose_fraction)),
            edge_probability=float(cfg.get("edge_probability", d.edge_probability)),
            overlap_probability=pair("overlap_probability", float),
            contact_gap_mm=pair("contact_gap_mm", float),
            give_up_after=int(cfg.get("give_up_after", d.give_up_after)),
            max_support_mm=float(cfg.get("max_support_mm", d.max_support_mm)),
        )


def _clump_centres(
    area: Sequence[float], st: ClumpSettings, rng: random.Random
) -> list[tuple[np.ndarray, float]]:
    """(centre, spread) of each clump; some are pushed against an edge of the area."""
    x0, y0, x1, y1 = area
    w, h = x1 - x0, y1 - y0
    out = []
    for _ in range(rng.randint(*st.clumps)):
        c = np.array(
            [rng.uniform(x0 + 0.1 * w, x1 - 0.1 * w), rng.uniform(y0 + 0.1 * h, y1 - 0.1 * h)]
        )
        if rng.random() < st.edge_probability:
            side = rng.randrange(4)
            depth = rng.uniform(0.0, 0.12) * (w if side < 2 else h)
            if side == 0:
                c[0] = x0 + depth
            elif side == 1:
                c[0] = x1 - depth
            elif side == 2:
                c[1] = y0 + depth
            else:
                c[1] = y1 - depth
        out.append((c, rng.uniform(*st.spread_mm)))
    return out


class _Index:
    """Placed footprints with bounding circles: exact overlap tests only for nearby parts."""

    def __init__(self) -> None:
        self.placed: list[Placed] = []
        self.normals: list[np.ndarray] = []
        self.centres = np.zeros((0, 2))
        self.radii = np.zeros(0)

    def add(self, p: Placed) -> None:
        c, r = _circle(p.footprint)
        self.placed.append(p)
        self.normals.append(edge_normals(p.footprint))
        self.centres = np.vstack([self.centres, c])
        self.radii = np.append(self.radii, r)

    def near_idx(self, poly: np.ndarray, reach: float) -> np.ndarray:
        if not self.placed:
            return np.zeros(0, dtype=int)
        c, r = _circle(poly)
        d = np.hypot(*(self.centres - c).T)
        return np.nonzero(d < self.radii + r + reach)[0]

    def near(self, poly: np.ndarray, reach: float) -> list[Placed]:
        return [self.placed[i] for i in self.near_idx(poly, reach)]

    def clear(self, poly: np.ndarray, gap: float) -> bool:
        """No overlap with layer-1 footprints (gap) nor with layer-2 ones (touching allowed)."""
        idx = self.near_idx(poly, max(gap, 0.0) + 1e-6)
        if len(idx) == 0:
            return True
        n_poly = edge_normals(poly)
        for i in idx:
            p = self.placed[i]
            need = gap if p.layer == 1 else 0.0
            if separation(poly, p.footprint, n_poly, self.normals[i]) < need - 1e-9:
                return False
        return True


def _circle(poly: np.ndarray) -> tuple[np.ndarray, float]:
    c = poly.mean(axis=0)
    d = poly - c
    return c, float(np.sqrt((d * d).sum(axis=1).max()))


def _circle_escape(
    centre: np.ndarray, radius: float, direction: np.ndarray, index: _Index, gap: float
) -> float:
    """Smallest t >= 0 where a circle (centre + t * direction, radius) clears every placed bounding
    circle: beyond it the footprint is certainly clear (circles enclose the footprints)."""
    if not index.placed:
        return 0.0
    rel = centre - index.centres
    reach = index.radii + radius + max(gap, 0.0) + 1e-6
    b = rel @ direction  # |rel + t d|^2 = t^2 + 2 b t + c  < reach^2  -> blocked
    c = np.einsum("ij,ij->i", rel, rel) - reach**2
    disc = b * b - c
    hit = disc > 0
    lo = -b[hit] - np.sqrt(disc[hit])
    hi = -b[hit] + np.sqrt(disc[hit])
    keep = hi > 0
    intervals = sorted(zip(np.maximum(lo[keep], 0.0), hi[keep], strict=True))
    t = 0.0
    for a, z in intervals:
        if a > t:
            break
        t = max(t, z)
    return float(t)


def _slide_out(
    hull: np.ndarray,
    yaw: float,
    start: np.ndarray,
    direction: np.ndarray,
    index: _Index,
    gap: float,
    max_dist: float,
) -> np.ndarray | None:
    """First position along start + t * direction (t >= 0) where the footprint is clear: the part
    is pushed out of the clump until it rests against its neighbours."""
    poly0 = footprint(hull, yaw, start)
    if index.clear(poly0, gap):
        return start
    c, r = _circle(poly0)
    hi = _circle_escape(c, r, direction, index, gap)
    if hi > max_dist:
        return None
    if not index.clear(footprint(hull, yaw, start + hi * direction), gap):
        hi += 0.05  # numerical edge: step just past the circle contact
        if not index.clear(footprint(hull, yaw, start + hi * direction), gap):
            return None
    lo = 0.0  # blocked at lo, clear at hi: bisect to the contact
    for _ in range(10):
        mid = (lo + hi) / 2
        if index.clear(footprint(hull, yaw, start + mid * direction), gap):
            hi = mid
        else:
            lo = mid
    return start + hi * direction


def _point_in_convex(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Is a point inside a CCW convex polygon?"""
    e = np.roll(poly, -1, axis=0) - poly
    d = pt - poly
    return bool(np.all(e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0] >= 0))


def _centre_inside(poly: np.ndarray, area: Sequence[float]) -> bool:
    c = poly.mean(axis=0)
    x0, y0, x1, y1 = area
    return bool(x0 <= c[0] <= x1 and y0 <= c[1] <= y1)


def place_clumped(
    hulls: Sequence[np.ndarray],
    area: Sequence[float],
    bounds: Sequence[float],
    rng: random.Random,
    st: ClumpSettings | None = None,
    max_tries: int = 60,
    heights: Sequence[float] | None = None,
) -> list[Placed]:
    """Clumped arrangement of footprints, like a handful of parts spread by hand.

    Layer 1 parts lie on the plate in a rest pose: each one aims at a point drawn around a clump
    centre and, if that spot is taken, slides outward until it touches (or nearly touches) the
    parts already there. A share of the parts is placed on top instead (layer 2): their footprint
    overlaps layer-1 parts but no other layer-2 part, so at most two layers form, and the physics
    settle drops them onto the parts below. Footprints must stay inside `bounds` (the lip around
    the plate) and their centre inside `area` (the visible working area), so parts pushed against
    the lip are cut by the image border like in a real photo.

    heights (mm, per hull) keep parts on top off tall ones (a screw standing on its head): the
    centre of a top part must lie over a layer-1 part at most st.max_support_mm tall.
    """
    st = st or ClumpSettings()
    clumps = _clump_centres(area, st, rng)
    p_top = rng.uniform(*st.overlap_probability)
    index = _Index()
    x0, y0, x1, y1 = area
    misses = 0
    diag = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])
    for i, hull in enumerate(hulls):
        if misses >= st.give_up_after:
            break
        before = len(index.placed)
        loose = rng.random() < st.loose_fraction
        has_floor = any(p.layer == 1 for p in index.placed)
        top = not loose and has_floor and rng.random() < p_top
        centre, spread = clumps[rng.randrange(len(clumps))]
        for _ in range(max_tries):
            yaw = rng.uniform(0, 2 * math.pi)
            if loose:
                target = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            else:
                target = centre + np.array([rng.gauss(0, spread), rng.gauss(0, spread)])
            if top:
                poly = footprint(hull, yaw, target)
                if not (inside_area(poly, bounds) and _centre_inside(poly, area)):
                    continue
                near = index.near(poly, 0.0)
                below = [p for p in near if p.layer == 1 and separation(poly, p.footprint) < -0.5]
                clear_top = all(separation(poly, p.footprint) >= 0 for p in near if p.layer == 2)
                low = heights is None or all(heights[p.index] <= st.max_support_mm for p in below)
                centre_c = poly.mean(axis=0)
                supported = any(_point_in_convex(centre_c, p.footprint) for p in below)
                if below and clear_top and low and supported:
                    index.add(Placed(i, poly, yaw, (float(target[0]), float(target[1])), 2))
                    break
                continue
            gap = rng.uniform(*st.contact_gap_mm) if not loose else max(st.contact_gap_mm[1], 1.0)
            d = target - centre
            n = float(np.linalg.norm(d))
            ang = rng.uniform(0, 2 * math.pi)
            direction = d / n if n > 1e-6 else np.array([math.cos(ang), math.sin(ang)])
            xy = _slide_out(hull, yaw, target, direction, index, gap, diag)
            if xy is None:
                continue
            poly = footprint(hull, yaw, xy)
            if inside_area(poly, bounds) and _centre_inside(poly, area):
                index.add(Placed(i, poly, yaw, (float(xy[0]), float(xy[1])), 1))
                break
        misses = 0 if len(index.placed) > before else misses + 1
    return index.placed


def nearest_gaps(placed: Sequence[Placed]) -> np.ndarray:
    """Distance from every footprint to its nearest neighbour (mm, negative = overlap)."""
    out = []
    for p in placed:
        others = [separation(p.footprint, q.footprint) for q in placed if q is not p]
        out.append(min(others) if others else math.inf)
    return np.asarray(out)
