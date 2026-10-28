"""2D profiles and the helical ISO metric thread surface.

Conventions: lengths in mm, the part axis is +z. Profiles for revolution are (r, z) polygons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

COS30 = math.cos(math.radians(30))
TAN30 = math.tan(math.radians(30))

# ISO 68-1 basic profile: H = 0.866 p. External thread depth h3 = 0.61343 p (root d3),
# internal thread depth 5H/8 = 0.54127 p (minor diameter D1).
EXT_DEPTH = 0.61343
INT_DEPTH = 0.54127
CREST = 1 / 8  # crest flat width (fraction of the pitch) at the major diameter
EXT_ROOT = 1 - CREST - 2 * EXT_DEPTH * TAN30  # flat approximating the rounded root
INT_ROOT = 1 / 4  # crest of the internal thread (flat at D1)


def thread_depth_fraction(u: np.ndarray, crest: float, root: float) -> np.ndarray:
    """Trapezoidal 60 deg profile over one pitch: 0 at the major diameter, 1 at the root.

    u is the axial phase in pitches (any real; taken modulo 1). The crest occupies [0, crest).
    """
    u = np.mod(u, 1.0)
    flank = (1.0 - crest - root) / 2.0
    a, b, c = crest, crest + flank, crest + flank + root
    return np.select(
        [u < a, u < b, u < c],
        [np.zeros_like(u), (u - a) / flank, np.ones_like(u)],
        1.0 - (u - c) / flank,
    )


@dataclass(frozen=True)
class RodSpec:
    """Surface of revolution r(theta, z) with an optional right-hand helical thread.

    The thread runs from z0 up to thread_end and fades to the plain radius over `runout`.
    tip_radius/tip_slope add a conical chamfer at z0 (r <= tip_radius + (z - z0) * tip_slope).
    """

    radius: float  # major radius (plain radius where there is no thread)
    pitch: float
    z0: float
    z1: float
    depth: float  # radial thread depth (mm), 0 = plain cylinder
    crest: float = CREST
    root: float = EXT_ROOT
    thread_end: float | None = None  # None = threaded over the whole length
    runout: float = 0.0
    tip_radius: float | None = None
    tip_slope: float = 1.0


def rod_mesh(spec: RodSpec, segments: int, axial_per_pitch: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed, outward-oriented triangle mesh (vertices, faces) of a RodSpec.

    Vertices lie on rings of `segments` points. The ring at theta = 0 samples the crest exactly, so
    the maximum vertex radius equals the major radius.
    """
    if spec.z1 <= spec.z0:
        raise ValueError("rod z1 must be > z0")
    n_z = max(2, math.ceil((spec.z1 - spec.z0) / spec.pitch * axial_per_pitch) + 1)
    z = np.linspace(spec.z0, spec.z1, n_z)
    theta = np.arange(segments) * (2 * math.pi / segments)
    zz, tt = np.meshgrid(z, theta, indexing="ij")  # (n_z, segments)

    if spec.depth > 0:
        u = zz / spec.pitch - tt / (2 * math.pi)  # constant along a right-hand helix
        frac = thread_depth_fraction(u, spec.crest, spec.root)
        if spec.thread_end is not None:
            if spec.runout > 0:
                frac = frac * np.clip((spec.thread_end - zz) / spec.runout, 0.0, 1.0)
            else:
                frac = np.where(zz <= spec.thread_end, frac, 0.0)
        r = spec.radius - spec.depth * frac
    else:
        r = np.full_like(zz, spec.radius)
    if spec.tip_radius is not None:
        r = np.minimum(r, spec.tip_radius + (zz - spec.z0) * spec.tip_slope)

    ring = np.stack([r * np.cos(tt), r * np.sin(tt), zz], axis=-1).reshape(-1, 3)
    bottom = np.array([[0.0, 0.0, spec.z0]])
    top = np.array([[0.0, 0.0, spec.z1]])
    vertices = np.vstack([ring, bottom, top])
    ib, it = len(ring), len(ring) + 1

    j = np.arange(n_z - 1)[:, None]
    i = np.arange(segments)[None, :]
    i1 = (i + 1) % segments
    a = j * segments + i
    b = j * segments + i1
    c = (j + 1) * segments + i1
    d = (j + 1) * segments + i
    side = np.concatenate(
        [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    )
    i = np.arange(segments)
    i1 = (i + 1) % segments
    cap0 = np.stack([np.full(segments, ib), i1, i], -1)
    last = (n_z - 1) * segments
    cap1 = np.stack([np.full(segments, it), last + i, last + i1], -1)
    faces = np.concatenate([side, cap0, cap1]).astype(np.int64)
    return vertices, faces


def rod_triangles(length: float, pitch: float, segments: int, axial_per_pitch: int) -> int:
    n_z = max(2, math.ceil(length / pitch * axial_per_pitch) + 1)
    return 2 * segments * (n_z - 1) + 2 * segments


# ------------------------------------------------------------------------------- 2D polygons


def regular_polygon(circumradius: float, n: int, phase: float = 0.0) -> np.ndarray:
    a = phase + np.arange(n) * (2 * math.pi / n)
    return np.stack([circumradius * np.cos(a), circumradius * np.sin(a)], -1)


def hexagon(across_flats: float) -> np.ndarray:
    """Hexagon with corners on +/-x and flats perpendicular to y (max |y| = s / 2)."""
    return regular_polygon(across_flats / 2 / COS30, 6)


def arc(
    center: tuple[float, float], radius: float, a0: float, a1: float, n: int
) -> list[tuple[float, float]]:
    """Points on a circular arc from angle a0 to a1 (degrees), inclusive."""
    angles = np.radians(np.linspace(a0, a1, n + 1))
    return [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]


def ccw(points: np.ndarray | list) -> np.ndarray:
    """Return the polygon in counter-clockwise order (positive area)."""
    p = np.asarray(points, dtype=float)
    x, y = p[:, 0], p[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return p if area > 0 else p[::-1].copy()


def rounded_rect_rz(r0: float, r1: float, z0: float, z1: float, rad: float, n: int) -> np.ndarray:
    """(r, z) rectangle with rounded corners, e.g. a washer cross-section."""
    rad = min(rad, (r1 - r0) / 2, (z1 - z0) / 2)
    pts = []
    pts += arc((r1 - rad, z0 + rad), rad, -90, 0, n)
    pts += arc((r1 - rad, z1 - rad), rad, 0, 90, n)
    pts += arc((r0 + rad, z1 - rad), rad, 90, 180, n)
    pts += arc((r0 + rad, z0 + rad), rad, 180, 270, n)
    return ccw(pts)
