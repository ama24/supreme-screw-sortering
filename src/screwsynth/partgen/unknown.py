"""Builders for the unknown (non-catalogue) parts: junk the machine must reject.

Every variant gets its parameters from the catalogue (`unknown:` section, drawn per part id) and
its shape details (head style, number of teeth, blob form) from dims["seed"], so a part id always
builds the same mesh. The shapes only need to look plausible; they are never measured.
"""

from __future__ import annotations

import math
import random

import numpy as np
from manifold3d import Manifold

from screwsynth.catalogue.schema import Part

from .builders import (
    BuiltPart,
    Lod,
    _centered,
    _cyl_points,
    _double_chamfer,
    _from_arrays,
    _hull,
    _prism,
    _revolve,
    _rod,
    _simple_result,
    directional_hull,
)
from .profiles import CREST, EXT_DEPTH, EXT_ROOT, RodSpec, hexagon, rod_triangles

METRIC = {3.0: 0.5, 4.0: 0.7, 5.0: 0.8, 6.0: 1.0, 8.0: 1.25, 10.0: 1.5}  # size -> coarse pitch


def _rng(part: Part) -> random.Random:
    return random.Random(int(part.dims.get("seed", 0)))


def _metric(d: float) -> tuple[float, float]:
    """Nearest metric size and its coarse pitch."""
    size = min(METRIC, key=lambda s: abs(s - d))
    return size, METRIC[size]


def _thread_lod(lod: Lod, length: float, pitch: float, budget: int = 150_000) -> Lod:
    k, seg = lod.axial_per_pitch, lod.thread_segments
    while k > 4 and rod_triangles(length, pitch, seg, k) > budget:
        k -= 1
    while seg > 16 and rod_triangles(length, pitch, seg, k) > budget:
        seg -= 4
    return Lod(seg, k, lod.round_segments)


def _box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Manifold:
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def _box_hull(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    return _hull(np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]))


def build_wood_screw(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """Pointed self-tapping / wood screw: countersunk or pan head with a cross recess."""
    d, L = part.dims["d"], part.dims["length"]
    rng = _rng(part)
    R = d / 2
    pitch = 0.45 * d
    depth = 0.16 * d
    countersunk = rng.random() < 0.5
    if countersunk:
        dk = 2.0 * d
        k = (dk - d) / 2  # 90 deg countersink
        head = _revolve([(0, 0), (R, 0), (dk / 2, k), (0, k)], lod.round_segments)
    else:
        dk, k = 1.9 * d, 0.7 * d
        n = max(6, lod.round_segments // 6)
        dome = [
            (dk / 2 * math.cos(a), 0.35 * k + 0.65 * k * math.sin(a))
            for a in np.linspace(0, math.pi / 2, n)
        ]
        head = _revolve([(0, 0), (dk / 2, 0), *dome, (0, k)], lod.round_segments)
    tip_len = 1.2 * d
    spec = RodSpec(
        radius=R,
        pitch=pitch,
        z0=-L,
        z1=0.4 * k,
        depth=depth,
        crest=CREST,
        root=EXT_ROOT,
        thread_end=-0.25 * L if L > 25 else None,  # long screws: plain shank under the head
        runout=pitch,
        tip_radius=0.05 * R,
        tip_slope=R / tip_len,
    )
    body = head + _rod(spec, _thread_lod(lod, L + k, pitch))
    w, cl = 0.16 * dk, 0.55 * dk  # cross recess
    body = body - _box(-cl / 2, cl / 2, -w / 2, w / 2, 0.45 * k, k + 1)
    body = body - _box(-w / 2, w / 2, -cl / 2, cl / 2, 0.45 * k, k + 1)
    head_proxy = _hull(_cyl_points(dk / 2, 0.0, k, coll_n))
    shank = _hull(np.vstack([_cyl_points(R, -L + tip_len, 0.05 * R, coll_n), [[0.0, 0.0, -L]]]))
    return _simple_result(body, [head_proxy, shank])


def build_threaded_rod(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """A cut piece of metric threaded rod."""
    size, pitch = _metric(part.dims["d"])
    L, R = part.dims["length"], size / 2
    depth = EXT_DEPTH * pitch
    spec = RodSpec(
        radius=R,
        pitch=pitch,
        z0=-L / 2,
        z1=L / 2,
        depth=depth,
        crest=CREST,
        root=EXT_ROOT,
        tip_radius=R - 1.05 * depth,
        tip_slope=1.0,
    )
    body = _rod(spec, _thread_lod(lod, L, pitch))
    return _simple_result(body, [_hull(_cyl_points(R, -L / 2, L / 2, coll_n))])


def build_split_washer(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """Spring lock washer (DIN 127 look): square-section ring, cut and twisted."""
    size, _ = _metric(part.dims["d"])
    r0 = (size + 0.2) / 2
    b, s = 0.32 * size + 0.4, 0.22 * size + 0.3
    ring = _revolve([(r0, 0), (r0 + b, 0), (r0 + b, s), (r0, s)], lod.round_segments)
    gap = 0.35 * b
    ring = ring - _box(0, r0 + b + 1, -gap / 2, gap / 2, -1, s + 1)
    rise = 0.8 * s

    def twist(v):
        a = math.atan2(v[1], v[0]) % (2 * math.pi)  # 0 at the gap, rising all the way round
        return (v[0], v[1], v[2] + rise * a / (2 * math.pi))

    body, shift = _centered(ring.warp(twist))
    pts = np.array(body.to_mesh().vert_properties)[:, :3]
    return _simple_result(body, [directional_hull(pts)])


def build_tooth_washer(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """Toothed lock washer: thin ring with external teeth."""
    size, _ = _metric(part.dims["d"])
    rng = _rng(part)
    r_in = (size + 0.3) / 2
    r_out = 1.0 * size + 1.0
    t = 0.07 * size + 0.25
    n = rng.randint(8, 12)
    outline = []
    for i in range(n):
        a = 2 * math.pi * i / n
        for frac, r in ((0.0, 0.82 * r_out), (0.18, r_out), (0.5, r_out), (0.68, 0.82 * r_out)):
            outline.append(
                (r * math.cos(a + frac * 2 * math.pi / n), r * math.sin(a + frac * 2 * math.pi / n))
            )
    body = _prism(np.array(outline), -t / 2, t / 2)
    body = body - Manifold.cylinder(t + 2, r_in, circular_segments=lod.round_segments).translate(
        [0, 0, -t / 2 - 1]
    )
    return _simple_result(body, [_hull(_cyl_points(r_out, -t / 2, t / 2, max(coll_n, 16)))])


def build_pin(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """Dowel / cylindrical pin with chamfered ends."""
    d, L = part.dims["d"], part.dims["length"]
    R, c = d / 2, 0.12 * d
    body = _revolve(
        [(0, -L / 2), (R - c, -L / 2), (R, -L / 2 + c), (R, L / 2 - c), (R - c, L / 2), (0, L / 2)],
        lod.round_segments,
    )
    return _simple_result(body, [_hull(_cyl_points(R, -L / 2, L / 2, coll_n))])


def build_bracket(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """Bent sheet-metal angle bracket with a hole in each leg."""
    a = part.dims["size"]
    rng = _rng(part)
    w = rng.uniform(0.4, 0.7) * a
    b = a * rng.uniform(0.6, 1.0)  # second leg
    t = 0.05 * a + 0.8
    base = _box(0, a, -w / 2, w / 2, 0, t)
    leg = _box(0, t, -w / 2, w / 2, 0, b)
    rh = 0.14 * w
    hole1 = Manifold.cylinder(t + 2, rh, circular_segments=lod.round_segments).translate(
        [a * 0.6, 0, -1]
    )
    hole2 = (
        Manifold.cylinder(t + 2, rh, circular_segments=lod.round_segments)
        .rotate([0, 90, 0])
        .translate([-1, 0, b * 0.6])
    )
    body = base + leg - hole1 - hole2
    body, shift = _centered(body)
    pieces = [_box_hull(0, a, -w / 2, w / 2, 0, t), _box_hull(0, t, -w / 2, w / 2, 0, b)]
    return _simple_result(body, pieces, shift=shift)


def build_standoff(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """Hex spacer / coupling nut with a through bore."""
    size, _ = _metric(part.dims["d"])
    L = part.dims["length"]
    s = 1.6 * size + 0.6
    body = _prism(hexagon(s), 0, L) ^ _double_chamfer(s, 0, L, lod.round_segments)
    body = body - Manifold.cylinder(
        L + 2, 0.43 * size, circular_segments=lod.round_segments
    ).translate([0, 0, -1])
    body, shift = _centered(body)
    proxy = _hull(np.vstack([np.c_[hexagon(s), np.zeros(6)], np.c_[hexagon(s), np.full(6, L)]]))
    return _simple_result(body, [proxy], shift=shift)


def build_plate(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """Flat plate (mending plate, strap) with a row of holes."""
    a = part.dims["size"]
    rng = _rng(part)
    b = rng.uniform(0.25, 0.6) * a
    t = rng.uniform(1.0, 3.0)
    c = 0.15 * b  # chamfered corners
    outline = [
        (-a / 2 + c, -b / 2),
        (a / 2 - c, -b / 2),
        (a / 2, -b / 2 + c),
        (a / 2, b / 2 - c),
        (a / 2 - c, b / 2),
        (-a / 2 + c, b / 2),
        (-a / 2, b / 2 - c),
        (-a / 2, -b / 2 + c),
    ]
    body = _prism(np.array(outline), -t / 2, t / 2)
    holes = rng.randint(2, 4)
    rh = min(0.18 * b, 0.4 * a / holes)
    for i in range(holes):
        x = -a / 2 + a * (i + 0.5) / holes
        body = body - Manifold.cylinder(t + 2, rh, circular_segments=lod.round_segments).translate(
            [x, 0, -t / 2 - 1]
        )
    return _simple_result(body, [_box_hull(-a / 2, a / 2, -b / 2, b / 2, -t / 2, t / 2)])


def build_blob(part: Part, lod: Lod, coll_n: int) -> BuiltPart:
    """Odd bit (plastic piece, broken part): two overlapping random convex lumps."""
    size = part.dims["size"]
    rng = _rng(part)
    lumps = []
    for k in range(2):
        axes = (
            np.array([1.0, rng.uniform(0.35, 1.0), rng.uniform(0.25, 0.8)])
            * size
            / 2
            * (1 - 0.3 * k)
        )
        n = rng.randint(10, 20)
        pts = []
        for _ in range(n):
            v = np.array([rng.gauss(0, 1) for _ in range(3)])
            pts.append(v / np.linalg.norm(v) * axes)
        offset = np.array([rng.uniform(-0.3, 0.3) * size, rng.uniform(-0.2, 0.2) * size, 0.0]) * k
        lumps.append(_hull(np.array(pts) + offset))
    body = _from_arrays(lumps[0].vertices, lumps[0].faces) + _from_arrays(
        lumps[1].vertices, lumps[1].faces
    )
    body, shift = _centered(body)
    return _simple_result(body, lumps, shift=shift)


UNKNOWN_BUILDERS = {
    "unknown_wood_screw": build_wood_screw,
    "unknown_threaded_rod": build_threaded_rod,
    "unknown_split_washer": build_split_washer,
    "unknown_tooth_washer": build_tooth_washer,
    "unknown_pin": build_pin,
    "unknown_bracket": build_bracket,
    "unknown_standoff": build_standoff,
    "unknown_plate": build_plate,
    "unknown_blob": build_blob,
}
