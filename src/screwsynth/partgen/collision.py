"""Collision proxies: a list of low-poly convex pieces (compound collision shape)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import trimesh

MAX_TRIANGLES = 500


def total_triangles(pieces: list[trimesh.Trimesh]) -> int:
    return int(sum(len(p.faces) for p in pieces))


def write_obj(pieces: list[trimesh.Trimesh], path: str | os.PathLike[str]) -> None:
    """Write the pieces as separate objects (o piece_<i>) of one OBJ file, in mm."""
    lines = ["# screw-synth collision proxy: one convex object per piece, units mm"]
    offset = 1
    for i, piece in enumerate(pieces):
        lines.append(f"o piece_{i}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in piece.vertices]
        lines += [f"f {a + offset} {b + offset} {c + offset}" for a, b, c in piece.faces]
        offset += len(piece.vertices)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_obj(path: str | os.PathLike[str]) -> list[trimesh.Trimesh]:
    pieces: list[tuple[list, list]] = []
    offset = 1
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("o "):
            if pieces:
                offset += len(pieces[-1][0])
            pieces.append(([], []))
        elif line.startswith("v "):
            pieces[-1][0].append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            pieces[-1][1].append([int(x.split("/")[0]) - offset for x in line.split()[1:4]])
    return [trimesh.Trimesh(np.array(v), np.array(f), process=False) for v, f in pieces]


def contained_fraction(mesh: trimesh.Trimesh, pieces: list[trimesh.Trimesh]) -> float:
    """Exact fraction of the render mesh volume inside the union of the convex pieces."""
    from manifold3d import Manifold, Mesh

    body = Manifold(
        Mesh(
            vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32),
        )
    )
    if body.is_empty() or body.volume() <= 0:
        return 0.0
    proxy = Manifold.batch_boolean(
        [Manifold.hull_points(np.asarray(p.vertices, dtype=float)) for p in pieces],
        _union_op(),
    )
    return float((body ^ proxy).volume() / body.volume())


def _union_op():
    from manifold3d import OpType

    return OpType.Add
