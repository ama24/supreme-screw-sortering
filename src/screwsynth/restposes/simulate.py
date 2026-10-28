"""Optional validation: drop a part many times in a physics sim and compare with the computed poses.

Uses pybullet when installed (`pip install pybullet`; there is no wheel for every Python/OS, e.g.
none for Python 3.12 on Windows). Resting on a flat plate only depends on the convex hull, so the
part is simulated as one convex hull. Without pybullet the validation is skipped with a warning.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from .compute import Pose, equivalent

log = logging.getLogger(__name__)

SCALE = 0.01  # mm -> simulation units (cm): small parts are unstable in metres


def available() -> bool:
    try:
        import pybullet  # noqa: F401
    except ImportError:
        return False
    return True


def classify(R: np.ndarray, poses: Sequence[Pose], shape: str, tol_deg: float = 8.0) -> int | None:
    """Index of the computed pose that a final rotation R corresponds to (None if no match)."""
    best, best_err = None, tol_deg
    for i, pose in enumerate(poses):
        for tol in (1.0, 3.0, tol_deg):
            if tol <= best_err and equivalent(R, pose.transform[:3, :3], shape, tol):
                best, best_err = i, tol
                break
    return best


@dataclass
class Comparison:
    drops: int
    empirical: list[float]
    computed: list[float]
    unmatched: float
    disagreements: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "drops": self.drops,
            "empirical": [round(x, 4) for x in self.empirical],
            "computed": [round(x, 4) for x in self.computed],
            "unmatched_fraction": round(self.unmatched, 4),
            "disagreements": self.disagreements,
        }


def compare(
    labels: Sequence[int | None], poses: Sequence[Pose], max_abs_diff: float = 0.15
) -> Comparison:
    n = len(labels)
    counts = np.zeros(len(poses))
    for lab in labels:
        if lab is not None:
            counts[lab] += 1
    emp = (counts / n).tolist() if n else [0.0] * len(poses)
    comp = [p.probability for p in poses]
    unmatched = sum(lab is None for lab in labels) / n if n else 0.0
    out = Comparison(n, emp, comp, unmatched)
    for i, (e, c) in enumerate(zip(emp, comp, strict=True)):
        if abs(e - c) > max_abs_diff:
            out.disagreements.append(
                f"pose {i} ({poses[i].category}, tilt {poses[i].axis_tilt_deg:.1f}): "
                f"computed {c:.2f}, simulated {e:.2f}"
            )
    if unmatched > 0.1:
        out.disagreements.append(f"{unmatched:.0%} of drops ended in a pose that was not computed")
    return out


def _random_quaternion(rng: np.random.Generator) -> list[float]:
    u1, u2, u3 = rng.random(3)
    return [
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ]


def drop(mesh: trimesh.Trimesh, n: int, seed: int = 0, max_steps: int = 4000) -> list[np.ndarray]:
    """Final rotations (3x3) of n drops from random orientations onto a plane."""
    import pybullet as pb

    hull = mesh.convex_hull
    com = np.asarray(mesh.center_mass)
    verts = ((hull.vertices - com) * SCALE).tolist()
    size = float(np.ptp(mesh.vertices, axis=0).max()) * SCALE
    rng = np.random.default_rng(seed)
    cid = pb.connect(pb.DIRECT)
    try:
        pb.setGravity(0, 0, -981.0, physicsClientId=cid)  # cm/s^2
        pb.setTimeStep(1 / 960, physicsClientId=cid)
        plane = pb.createCollisionShape(pb.GEOM_PLANE, physicsClientId=cid)
        pb.createMultiBody(0, plane, physicsClientId=cid)
        shape = pb.createCollisionShape(pb.GEOM_MESH, vertices=verts, physicsClientId=cid)
        mass = max(1e-3, float(mesh.volume) * 7.85e-3)  # grams, steel
        out = []
        for _ in range(n):
            body = pb.createMultiBody(
                mass,
                shape,
                basePosition=[0, 0, 1.5 * size],
                baseOrientation=_random_quaternion(rng),
                physicsClientId=cid,
            )
            pb.changeDynamics(
                body, -1, lateralFriction=0.6, rollingFriction=1e-4, physicsClientId=cid
            )
            for step in range(max_steps):
                pb.stepSimulation(physicsClientId=cid)
                if step > 200 and step % 50 == 0:
                    lin, ang = pb.getBaseVelocity(body, physicsClientId=cid)
                    if np.linalg.norm(lin) < 1e-3 and np.linalg.norm(ang) < 1e-2:
                        break
            _, quat = pb.getBasePositionAndOrientation(body, physicsClientId=cid)
            out.append(np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3))
            pb.removeBody(body, physicsClientId=cid)
        return out
    finally:
        pb.disconnect(cid)


def validate(
    mesh: trimesh.Trimesh, poses: Sequence[Pose], shape: str, n: int, seed: int = 0
) -> Comparison | None:
    if not available():
        log.warning("pybullet is not installed; skipping the drop validation")
        return None
    rotations = drop(mesh, n, seed)
    return compare([classify(R, poses, shape) for R in rotations], poses)
