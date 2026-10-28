"""Unit conversions.

Configs and metadata use millimetres; the Blender scene is scaled by WORLD_SCALE so Bullet physics
stays stable: blender_units = mm * 0.001 * world_scale.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

DEFAULT_WORLD_SCALE = 10.0
MM_PER_M = 1000.0


def _check_scale(world_scale: float) -> None:
    if world_scale <= 0:
        raise ValueError(f"world_scale must be > 0, got {world_scale}")


def mm_to_bu(mm: float, world_scale: float = DEFAULT_WORLD_SCALE) -> float:
    """Convert millimetres to Blender units."""
    _check_scale(world_scale)
    return mm / MM_PER_M * world_scale


def bu_to_mm(bu: float, world_scale: float = DEFAULT_WORLD_SCALE) -> float:
    """Convert Blender units to millimetres."""
    _check_scale(world_scale)
    return bu / world_scale * MM_PER_M


def mm_to_bu_vec(v: Iterable[float], world_scale: float = DEFAULT_WORLD_SCALE) -> list[float]:
    return [mm_to_bu(x, world_scale) for x in v]


def bu_to_mm_vec(v: Iterable[float], world_scale: float = DEFAULT_WORLD_SCALE) -> list[float]:
    return [bu_to_mm(x, world_scale) for x in v]


def deg2rad(deg: float) -> float:
    return math.radians(deg)


def rad2deg(rad: float) -> float:
    return math.degrees(rad)
