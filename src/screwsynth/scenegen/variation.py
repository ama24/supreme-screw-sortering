"""Camera views and HDRI environments per layout: settings, even spreading and the camera pose.

Every setting is one value (fixed) or a range [min, max]. Ranges are covered evenly with a
low-discrepancy sequence indexed by the layout number (the Halton sequence, one prime base per
setting, shifted per seed): the first N layouts cover each range evenly for any N, layouts added later fill
the gaps instead of repeating, and the same seed and layout number always give the same view, also
with parallel render workers. Pure Python + numpy: importable inside Blender.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from screwsynth.common.seed import derive_seed

from .camera import RenderCamera

# sequence dimensions: one per setting, so changing one range never moves another one's values;
# each uses its own prime base of the Halton sequence
DIMS = ("height", "tilt", "direction", "roll", "hdri", "hdri_strength", "hdri_rotation")
BASES = (2, 3, 5, 7, 11, 13, 17)
MAX_RAY_DEG = 80.0  # steepest view ray (from vertical) that still hits the plate sensibly


def _radical_inverse(n: int, base: int) -> float:
    """n written in `base`, digits mirrored behind the point (van der Corput)."""
    out, f = 0.0, 1.0 / base
    while n:
        n, digit = divmod(n, base)
        out += digit * f
        f /= base
    return out


def sequence_point(index: int, seed: int) -> dict[str, float]:
    """Point `index` of the seeded sequence: one value in [0, 1) per setting."""
    out = {}
    for name, base in zip(DIMS, BASES, strict=True):
        shift = random.Random(derive_seed(seed, "view-sequence", name)).random()
        out[name] = (shift + _radical_inverse(index + 1, base)) % 1.0
    return out


# ---------------------------------------------------------------------------------- settings


def parse_value(
    value: Any, name: str, allow: Sequence[str] = ()
) -> float | tuple[float, float] | str:
    """One number (fixed), [min, max] (a range) or one of the `allow` keywords."""
    if isinstance(value, str) and value in allow:
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, list | tuple) and len(value) == 2:
        lo, hi = (float(v) for v in value)
        if hi < lo:
            raise ValueError(f"{name}: range [{lo:g}, {hi:g}] has min > max")
        return (lo, hi) if hi > lo else lo
    words = f" or {', '.join(allow)}" if allow else ""
    raise ValueError(f"{name}: expected one number or [min, max]{words}, got {value!r}")


def pick(spec: float | tuple[float, float], u: float) -> float:
    """The value at sequence position u (0..1) of a fixed value or a range."""
    if isinstance(spec, tuple):
        return spec[0] + u * (spec[1] - spec[0])
    return float(spec)


def extremes(spec: float | tuple[float, float], n: int = 2) -> list[float]:
    if isinstance(spec, tuple):
        return [spec[0] + (spec[1] - spec[0]) * i / max(1, n - 1) for i in range(n)]
    return [float(spec)]


@dataclass
class CameraSettings:
    height: float | tuple[float, float] | str = "calibrated"
    tilt: float | tuple[float, float] = 0.0
    direction: float | tuple[float, float] = (0.0, 360.0)
    roll: float | tuple[float, float] = 0.0

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None) -> CameraSettings:
        c = dict(section or {})
        out = cls(
            height=parse_value(c.get("height_mm", "calibrated"), "height_mm", ("calibrated",)),
            tilt=parse_value(c.get("tilt_deg", 0), "tilt_deg"),
            direction=parse_value(c.get("tilt_direction_deg", [0, 360]), "tilt_direction_deg"),
            roll=parse_value(c.get("roll_deg", 0), "roll_deg"),
        )
        for t in extremes(out.tilt):
            if not 0 <= t < 90:
                raise ValueError(f"tilt_deg must be between 0 and 90 (got {t:g})")
        if not isinstance(out.height, str) and min(extremes(out.height)) <= 0:
            raise ValueError("height_mm must be positive")
        return out

    @property
    def is_rig_view(self) -> bool:
        """The defaults: the base (calibrated or nominal) camera, unchanged."""
        return self.height == "calibrated" and self.tilt == 0.0 and self.roll == 0.0

    def to_json(self) -> dict[str, Any]:
        def j(v: Any) -> Any:
            return list(v) if isinstance(v, tuple) else v

        return {
            "height_mm": j(self.height),
            "tilt_deg": j(self.tilt),
            "tilt_direction_deg": j(self.direction),
            "roll_deg": j(self.roll),
        }


# ------------------------------------------------------------------------------------- poses


def base_height(base: RenderCamera) -> float:
    return float(base.centre_in_plate[2])


def view_pose(
    base: RenderCamera,
    target_xy: Sequence[float],
    height_mm: float,
    tilt_deg: float,
    direction_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """T_cam_plate of a camera `height_mm` above the plate, tilted towards `target_xy`.

    Straight down (tilt 0) keeps the base camera's in-plane orientation. The camera then swings
    around the target, away from it towards `direction_deg` (in the plate frame, from +x towards
    +y), keeping its height above the plate; roll turns the image around the view axis.
    """
    R0 = np.asarray(base.T_cam_plate, dtype=float)[:3, :3]
    x = np.array([R0[0, 0], R0[0, 1], 0.0])  # base image x axis, laid flat on the plate
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, -1.0])  # looking straight down
    R_nadir = np.vstack([x, np.cross(z, x), z])  # rows: camera axes in the plate frame
    t, p = math.radians(tilt_deg), math.radians(direction_deg)
    a = np.array([-math.sin(p), math.cos(p), 0.0])  # tilt axis, in the plate
    A = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    Q = np.eye(3) + math.sin(t) * A + (1 - math.cos(t)) * (A @ A)  # Rodrigues
    r = math.radians(roll_deg)
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1.0]])
    R = Rz @ R_nadir @ Q.T
    centre = np.array([target_xy[0], target_xy[1], 0.0]) + (height_mm / math.cos(t)) * (
        Q @ np.array([0.0, 0.0, 1.0])
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ centre
    return T


def with_pose(base: RenderCamera, T_cam_plate: np.ndarray) -> RenderCamera:
    return RenderCamera(
        K_full=base.K_full,
        dist=base.dist,
        image_size=base.image_size,
        roi=dict(base.roi),
        T_cam_plate=np.asarray(T_cam_plate, dtype=float),
        source=base.source,
        scale=base.scale,
    )


def steepest_ray_deg(cam: RenderCamera) -> float:
    """Largest angle from vertical of the rays through the image corners."""
    w, h = cam.resolution
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], dtype=float)
    d = (corners @ np.linalg.inv(cam.K).T) @ np.asarray(cam.T_cam_plate)[:3, :3]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip(-d[:, 2], -1, 1))).max())


@dataclass
class View:
    """The camera of one layout and the values it was made from (recorded in meta.json)."""

    camera: RenderCamera
    values: dict[str, float]
    rig_view: bool

    def to_json(self) -> dict[str, Any]:
        return {"rig_view": self.rig_view, **{k: round(v, 4) for k, v in self.values.items()}}


def layout_view(
    base: RenderCamera,
    settings: CameraSettings,
    target_xy: Sequence[float],
    index: int,
    seed: int,
) -> View:
    h0 = base_height(base)
    if settings.is_rig_view:
        values = {"height_mm": h0, "tilt_deg": 0.0, "tilt_direction_deg": 0.0, "roll_deg": 0.0}
        return View(base, values, True)
    u = sequence_point(index, seed)
    h = h0 if settings.height == "calibrated" else pick(settings.height, u["height"])
    values = {
        "height_mm": h,
        "tilt_deg": pick(settings.tilt, u["tilt"]),
        "tilt_direction_deg": pick(settings.direction, u["direction"]) % 360.0,
        "roll_deg": pick(settings.roll, u["roll"]) % 360.0,
    }
    T = view_pose(
        base,
        target_xy,
        h,
        values["tilt_deg"],
        values["tilt_direction_deg"],
        values["roll_deg"],
    )
    return View(with_pose(base, T), values, False)


def view_envelope(
    base: RenderCamera, settings: CameraSettings, target_xy: Sequence[float]
) -> dict[str, Any]:
    """What every view of the settings can see: plate rectangle, camera positions, steepest ray.

    The static scene (plate, shroud, lights) is sized from this once, before the first layout.
    """
    if settings.is_rig_view:
        c = base.centre_in_plate
        return {
            "rect": base.visible_plate_rect(0.0),
            "max_offset_mm": 0.0,
            "max_height_mm": float(c[2]),
            "steepest_ray_deg": steepest_ray_deg(base),
        }
    h0 = base_height(base)
    heights = [h0] if settings.height == "calibrated" else extremes(settings.height)
    tilts = extremes(settings.tilt)
    dirs = extremes(settings.direction, 25)
    rolls = extremes(settings.roll, 13)
    rects, offsets, tops, steepest = [], [], [], 0.0
    for h in heights:
        for t in tilts:
            for p in dirs:
                for r in rolls:
                    cam = with_pose(base, view_pose(base, target_xy, h, t, p, r))
                    ray = steepest_ray_deg(cam)
                    steepest = max(steepest, ray)
                    if ray >= MAX_RAY_DEG:
                        continue
                    rects.append(cam.visible_plate_rect(0.0))
                    c = cam.centre_in_plate
                    offsets.append(math.hypot(c[0] - target_xy[0], c[1] - target_xy[1]))
                    tops.append(float(c[2]))
    if steepest >= MAX_RAY_DEG:
        raise ValueError(
            f"the steepest tilt shows the image corners {steepest:.0f} degrees from vertical "
            f"(limit {MAX_RAY_DEG:.0f}): the camera would see the horizon. Lower tilt_deg."
        )
    R = np.array(rects)
    return {
        "rect": (R[:, 0].min(), R[:, 1].min(), R[:, 2].max(), R[:, 3].max()),
        "max_offset_mm": max(offsets),
        "max_height_mm": max(tops),
        "steepest_ray_deg": steepest,
    }


def view_angle_deg(cam: RenderCamera, point: Sequence[float]) -> float:
    """Angle between the camera ray to a plate point and the vertical (0 = seen from straight above)."""
    c = cam.centre_in_plate
    d = np.asarray(point, dtype=float) - c
    return float(math.degrees(math.acos(min(1.0, abs(d[2]) / max(np.linalg.norm(d), 1e-9)))))


# ------------------------------------------------------------------------------- environment


@dataclass
class EnvironmentSettings:
    enabled: bool = False
    strength: float | tuple[float, float] = (0.3, 1.5)
    rotation: float | tuple[float, float] = (0.0, 360.0)
    panel_light: bool = True

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None) -> EnvironmentSettings:
        c = dict(section or {})
        return cls(
            enabled=bool(c.get("enabled", False)),
            strength=parse_value(c.get("strength", [0.3, 1.5]), "strength"),
            rotation=parse_value(c.get("rotation_deg", [0, 360]), "rotation_deg"),
            panel_light=bool(c.get("panel_light", True)),
        )


def layout_environment(
    files: Sequence[str], settings: EnvironmentSettings, index: int, seed: int
) -> dict[str, Any] | None:
    """HDRI file, strength and rotation of one layout; files are used equally often."""
    if not settings.enabled or not files:
        return None
    u = sequence_point(index, seed)
    # every block of len(files) layouts uses each file once, in a seeded shuffled order: equally
    # often for any number of layouts, and not tied to the other settings
    block, pos = divmod(index, len(files))
    order = list(files)
    random.Random(derive_seed(seed, "hdri-order", block)).shuffle(order)
    return {
        "file": order[pos],
        "strength": pick(settings.strength, u["hdri_strength"]),
        "rotation_deg": pick(settings.rotation, u["hdri_rotation"]) % 360.0,
    }
