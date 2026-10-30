"""Plate backgrounds: what the surface under the parts looks like (pure Python, Blender-safe).

Three sources, one drawn per scene by `scenegen.plate.background_weights`:
- calibrated_photo: the calibrated camera's photo of the empty plate (data/calibration/plate_empty.png),
  mapped pixel-exact through the camera; only available with the calibrated camera;
- images: photos of surfaces (craft mat, bare table, ...) in paths.plate_dir, listed in
  backgrounds.yaml with their real width in mm, laid on the plate at true scale with a random
  rotation and offset (mirror-tiled when smaller than the plate);
- colours: one of `scenegen.plate.colours` (name -> sRGB hex), brightness-jittered.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LIBRARY = "backgrounds.yaml"
SOURCES = ("calibrated_photo", "images", "colours")
DEFAULT_COLOURS = [{"name": "silicone_blue", "hex": "#3A86D4"}]  # silicone phone-repair mat
DEFAULT_WEIGHTS = {"calibrated_photo": 0.6, "images": 0.2, "colours": 0.2}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


@dataclass
class BackgroundImage:
    file: str  # file name inside the plate dir
    width_mm: float  # real width of the whole photo
    weight: float = 1.0
    enabled: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------- colours


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def parse_colour(value: Any) -> tuple[float, float, float]:
    """'#RRGGBB' (sRGB, what colour pickers show) or [r, g, b] (linear, 0-1) -> linear RGB."""
    if isinstance(value, str):
        h = value.strip().lstrip("#")
        if len(h) != 6:
            raise ValueError(f"colour {value!r}: use #RRGGBB")
        rgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        return tuple(srgb_to_linear(c) for c in rgb)  # type: ignore[return-value]
    r, g, b = (float(x) for x in value)
    if not all(0 <= x <= 1 for x in (r, g, b)):
        raise ValueError(f"colour {value!r}: linear RGB components must be within 0-1")
    return r, g, b


def to_hex(linear: Sequence[float]) -> str:
    return "#" + "".join(f"{round(min(1, max(0, linear_to_srgb(c))) * 255):02X}" for c in linear)


def colour_list(plate: Mapping[str, Any]) -> list[dict[str, str]]:
    """`colours` as [{name, hex}] (also accepts a name -> colour mapping and the older
    `solid_colours` list of linear RGB). A list, so a recipe can remove the default colour."""
    raw = plate.get("colours")
    if raw is None and plate.get("solid_colours") is not None:  # configs before backgrounds
        raw = [
            {"name": f"colour_{i + 1}", "hex": to_hex(parse_colour(c))}
            for i, c in enumerate(plate["solid_colours"])
        ]
    if raw is None:
        raw = DEFAULT_COLOURS
    if isinstance(raw, Mapping):
        raw = [
            {"name": k, "hex": v if isinstance(v, str) else to_hex(parse_colour(v))}
            for k, v in raw.items()
        ]
    return [{"name": str(c["name"]), "hex": str(c["hex"]).upper()} for c in raw]


def colours_from_config(plate: Mapping[str, Any]) -> dict[str, tuple[float, float, float]]:
    """name -> linear RGB of the solid colours."""
    return {c["name"]: parse_colour(c["hex"]) for c in colour_list(plate)}


# --------------------------------------------------------------------------------- library


def load_library(plate_dir: Path) -> list[BackgroundImage]:
    """backgrounds.yaml of the plate dir (missing file = empty library)."""
    from screwsynth.common.io import read_yaml

    path = Path(plate_dir) / LIBRARY
    if not path.is_file():
        return []
    data = read_yaml(path) or {}
    out = []
    for raw in data.get("backgrounds", []) or []:
        out.append(
            BackgroundImage(
                file=str(raw["file"]),
                width_mm=float(raw["width_mm"]),
                weight=float(raw.get("weight", 1.0)),
                enabled=bool(raw.get("enabled", True)),
                note=str(raw.get("note", "")),
            )
        )
    return out


def save_library(plate_dir: Path, images: Iterable[BackgroundImage]) -> Path:
    from screwsynth.common.io import write_yaml

    path = Path(plate_dir) / LIBRARY
    write_yaml(path, {"backgrounds": [im.to_dict() for im in images]})
    return path


def usable_images(plate_dir: Path, library: Iterable[BackgroundImage]) -> list[BackgroundImage]:
    """Enabled images with a positive weight and width whose file exists."""
    return [
        im
        for im in library
        if im.enabled
        and im.weight > 0
        and im.width_mm > 0
        and (Path(plate_dir) / im.file).is_file()
    ]


def width_from_ruler(
    image_width_px: float, p1: Sequence[float], p2: Sequence[float], distance_mm: float
) -> float:
    """Real width of a photo from two points on a ruler in it and their distance (mm)."""
    d_px = math.dist(p1, p2)
    if d_px <= 0 or distance_mm <= 0:
        raise ValueError("pick two different points and a positive distance")
    return image_width_px * distance_mm / d_px


# --------------------------------------------------------------------------------- choice


def weights(plate: Mapping[str, Any], have_photo: bool, have_images: bool) -> dict[str, float]:
    """Source weights with the unavailable sources removed (the rest renormalised by choice)."""
    raw = plate.get("background_weights")
    if raw is None and "solid_colour_probability" in plate:  # configs before backgrounds
        p = float(plate["solid_colour_probability"])
        raw = {"calibrated_photo": 1 - p, "images": 0.0, "colours": p}
    if raw is None:
        raw = DEFAULT_WEIGHTS
    w = {s: max(0.0, float(dict(raw).get(s, DEFAULT_WEIGHTS[s]))) for s in SOURCES}
    if not have_photo:
        w["calibrated_photo"] = 0.0
    if not have_images:
        w["images"] = 0.0
    if sum(w.values()) <= 0:
        w["colours"] = 1.0  # always something to show
    return w


def choose(
    plate: Mapping[str, Any],
    images: Sequence[BackgroundImage],
    have_photo: bool,
    rng: random.Random,
) -> dict[str, Any]:
    """The background of one scene (deterministic from rng)."""
    w = weights(plate, have_photo, bool(images))
    kinds = [s for s in SOURCES if w[s] > 0]
    kind = rng.choices(kinds, weights=[w[k] for k in kinds])[0]
    if kind == "calibrated_photo":
        return {"kind": kind}
    if kind == "images":
        im = rng.choices(list(images), weights=[i.weight for i in images])[0]
        return {
            "kind": kind,
            "file": im.file,
            "width_mm": im.width_mm,
            "rotation_deg": rng.uniform(0, 360),
            "offset": [rng.random(), rng.random()],  # in image widths/heights
        }
    colours = colours_from_config(plate)
    name = rng.choice(sorted(colours))
    jitter = float(plate.get("colour_jitter", 0.15))
    k = 1 + rng.uniform(-jitter, jitter)
    rgb = [min(1.0, max(0.0, c * k)) for c in colours[name]]
    return {"kind": kind, "name": name, "rgb": rgb, "hex": to_hex(rgb)}
