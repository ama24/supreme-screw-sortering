"""Helpers shared by the Raspberry Pi capture scripts (no Picamera2 import here, so it is testable).

Copy this folder to the Pi; it does not need the screwsynth package.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("pi")

FULL_RES = (4608, 2592)
# tolerances when checking that the camera really runs with the locked settings
TOLERANCE = {"ExposureTime": 0.01, "AnalogueGain": 0.02, "ColourGains": 0.01, "LensPosition": 0.02}


def focus_metric(gray: np.ndarray, centre_fraction: float = 0.4) -> float:
    """Variance of the Laplacian on the centre region (higher = sharper)."""
    h, w = gray.shape[:2]
    ch, cw = int(h * centre_fraction), int(w * centre_fraction)
    crop = gray[(h - ch) // 2 : (h + ch) // 2, (w - cw) // 2 : (w + cw) // 2].astype(np.float64)
    lap = (
        -4 * crop[1:-1, 1:-1] + crop[:-2, 1:-1] + crop[2:, 1:-1] + crop[1:-1, :-2] + crop[1:-1, 2:]
    )
    return float(lap.var())


def best_focus(positions: list[float], scores: list[float]) -> float:
    """Peak of the focus curve with a parabolic fit around the best sample."""
    i = int(np.argmax(scores))
    if 0 < i < len(scores) - 1:
        x = np.array(positions[i - 1 : i + 2])
        y = np.array(scores[i - 1 : i + 2])
        a, b, _ = np.polyfit(x, y, 2)
        if a < 0:
            return float(np.clip(-b / (2 * a), x[0], x[-1]))
    return float(positions[i])


def settings_mismatch(locked: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    """Differences between the locked settings and the metadata of an actual frame."""
    problems = []
    for key, tol in TOLERANCE.items():
        if key not in locked:
            continue
        want, got = locked[key], metadata.get(key)
        if got is None:
            problems.append(f"{key}: missing in frame metadata")
            continue
        want_a, got_a = (
            np.atleast_1d(np.asarray(want, float)),
            np.atleast_1d(np.asarray(got, float)),
        )
        rel = np.abs(got_a - want_a) / np.maximum(np.abs(want_a), 1e-9)
        if np.any(rel > tol):
            problems.append(f"{key}: locked {want}, camera reports {got}")
    if (
        locked.get("size")
        and metadata.get("size")
        and list(locked["size"]) != list(metadata["size"])
    ):
        problems.append(f"size: locked {locked['size']}, got {metadata['size']}")
    return problems


def controls_from_settings(s: dict[str, Any]) -> dict[str, Any]:
    """Picamera2 controls that freeze exposure, gain, white balance and focus."""
    c = {
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": int(s["ExposureTime"]),
        "AnalogueGain": float(s["AnalogueGain"]),
        "ColourGains": tuple(float(g) for g in s["ColourGains"]),
        "AfMode": 0,  # controls.AfModeEnum.Manual
        "LensPosition": float(s["LensPosition"]),
    }
    for key in ("Sharpness", "Contrast", "Saturation", "Brightness", "NoiseReductionMode"):
        if key in s:
            c[key] = s[key]
    return c


class Coverage:
    """Where board corners have been seen across the frame (grid of counts)."""

    def __init__(self, size: tuple[int, int] = FULL_RES, cells: tuple[int, int] = (16, 9)) -> None:
        self.size, self.cells = size, cells
        self.grid = np.zeros((cells[1], cells[0]), dtype=int)

    def add(self, corners_px: np.ndarray) -> None:
        w, h = self.size
        cx = np.clip((corners_px[:, 0] / w * self.cells[0]).astype(int), 0, self.cells[0] - 1)
        cy = np.clip((corners_px[:, 1] / h * self.cells[1]).astype(int), 0, self.cells[1] - 1)
        np.add.at(self.grid, (cy, cx), 1)

    @property
    def fraction(self) -> float:
        return float((self.grid > 0).mean())

    def ascii(self) -> str:
        chars = " .:-=+*#"
        top = max(int(self.grid.max()), 1)
        return "\n".join(
            "".join(chars[min(7, int(v * 7 / top))] if v else "_" for v in row) for row in self.grid
        )

    def image(self, scale: int = 40) -> np.ndarray:
        g = (np.clip(self.grid / max(self.grid.max(), 1), 0, 1) * 255).astype(np.uint8)
        return np.kron(g, np.ones((scale, scale), dtype=np.uint8))


def save_capture(
    out_dir: str | os.PathLike[str],
    name: str,
    image_bgr: np.ndarray,
    metadata: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Lossless PNG + JSON sidecar with the actual frame metadata."""
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"could not write {path}")
    sidecar = {
        "file": path.name,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metadata": {k: _jsonable(v) for k, v in metadata.items()},
        **(extra or {}),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return path


def _jsonable(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if hasattr(v, "tolist"):
        return v.tolist()
    return str(v)
