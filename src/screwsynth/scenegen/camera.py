"""Render camera: intrinsics of the ROI crop and the plate -> camera pose (pure numpy).

Importable inside Blender (numpy + stdlib only).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

# OpenCV camera (x right, y down, z forward) -> Blender camera (x right, y up, looking along -z)
CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class RenderCamera:
    K_full: np.ndarray  # 3x3 full sensor image
    dist: np.ndarray
    image_size: tuple[int, int]  # full sensor (w, h)
    roi: dict[str, int]  # {x0, y0, w, h}
    T_cam_plate: np.ndarray  # 4x4 mm
    source: str = "calibration"
    scale: float = 1.0  # resolution scale for previews (applied to the ROI)

    @property
    def resolution(self) -> tuple[int, int]:
        return (
            max(1, round(self.roi["w"] * self.scale)),
            max(1, round(self.roi["h"] * self.scale)),
        )

    @property
    def K(self) -> np.ndarray:
        """Intrinsics of the rendered image: ROI crop (principal point shifted), then scaled.

        Scaling keeps pixel centres consistent: u' = (u + 0.5) * s - 0.5.
        """
        K = np.array(self.K_full, dtype=float)
        K[0, 2] -= self.roi["x0"]
        K[1, 2] -= self.roi["y0"]
        if self.scale != 1.0:
            sx = self.resolution[0] / self.roi["w"]
            sy = self.resolution[1] / self.roi["h"]
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
            K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        return K

    def cam2world_blender(self, world_scale: float) -> np.ndarray:
        """Blender camera pose in the scaled Blender world (= plate frame * 0.001 * world_scale)."""
        cam2plate = np.linalg.inv(self.T_cam_plate)
        cam2plate[:3, 3] *= 0.001 * world_scale
        return cam2plate @ CV_TO_BLENDER

    def project(self, X_plate: np.ndarray) -> np.ndarray:
        """Undistorted pinhole projection of plate points (mm) into render pixels."""
        X = np.asarray(X_plate, dtype=float).reshape(-1, 3)
        Xc = X @ self.T_cam_plate[:3, :3].T + self.T_cam_plate[:3, 3]
        uv = Xc @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def depth(self, X_plate: np.ndarray) -> np.ndarray:
        X = np.asarray(X_plate, dtype=float).reshape(-1, 3)
        return (X @ self.T_cam_plate[:3, :3].T + self.T_cam_plate[:3, 3])[:, 2]

    @property
    def centre_in_plate(self) -> np.ndarray:
        return -self.T_cam_plate[:3, :3].T @ self.T_cam_plate[:3, 3]

    def visible_plate_rect(self, margin_mm: float = 0.0) -> tuple[float, float, float, float]:
        """Axis-aligned plate rectangle (x0, y0, x1, y1) seen by the render (z = 0)."""
        w, h = self.resolution
        corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=float)
        Kinv = np.linalg.inv(self.K)
        R = self.T_cam_plate[:3, :3]
        c = self.centre_in_plate
        d = (np.c_[corners, np.ones(4)] @ Kinv.T) @ R  # ray directions in the plate frame
        s = -c[2] / d[:, 2]
        P = c[None, :2] + s[:, None] * d[:, :2]
        return (
            P[:, 0].min() - margin_mm,
            P[:, 1].min() - margin_mm,
            P[:, 0].max() + margin_mm,
            P[:, 1].max() + margin_mm,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "K": self.K.tolist(),
            "K_full": np.asarray(self.K_full).tolist(),
            "dist_coeffs": np.asarray(self.dist).ravel().tolist(),
            "image_size_full": list(self.image_size),
            "roi": dict(self.roi),
            "resolution": list(self.resolution),
            "resolution_scale": self.scale,
            "T_cam_plate": np.asarray(self.T_cam_plate).tolist(),
            "distorted": False,
        }


def from_calibration(calib: Mapping[str, Any], scale: float = 1.0) -> RenderCamera:
    ext = calib.get("extrinsic")
    if not ext or "roi" not in calib:
        raise ValueError(
            "calibration.json has no extrinsic/ROI: run screwsynth-calib extrinsic (or --nominal-camera)"
        )
    return RenderCamera(
        K_full=np.asarray(calib["camera_matrix"], dtype=float),
        dist=np.asarray(calib["dist_coeffs"], dtype=float).ravel(),
        image_size=tuple(calib["image_size"]),
        roi=dict(calib["roi"]),
        T_cam_plate=np.asarray(ext["T_cam_plate"], dtype=float),
        source="calibration",
        scale=scale,
    )


def nominal(
    focal_mm: float = 4.74,
    pixel_um: float = 1.4,
    size: tuple[int, int] = (4608, 2592),
    height_mm: float = 300.0,
    area_origin: tuple[float, float] = (0.0, 0.0),
    area_size: tuple[float, float] = (220.0, 220.0),
    roi_px: int = 2560,
    scale: float = 1.0,
) -> RenderCamera:
    """Nominal Pi Camera 3: straight down over the working-area centre, square ROI in the middle."""
    f = focal_mm / (pixel_um * 1e-3)
    w, h = size
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    cx, cy = area_origin[0] + area_size[0] / 2, area_origin[1] + area_size[1] / 2
    R = np.diag([1.0, -1.0, -1.0])  # camera x = plate x, camera y = -plate y, looking down
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.array([cx, cy, height_mm])
    side = min(roi_px, h)
    roi = {"x0": (w - side) // 2, "y0": (h - side) // 2, "w": side, "h": side}
    return RenderCamera(K, np.zeros(5), size, roi, T, source="nominal", scale=scale)


def blender_fov_check(cam: RenderCamera) -> dict[str, float]:
    """Horizontal/vertical field of view of the render (degrees), for logging."""
    w, h = cam.resolution
    K = cam.K
    return {
        "fov_x_deg": math.degrees(2 * math.atan(w / 2 / K[0, 0])),
        "fov_y_deg": math.degrees(2 * math.atan(h / 2 / K[1, 1])),
    }
