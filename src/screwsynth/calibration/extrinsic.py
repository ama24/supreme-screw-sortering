"""Extrinsic calibration: pose of the plate frame from a ChArUco board lying flat on the TPU plate.

Plate frame: origin at the board's first chessboard corner, x along the board's x (squares_x),
z up towards the camera, y = z x x, z = 0 on the TPU surface (board surface at z = thickness).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from .board import BoardSpec, detect
from .geometry import Camera, roi_for_area

log = logging.getLogger(__name__)

# board frame (OpenCV: z away from the camera) -> plate frame (z towards the camera)
_FLIP = np.diag([1.0, -1.0, -1.0])


def board_pose(image: np.ndarray, spec: BoardSpec, cam: Camera) -> tuple[np.ndarray, float, int]:
    """T_cam_board (4x4), reprojection RMS (px) and number of corners."""
    d = detect(image, spec)
    if d is None or d.n < 6:
        raise ValueError("ChArUco board not found (need >= 6 corners)")
    ok, rvec, tvec = cv2.solvePnP(d.object_points, d.corners, cam.K, cam.dist)
    rvec, tvec = cv2.solvePnPRefineLM(d.object_points, d.corners, cam.K, cam.dist, rvec, tvec)
    proj, _ = cv2.projectPoints(d.object_points, rvec, tvec, cam.K, cam.dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - d.corners) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, rms, d.n


def plate_from_board(T_cam_board: np.ndarray, thickness_mm: float) -> np.ndarray:
    """X_board = FLIP X_plate + (0, 0, thickness)  =>  T_cam_plate."""
    o = np.array([0.0, 0.0, thickness_mm])
    T = np.eye(4)
    T[:3, :3] = T_cam_board[:3, :3] @ _FLIP
    T[:3, 3] = T_cam_board[:3, :3] @ o + T_cam_board[:3, 3]
    return T


def corner_marker_area(
    image: np.ndarray, cam: Camera, ids: Sequence[int], dictionary: str
) -> tuple[list[float], list[float]] | None:
    """Working area = bounding box (plate mm) of the centres of the plate-corner ArUco markers."""
    if not ids:
        return None
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)))
    corners, found, _ = det.detectMarkers(gray)
    if found is None:
        return None
    centres = {
        int(i): c.reshape(4, 2).mean(axis=0) for c, i in zip(corners, found.ravel(), strict=True)
    }
    if not all(i in centres for i in ids):
        log.warning("corner markers found: %s, expected %s", sorted(centres), list(ids))
        return None
    pts = cam.pixel_to_plate(np.array([centres[i] for i in ids]))
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    return lo.tolist(), (hi - lo).tolist()


def solve(
    image: np.ndarray,
    spec: BoardSpec,
    cam: Camera,
    thickness_mm: float = 0.0,
    area: Mapping[str, Any] | None = None,
    corner_ids: Sequence[int] = (),
    corner_dictionary: str = "DICT_4X4_50",
    roi_multiple: int = 32,
) -> dict[str, Any]:
    T_cb, rms, n = board_pose(image, spec, cam)
    T = plate_from_board(T_cb, thickness_mm)
    cam = Camera(cam.K, cam.dist, cam.image_size, T)
    area = dict(area or {})
    size = list(area.get("size_mm", [220.0, 220.0]))
    margin = float(area.get("margin_mm", 5.0))
    source = "config"
    found = corner_marker_area(image, cam, list(corner_ids), corner_dictionary)
    if found is not None:
        origin, size = found
        source = "corner_markers"
    elif area.get("origin_mm") is not None:
        origin = list(area["origin_mm"])
    else:
        bw, bh = spec.size_mm
        origin = [bw / 2 - size[0] / 2, -bh / 2 - size[1] / 2]  # plate y = -board y
        source = "centred_on_board (set calibration.working_area.origin_mm!)"
        log.warning("working area not given; centring it on the board. Set working_area.origin_mm")
    roi = roi_for_area(cam, origin, size, margin, roi_multiple)
    centre = [origin[0] + size[0] / 2, origin[1] + size[1] / 2]
    return {
        "T_cam_plate": T.tolist(),
        "T_cam_board": T_cb.tolist(),
        "board_thickness_mm": thickness_mm,
        "reprojection_rms_px": rms,
        "corners": n,
        "camera_height_mm": float(cam.centre_in_plate[2]),
        "camera_centre_plate_mm": cam.centre_in_plate.tolist(),
        "plate_homography": cam.homography().tolist(),
        "px_per_mm": cam.px_per_mm(centre),
        "working_area_mm": {"origin": origin, "size": size, "margin": margin, "source": source},
        "roi": roi,
    }
