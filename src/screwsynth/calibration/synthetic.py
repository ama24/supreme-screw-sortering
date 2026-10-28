"""Synthetic captures with known ground truth, for tests and the CI sample set."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from screwsynth.common.io import write_json

from .board import BoardSpec, board_image
from .geometry import Camera

SMALL_SIZE = (1280, 960)


def small_camera(dist: tuple[float, ...] = (-0.12, 0.05, 0.0005, -0.0003, 0.0)) -> Camera:
    """A downscaled Pi-like camera with noticeable distortion (fast tests)."""
    w, h = SMALL_SIZE
    K = np.array([[1100.0, 0, w / 2 + 7.5], [0, 1102.0, h / 2 - 5.0], [0, 0, 1]])
    return Camera(K, np.array(dist, dtype=float), SMALL_SIZE)


def pose_matrix(
    rx_deg: float, ry_deg: float, rz_deg: float, t: tuple[float, float, float]
) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.radians([rx_deg, ry_deg, rz_deg]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def render_plane(
    cam: Camera,
    T_cam_plane: np.ndarray,
    texture: np.ndarray,
    px_per_mm: float,
    background: int | tuple[int, ...] = 90,
    origin_mm: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Render a planar texture (plane frame: x right, y down the texture, z = 0) through cam."""
    w, h = cam.image_size
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
    n = cv2.undistortPoints(pts, cam.K, cam.dist).reshape(-1, 2)
    d = np.c_[n, np.ones(len(n))]
    R, t = T_cam_plane[:3, :3], T_cam_plane[:3, 3]
    c = -R.T @ t
    dp = d @ R  # directions in the plane frame
    s = -c[2] / dp[:, 2]
    X = c[None, :2] + s[:, None] * dp[:, :2]
    ok = s > 0
    mx = ((X[:, 0] - origin_mm[0]) * px_per_mm).astype(np.float32)
    my = ((X[:, 1] - origin_mm[1]) * px_per_mm).astype(np.float32)
    mx[~ok] = -1
    my[~ok] = -1
    if isinstance(background, int):
        background = (background,) * (texture.shape[2] if texture.ndim == 3 else 1)
    out = cv2.remap(
        texture,
        mx.reshape(h, w),
        my.reshape(h, w),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    return out


def render_board(
    cam: Camera,
    spec: BoardSpec,
    T_cam_board: np.ndarray,
    rng: np.random.Generator,
    noise: float = 2.0,
    blur: float = 0.6,
) -> np.ndarray:
    px_per_mm = 12.0
    tex = board_image(spec, dpi=round(px_per_mm * 25.4))
    tex = cv2.GaussianBlur(tex, (0, 0), 0.8)
    img = render_plane(cam, T_cam_board, tex, tex.shape[1] / spec.size_mm[0], 70)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), blur)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def intrinsic_views(n: int = 14, seed: int = 0) -> list[np.ndarray]:
    """Board poses covering the whole frame (3 x 3 grid + extras) with +/-15 deg tilts, ~300 mm away."""
    rng = np.random.default_rng(seed)
    centres = [(x, y) for y in (-75, 0, 75) for x in (-100, 0, 100)]
    centres += [
        (rng.uniform(-60, 60), rng.uniform(-45, 45)) for _ in range(max(0, n - len(centres)))
    ]
    poses = []
    for cx, cy in centres[:n]:
        rx, ry = rng.uniform(-15, 15, 2)
        rz = rng.uniform(-20, 20)
        poses.append(pose_matrix(rx, ry, rz, (cx - 67.5, cy - 52.5, rng.uniform(285, 315))))
    return poses


def make_sample(out_dir: str | os.PathLike[str], seed: int = 0) -> dict:
    """Write a tiny synthetic capture set with ground truth (intrinsics, extrinsic, sensor)."""
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    cam = small_camera()
    spec = BoardSpec()
    (out / "intrinsics").mkdir(parents=True, exist_ok=True)
    for i, T in enumerate(intrinsic_views(14, seed)):
        cv2.imwrite(str(out / "intrinsics" / f"view_{i:02d}.png"), render_board(cam, spec, T, rng))
    # extrinsic: board flat on the plate, camera looking straight down from 300 mm
    T_board = pose_matrix(0.0, 0.0, 3.0, (-60, -50, 300))
    (out / "extrinsic").mkdir(exist_ok=True)
    cv2.imwrite(str(out / "extrinsic" / "board_flat.png"), render_board(cam, spec, T_board, rng))
    # sensor frames: green plate with vignetting and signal-dependent noise var = a*I + b
    (out / "sensor").mkdir(exist_ok=True)
    a, b = 0.6, 2.0
    w, h = cam.image_size
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w / 2) ** 2
    vignette = 1 - 0.25 * r2
    plate = np.array([60, 150, 70], dtype=float)  # BGR
    ramp = np.linspace(0.3, 1.2, w)[None, :, None]
    scene = plate[None, None, :] * ramp * vignette[..., None]
    for i in range(8):
        noisy = scene + rng.normal(0, 1, scene.shape) * np.sqrt(a * scene + b)
        cv2.imwrite(
            str(out / "sensor" / f"frame_{i:02d}.png"), np.clip(noisy, 0, 255).astype(np.uint8)
        )
    white = 230 * vignette[..., None] * np.ones(3)
    cv2.imwrite(
        str(out / "sensor" / "white.png"),
        np.clip(white + rng.normal(0, 1.5, white.shape), 0, 255).astype(np.uint8),
    )
    empty = plate[None, None, :] * vignette[..., None]
    cv2.imwrite(
        str(out / "sensor" / "empty.png"),
        np.clip(empty + rng.normal(0, 1.5, empty.shape), 0, 255).astype(np.uint8),
    )
    truth = {
        "camera_matrix": cam.K.tolist(),
        "dist_coeffs": cam.dist.tolist(),
        "image_size": list(cam.image_size),
        "T_cam_board_extrinsic": T_board.tolist(),
        "noise": {"a": a, "b": b},
        "vignette": "1 - 0.25 * r^2, r normalised by half width",
        "board": spec.to_dict(),
    }
    write_json(out / "truth.json", truth)
    return truth
