"""Small orthographic previews (front + top) rendered with numpy point splatting and Pillow."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import trimesh
from PIL import Image, ImageDraw

BACKGROUND = (236, 238, 242)
SLOT_COLOURS = {"body": np.array([150, 158, 170]), "insert": np.array([40, 90, 200])}
KP_COLOURS = {"head_top": (220, 40, 40), "bearing": (30, 150, 40), "tip": (40, 80, 220)}
LIGHT = np.array([-0.4, 0.5, 0.77])

# view rotations: rows are the image x axis, image y axis (up) and the viewing axis (towards viewer)
VIEWS = {
    "front": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
    "top": np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
}


def _face_colours(mesh: trimesh.Trimesh, slots: Mapping[str, Sequence[int]] | None) -> np.ndarray:
    colours = np.tile(SLOT_COLOURS["body"], (len(mesh.faces), 1))
    for name, (a, b) in (slots or {}).items():
        colours[a:b] = SLOT_COLOURS.get(name, SLOT_COLOURS["body"])
    return colours


def render_view(
    mesh: trimesh.Trimesh,
    view: str,
    px_per_mm: float,
    size: tuple[int, int],
    centre: np.ndarray,
    slots: Mapping[str, Sequence[int]] | None = None,
    keypoints: Mapping[str, Sequence[float]] | None = None,
) -> Image.Image:
    R = VIEWS[view]
    out_size = size
    # Point splatting needs a few samples per pixel or hidden surfaces show through the gaps.
    # With a capped sample budget, render smaller and upscale instead.
    budget = 400_000
    f = min(1.0, (budget / max(mesh.area * px_per_mm**2 * 3, 1.0)) ** 0.5)
    px_per_mm *= f
    w, h = max(1, int(size[0] * f)), max(1, int(size[1] * f))
    n = int(min(budget, max(20_000, mesh.area * px_per_mm**2 * 3)))
    pts, fidx = trimesh.sample.sample_surface(mesh, n, seed=0)
    v = (pts - centre) @ R.T
    normals = mesh.face_normals[fidx] @ R.T
    shade = 0.3 + 0.7 * np.clip(normals @ (LIGHT / np.linalg.norm(LIGHT)), 0, 1)
    rgb = (_face_colours(mesh, slots)[fidx] * shade[:, None]).clip(0, 255).astype(np.uint8)
    ix = np.round(v[:, 0] * px_per_mm + w / 2).astype(int)
    iy = np.round(h / 2 - v[:, 1] * px_per_mm).astype(int)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy, depth, rgb = ix[ok], iy[ok], v[ok, 2], rgb[ok]
    order = np.argsort(-depth)  # nearest first
    lin = (iy * w + ix)[order]
    _, first = np.unique(lin, return_index=True)
    img = np.empty((h * w, 3), dtype=np.uint8)
    img[:] = BACKGROUND
    img[lin[first]] = rgb[order][first]
    image = Image.fromarray(img.reshape(h, w, 3))
    if (w, h) != out_size:
        image = image.resize(out_size, Image.Resampling.BILINEAR)
    if keypoints:
        draw = ImageDraw.Draw(image)
        s = 1 / f
        for name, p in keypoints.items():
            q = (np.asarray(p) - centre) @ R.T
            x, y = (q[0] * px_per_mm + w / 2) * s, (h / 2 - q[1] * px_per_mm) * s
            draw.ellipse(
                [x - 4, y - 4, x + 4, y + 4], outline=KP_COLOURS.get(name, (0, 0, 0)), width=2
            )
    return image


def render_preview(
    mesh: trimesh.Trimesh,
    title: str,
    height_px: int = 360,
    slots: Mapping[str, Sequence[int]] | None = None,
    keypoints: Mapping[str, Sequence[float]] | None = None,
) -> Image.Image:
    """Front and top orthographic views side by side, same scale, keypoints circled."""
    lo, hi = mesh.bounds
    centre = (lo + hi) / 2
    ext = hi - lo
    margin = 0.12
    px_per_mm = height_px * (1 - 2 * margin) / max(ext[2], ext[1], 1e-6)
    width = int(max(ext[0], 1e-6) * px_per_mm / (1 - 2 * margin)) + 2
    width = max(width, height_px // 2)
    views = [
        render_view(mesh, name, px_per_mm, (width, height_px), centre, slots, keypoints)
        for name in VIEWS
    ]
    pad = 8
    sheet = Image.new("RGB", (2 * width + 3 * pad, height_px + 30), BACKGROUND)
    for i, img in enumerate(views):
        sheet.paste(img, (pad + i * (width + pad), 24))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (pad, 6),
        f"{title}   front | top   ({ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm)",
        fill=(20, 20, 20),
    )
    return sheet


def contact_sheet(
    images: Sequence[Image.Image],
    path: str | os.PathLike[str],
    columns: int = 4,
    cell: tuple[int, int] = (560, 340),
) -> None:
    """Tile preview images into one PNG; each image is fitted (aspect kept) into a fixed cell."""
    if not images:
        raise ValueError("no images for the contact sheet")
    cw, ch = cell
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cw, rows * ch), BACKGROUND)
    for i, im in enumerate(images):
        scale = min((cw - 8) / im.width, (ch - 8) / im.height)
        fitted = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        x = (i % columns) * cw + (cw - fitted.width) // 2
        y = (i // columns) * ch + (ch - fitted.height) // 2
        sheet.paste(fitted, (x, y))
    sheet.save(path)
