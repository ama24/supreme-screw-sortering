"""ChArUco board definition, detection, printable PDF and plate-corner ArUco markers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 9
    squares_y: int = 7
    square_mm: float = 15.0
    marker_mm: float = 11.0
    dictionary: str = "DICT_5X5_100"

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> BoardSpec:
        keys = cls.__dataclass_fields__
        return cls(**{k: m[k] for k in keys if k in m})

    def with_square(self, square_mm: float | None) -> BoardSpec:
        """Use the MEASURED square size; the marker scales with it."""
        if not square_mm:
            return self
        f = square_mm / self.square_mm
        return BoardSpec(
            self.squares_x, self.squares_y, square_mm, self.marker_mm * f, self.dictionary
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cv_dictionary(self) -> cv2.aruco.Dictionary:
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))

    def cv_board(self) -> cv2.aruco.CharucoBoard:
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_mm, self.marker_mm, self.cv_dictionary()
        )

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def size_mm(self) -> tuple[float, float]:
        return self.squares_x * self.square_mm, self.squares_y * self.square_mm


@dataclass
class Detection:
    corners: np.ndarray  # (N, 2) pixels
    ids: np.ndarray  # (N,)
    object_points: np.ndarray  # (N, 3) board frame, mm
    sharpness: float

    @property
    def n(self) -> int:
        return len(self.ids)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian (focus/blur metric)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect(image: np.ndarray, spec: BoardSpec) -> Detection | None:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    board = spec.cv_board()
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 4:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    # sharpness around the detected board only
    x0, y0 = np.floor(corners.reshape(-1, 2).min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.reshape(-1, 2).max(axis=0)).astype(int)
    crop = gray[max(y0, 0) : y1 + 1, max(x0, 0) : x1 + 1]
    return Detection(
        corners=img.reshape(-1, 2).astype(float),
        ids=ids.ravel(),
        object_points=obj.reshape(-1, 3).astype(float),
        sharpness=sharpness(crop) if crop.size else 0.0,
    )


# ------------------------------------------------------------------------------- printing


def board_image(spec: BoardSpec, dpi: int = 600) -> np.ndarray:
    px_per_mm = dpi / 25.4
    w_mm, h_mm = spec.size_mm
    return spec.cv_board().generateImage(
        (round(w_mm * px_per_mm), round(h_mm * px_per_mm)), marginSize=0, borderBits=1
    )


def write_board_pdf(spec: BoardSpec, path: str | os.PathLike[str], dpi: int = 600) -> None:
    """A4 PDF at exact scale: board, 100 mm scale bar and printing instructions."""
    from PIL import Image, ImageDraw

    px_per_mm = dpi / 25.4
    page = Image.new("L", (round(210 * px_per_mm), round(297 * px_per_mm)), 255)
    board = Image.fromarray(board_image(spec, dpi))
    left = round(20 * px_per_mm)
    top = round(30 * px_per_mm)
    page.paste(board, (left, top))
    draw = ImageDraw.Draw(page)
    y_bar = top + board.height + round(15 * px_per_mm)
    x1 = left + round(100 * px_per_mm)
    lw = max(2, round(0.3 * px_per_mm))
    draw.line([(left, y_bar), (x1, y_bar)], fill=0, width=lw)
    for k in range(11):
        x = left + round(k * 10 * px_per_mm)
        tick = round((4 if k % 5 == 0 else 2) * px_per_mm)
        draw.line([(x, y_bar - tick), (x, y_bar)], fill=0, width=lw)
    text = [
        "100 mm scale bar - check with calipers after printing",
        f"ChArUco {spec.squares_x} x {spec.squares_y} squares, {spec.square_mm} mm squares, "
        f"{spec.marker_mm} mm markers, {spec.dictionary}",
        "Print at 100% / actual size (no 'fit to page'). Glue flat onto glass or an aluminium plate.",
        "Measure several squares with calipers and pass the mean: screwsynth-calib intrinsics --square-mm X",
    ]
    font_px = round(3.2 * px_per_mm)
    try:
        from PIL import ImageFont

        font = ImageFont.load_default(size=font_px)
    except TypeError:  # Pillow < 10.1
        from PIL import ImageFont

        font = ImageFont.load_default()
    y = y_bar + round(8 * px_per_mm)
    for line in text:
        draw.text((left, y), line, fill=0, font=font)
        y += round(6 * px_per_mm)
    page.save(path, "PDF", resolution=dpi)


def corner_marker_svgs(
    ids: list[int], dictionary: str, size_mm: float, out_dir: str | os.PathLike[str]
) -> list[Path]:
    """One SVG per plate-corner ArUco marker (cell squares as paths, exact mm), for printing/3D."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    out = []
    for mid in ids:
        bits = d.generateImageMarker(mid, d.markerSize + 2) // 255  # 1 = white, incl. border
        n = bits.shape[0]
        cell = size_mm / n
        rects = [
            f'<rect x="{c * cell:.4f}" y="{r * cell:.4f}" width="{cell:.4f}" height="{cell:.4f}"/>'
            for r in range(n)
            for c in range(n)
            if bits[r, c] == 0
        ]
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_mm}mm" height="{size_mm}mm" '
            f'viewBox="0 0 {size_mm} {size_mm}"><rect width="{size_mm}" height="{size_mm}" fill="white"/>'
            f'<g fill="black">{"".join(rects)}</g></svg>\n'
        )
        path = Path(out_dir) / f"aruco_{dictionary}_{mid}.svg"
        path.write_text(svg, encoding="utf-8")
        out.append(path)
    return out
