"""HTML report with all calibration plots and checks (matplotlib, images embedded)."""

from __future__ import annotations

import base64
import html
import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img_tag(b64: str, title: str) -> str:
    return f'<figure><img src="data:image/png;base64,{b64}"/><figcaption>{html.escape(title)}</figcaption></figure>'


def _image_png(img: np.ndarray, max_w: int = 900) -> str:
    if img.shape[1] > max_w:
        img = cv2.resize(
            img, (max_w, int(img.shape[0] * max_w / img.shape[1])), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def intrinsic_figures(calib: Mapping[str, Any], residuals: np.ndarray | None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    w, h = calib["image_size"]
    views = [v for v in calib.get("views", []) if v["used"]]
    if residuals is not None and len(residuals):
        fig, ax = plt.subplots(figsize=(7, 7 * h / w))
        ax.scatter(residuals[:, 0], residuals[:, 1], s=2, c="tab:blue")
        ax.set(xlim=(0, w), ylim=(h, 0), title="Coverage: detected corners over all used views")
        out.append(
            _img_tag(_png(fig), "Coverage map (aim for the whole frame, especially corners)")
        )
        fig, ax = plt.subplots(figsize=(7, 7 * h / w))
        scale = 0.25 * w / max(np.abs(residuals[:, 2:]).max(), 1e-6) / 20
        ax.quiver(
            residuals[:, 0],
            residuals[:, 1],
            residuals[:, 2],
            -residuals[:, 3],
            angles="xy",
            scale_units="xy",
            scale=1 / scale,
            width=0.002,
        )
        ax.set(xlim=(0, w), ylim=(h, 0), title=f"Residuals (x{scale:.0f})")
        out.append(_img_tag(_png(fig), "Reprojection residual vectors (should look random)"))
    if views:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(views)), [v["rms_px"] for v in views])
        ax.axhline(0.5, color="r", ls="--", lw=1)
        ax.set(xlabel="view", ylabel="RMS px", title="Per-view reprojection error")
        out.append(_img_tag(_png(fig), "Per-view error (retake views far above the others)"))
    return out


def sensor_figures(profile: Mapping[str, Any], flatfield: np.ndarray | None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    bins = profile.get("noise_bins", {})
    if bins:
        fig, ax = plt.subplots(figsize=(6, 4))
        for ch, col in zip(("r", "g", "b"), ("tab:red", "tab:green", "tab:blue"), strict=True):
            m = np.array(bins[ch]["mean"])
            ax.plot(m, bins[ch]["var"], "o", ms=3, color=col)
            n = profile["noise"][ch]
            ax.plot(
                m,
                n["a"] * m + n["b"],
                "-",
                color=col,
                label=f"{ch}: var = {n['a']:.3f} I + {n['b']:.2f}",
            )
        ax.set(xlabel="mean intensity", ylabel="variance", title="Noise model")
        ax.legend()
        out.append(_img_tag(_png(fig), "Noise vs intensity"))
    if flatfield is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(flatfield, cmap="viridis")
        fig.colorbar(im)
        ax.set_title("Flat-field (normalised)")
        out.append(_img_tag(_png(fig), "Flat-field / vignetting"))
    return out


def write_report(
    calib_dir: str | os.PathLike[str],
    calib: Mapping[str, Any],
    profile: Mapping[str, Any] | None = None,
    residuals: np.ndarray | None = None,
    extrinsic_image: np.ndarray | None = None,
) -> Path:
    d = Path(calib_dir)
    parts = ["<h1>screw-synth camera calibration</h1>"]
    rows = []
    if "rms_px" in calib:
        flag = "OK" if calib["rms_px"] < 0.5 else "WARNING: above 0.5 px target"
        rows += [
            ("Intrinsics RMS", f"{calib['rms_px']:.3f} px ({flag})"),
            ("Distortion model", calib.get("distortion_model")),
            ("Hold-out RMS", calib.get("holdout_rms_px")),
            ("K", np.round(np.asarray(calib["camera_matrix"]), 3).tolist()),
            ("Distortion", np.round(np.asarray(calib["dist_coeffs"]), 6).tolist()),
        ]
    ext = calib.get("extrinsic")
    if ext:
        rows += [
            ("Extrinsic RMS", f"{ext['reprojection_rms_px']:.3f} px"),
            ("Camera height", f"{ext['camera_height_mm']:.2f} mm"),
            ("px per mm", f"{ext['px_per_mm']:.3f}"),
            ("ROI", ext["roi"]),
            ("Working area", ext["working_area_mm"]),
        ]
    if profile:
        rows += [
            ("Noise", profile["noise"]),
            ("Black level", profile["black_level"]),
            ("Empty plate RGB", profile["plate_empty"]["mean_rgb"]),
        ]
    if calib.get("validation"):
        v = calib["validation"]
        rows += [
            (
                "Validation",
                f"mean |err| {v['mean_abs_error_mm']:.3f} mm, max {v['max_abs_error_mm']:.3f} mm",
            )
        ]
    parts.append(
        "<table>"
        + "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in rows
        )
        + "</table>"
    )
    if "image_size" in calib and "views" in calib:
        parts += intrinsic_figures(calib, residuals)
    if extrinsic_image is not None and ext:
        img = extrinsic_image.copy()
        r = ext["roi"]
        cv2.rectangle(img, (r["x0"], r["y0"]), (r["x0"] + r["w"], r["y0"] + r["h"]), (0, 0, 255), 3)
        from .geometry import Camera

        cam = Camera.from_calibration(calib)
        axes = cam.project(np.array([[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50.0]]))
        o = tuple(int(v) for v in axes[0])
        for p, col in zip(axes[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), strict=True):
            cv2.arrowedLine(img, o, tuple(int(v) for v in p), col, 4)
        parts.append(
            _img_tag(
                _image_png(img), "Extrinsic: plate axes (x red, y green, z blue, 50 mm) and ROI"
            )
        )
    if profile:
        ff_path = d / "flatfield.png"
        ff = cv2.imread(str(ff_path), cv2.IMREAD_UNCHANGED) / 65535.0 if ff_path.is_file() else None
        parts += sensor_figures(profile, ff)
        pe = d / "plate_empty.png"
        if pe.is_file():
            parts.append(
                _img_tag(_image_png(cv2.imread(str(pe))), "plate_empty.png (undistorted ROI)")
            )
    style = (
        "body{font-family:sans-serif;max-width:1000px;margin:auto;padding:16px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}"
        "img{max-width:100%}figure{margin:16px 0}"
    )
    path = d / "report.html"
    path.write_text(
        f"<!doctype html><meta charset='utf-8'><title>Calibration report</title>"
        f"<style>{style}</style>{''.join(parts)}",
        encoding="utf-8",
    )
    return path
