"""Realism check: synthetic vs real image statistics as an HTML report."""

from __future__ import annotations

import base64
import html
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_images(folder: Path, limit: int) -> list[Path]:
    """Image files under a folder, excluding masks (instance*/mask*), sorted."""
    files = sorted(
        p
        for p in Path(folder).rglob("*")
        if p.suffix.lower() in IMAGE_EXT and not p.stem.startswith(("instance", "mask", "coverage"))
    )
    return files[:limit]


def plate_mask(img_rgb: np.ndarray, instance: np.ndarray | None = None) -> np.ndarray:
    """Background (plate) pixels: from the instance mask, or by colour distance for real images."""
    if instance is not None:
        return instance == 0
    from screwsynth.calibration.measure import segment_objects

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask = np.ones(img_rgb.shape[:2], dtype=bool)
    fg = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for c in segment_objects(bgr, min_area_px=50):
        cv2.fillPoly(fg, [c.astype(np.int32)], 255)
    fg = cv2.dilate(fg, np.ones((9, 9), np.uint8))
    mask[fg > 0] = False
    return mask


def noise_curve(img_rgb: np.ndarray, bg: np.ndarray, bins: int = 16) -> dict[str, list[float]]:
    """Noise level vs intensity from flat plate regions (residual after a 5x5 median)."""
    out = {}
    for c, ch in enumerate("rgb"):
        x = img_rgb[..., c].astype(np.float32)
        resid = x - cv2.medianBlur(img_rgb[..., c], 5).astype(np.float32)
        m, r = x[bg], resid[bg]
        if len(m) < 1000:
            out[ch] = {"mean": [], "std": []}
            continue
        edges = np.quantile(m, np.linspace(0, 1, bins + 1))
        means, stds = [], []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            sel = (m >= lo) & (m <= hi)
            if sel.sum() > 200:
                means.append(float(m[sel].mean()))
                stds.append(float(r[sel].std()))
        out[ch] = {"mean": means, "std": stds}
    return out


def edge_width(img_rgb: np.ndarray, fg: np.ndarray, samples: int = 400, seed: int = 0) -> float:
    """Median 10-90% edge rise distance (px) across object borders (smaller = sharper)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edges = cv2.Canny(fg.astype(np.uint8) * 255, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(ys) == 0:
        return float("nan")
    gx = cv2.Sobel(fg.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(fg.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ys), min(samples, len(ys)), replace=False)
    widths = []
    t = np.arange(-6, 6.01, 0.25)
    for i in idx:
        y, x = ys[i], xs[i]
        n = np.array([gx[y, x], gy[y, x]])
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        px = (x + t * n[0]).astype(np.float32)
        py = (y + t * n[1]).astype(np.float32)
        prof = cv2.remap(
            gray, px[None, :], py[None, :], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )[0]
        lo, hi = prof[:8].mean(), prof[-8:].mean()
        if abs(hi - lo) < 20:
            continue
        p = (prof - lo) / (hi - lo)
        try:
            a = t[np.argmax(p > 0.1)]
            b = t[np.argmax(p > 0.9)]
        except ValueError:
            continue
        if b > a:
            widths.append(b - a)
    return float(np.median(widths)) if widths else float("nan")


def image_stats(img_rgb: np.ndarray, instance: np.ndarray | None = None) -> dict[str, Any]:
    bg = plate_mask(img_rgb, instance)
    fg = ~bg
    hist = {
        ch: np.histogram(img_rgb[..., c], bins=64, range=(0, 256))[0].tolist()
        for c, ch in enumerate("rgb")
    }
    return {
        "plate_rgb_mean": img_rgb[bg].mean(axis=0).tolist() if bg.any() else None,
        "plate_rgb_std": img_rgb[bg].std(axis=0).tolist() if bg.any() else None,
        "histogram": hist,
        "noise": noise_curve(img_rgb, bg),
        "edge_width_px": edge_width(img_rgb, fg),
        "mean_brightness": float(img_rgb.mean()),
    }


def aggregate(stats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def mean_of(key):
        vals = [s[key] for s in stats if s[key] is not None]
        return np.mean(vals, axis=0).tolist() if vals else None

    hist = {ch: np.sum([s["histogram"][ch] for s in stats], axis=0).tolist() for ch in "rgb"}
    ew = [s["edge_width_px"] for s in stats if np.isfinite(s["edge_width_px"])]
    return {
        "n": len(stats),
        "plate_rgb_mean": mean_of("plate_rgb_mean"),
        "plate_rgb_std": mean_of("plate_rgb_std"),
        "histogram": hist,
        "noise": [s["noise"] for s in stats],
        "edge_width_px": float(np.median(ew)) if ew else float("nan"),
        "mean_brightness": float(np.mean([s["mean_brightness"] for s in stats])),
    }


def suggestions(syn: dict[str, Any], real: dict[str, Any]) -> list[str]:
    out = []
    if syn["plate_rgb_mean"] and real["plate_rgb_mean"]:
        d = np.asarray(syn["plate_rgb_mean"]) - np.asarray(real["plate_rgb_mean"])
        if np.abs(d).max() > 10:
            out.append(
                f"plate colour differs by {np.round(d, 1).tolist()} (RGB): check postprocess.colour.match_plate / lights"
            )
    if np.isfinite(syn["edge_width_px"]) and np.isfinite(real["edge_width_px"]):
        r = syn["edge_width_px"] / real["edge_width_px"]
        if r < 0.8:
            out.append(
                f"synthetic edges too sharp ({syn['edge_width_px']:.2f} vs {real['edge_width_px']:.2f} px): raise blur.sigma_px"
            )
        elif r > 1.25:
            out.append(
                f"synthetic edges too soft ({syn['edge_width_px']:.2f} vs {real['edge_width_px']:.2f} px): lower blur.sigma_px"
            )
    b = syn["mean_brightness"] - real["mean_brightness"]
    if abs(b) > 12:
        out.append(
            f"synthetic images {'brighter' if b > 0 else 'darker'} by {abs(b):.0f} levels: adjust scenegen.lights.energy_w or colour.gain"
        )
    return out or ["no large differences found"]


def _png(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _thumb(img_rgb: np.ndarray, w: int = 420) -> str:
    small = cv2.resize(
        img_rgb, (w, int(img_rgb.shape[0] * w / img_rgb.shape[1])), interpolation=cv2.INTER_AREA
    )
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    return base64.b64encode(buf.tobytes()).decode()


def _crop(img_rgb: np.ndarray, size: int = 200) -> str:
    h, w = img_rgb.shape[:2]
    c = img_rgb[(h - size) // 2 : (h + size) // 2, (w - size) // 2 : (w + size) // 2]
    c = cv2.resize(c, (size * 2, size * 2), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def report(
    syn_items: Sequence[tuple[np.ndarray, np.ndarray | None]],
    real_images: Sequence[np.ndarray],
    out_path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    syn = aggregate([image_stats(i, m) for i, m in syn_items])
    real = aggregate([image_stats(i) for i in real_images])
    parts = ["<h1>Synthetic vs real</h1>"]
    rows = [
        ("images", syn["n"], real["n"]),
        (
            "plate RGB mean",
            np.round(syn["plate_rgb_mean"] or [], 1).tolist(),
            np.round(real["plate_rgb_mean"] or [], 1).tolist(),
        ),
        (
            "plate RGB std",
            np.round(syn["plate_rgb_std"] or [], 2).tolist(),
            np.round(real["plate_rgb_std"] or [], 2).tolist(),
        ),
        ("edge width 10-90% (px)", round(syn["edge_width_px"], 2), round(real["edge_width_px"], 2)),
        ("mean brightness", round(syn["mean_brightness"], 1), round(real["mean_brightness"], 1)),
    ]
    parts.append(
        "<table><tr><th></th><th>synthetic</th><th>real</th></tr>"
        + "".join(
            f"<tr><th>{html.escape(str(a))}</th><td>{html.escape(str(b))}</td><td>{html.escape(str(c))}</td></tr>"
            for a, b, c in rows
        )
        + "</table>"
    )
    parts.append(
        "<h2>Suggestions</h2><ul>"
        + "".join(f"<li>{html.escape(s)}</li>" for s in suggestions(syn, real))
        + "</ul>"
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, ch, col in zip(axes, "rgb", ("tab:red", "tab:green", "tab:blue"), strict=True):
        x = np.arange(64) * 4
        hs, hr = np.asarray(syn["histogram"][ch], float), np.asarray(real["histogram"][ch], float)
        ax.plot(x, hs / max(hs.sum(), 1), color=col, label="synthetic")
        ax.plot(x, hr / max(hr.sum(), 1), color=col, ls="--", label="real")
        ax.set_title(f"histogram {ch}")
        ax.legend()
    parts.append(f'<img src="data:image/png;base64,{_png(fig)}"/>')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, ch in zip(axes, "rgb", strict=True):
        for curves, ls, lab in ((syn["noise"], "-", "synthetic"), (real["noise"], "--", "real")):
            for i, c in enumerate(curves):
                ax.plot(
                    c[ch]["mean"],
                    c[ch]["std"],
                    ls,
                    color="tab:orange" if lab == "synthetic" else "tab:blue",
                    alpha=0.5,
                    label=lab if i == 0 else None,
                )
        ax.set(title=f"noise {ch}", xlabel="intensity", ylabel="std")
        ax.legend()
    parts.append(f'<img src="data:image/png;base64,{_png(fig)}"/>')
    parts.append("<h2>Tiles (left synthetic, right real) and 2x centre crops</h2><div>")
    for k in range(min(6, len(syn_items), max(len(real_images), 1))):
        s_img = syn_items[k][0]
        r_img = real_images[k % len(real_images)] if real_images else None
        parts.append(
            "<p>"
            + f'<img src="data:image/jpeg;base64,{_thumb(s_img)}"/>'
            + (f' <img src="data:image/jpeg;base64,{_thumb(r_img)}"/>' if r_img is not None else "")
            + "<br/>"
            + f'<img src="data:image/png;base64,{_crop(s_img)}"/>'
            + (f' <img src="data:image/png;base64,{_crop(r_img)}"/>' if r_img is not None else "")
            + "</p>"
        )
    parts.append("</div>")
    style = "body{font-family:sans-serif;max-width:1300px;margin:auto}td,th{border:1px solid #ccc;padding:4px 8px}img{max-width:100%}"
    out_path.write_text(
        f"<!doctype html><meta charset='utf-8'><title>Realism report</title><style>{style}</style>{''.join(parts)}",
        encoding="utf-8",
    )
    return {
        "synthetic": {k: v for k, v in syn.items() if k not in ("histogram", "noise")},
        "real": {k: v for k, v in real.items() if k not in ("histogram", "noise")},
        "suggestions": suggestions(syn, real),
    }
