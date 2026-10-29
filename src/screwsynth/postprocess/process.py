"""Post-process one rendered scene: realistic image + distorted mask + updated meta.json."""

from __future__ import annotations

import copy
import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screwsynth.common.io import read_json, write_json
from screwsynth.common.png import write_png
from screwsynth.common.seed import derive_seed

from . import effects as fx

log = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Inputs from tool 03 (all optional: the nominal camera has no calibration)."""

    noise: dict[str, dict[str, float]] | None = None
    flatfield: np.ndarray | None = None  # full-sensor float image, max 1
    plate_rgb: list[float] | None = None  # empty-plate mean RGB (8-bit)

    @classmethod
    def load(cls, calib_dir: Path) -> SensorData:
        prof_path = calib_dir / "sensor_profile.json"
        prof = read_json(prof_path) if prof_path.is_file() else {}
        ff_path = calib_dir / "flatfield.png"
        ff = None
        if ff_path.is_file():
            raw = cv2.imread(str(ff_path), cv2.IMREAD_UNCHANGED)
            ff = raw.astype(np.float32) / (65535.0 if raw.dtype == np.uint16 else 255.0)
        return cls(prof.get("noise"), ff, (prof.get("plate_empty") or {}).get("mean_rgb"))


def _on(cfg: Mapping[str, Any], key: str, rng: random.Random) -> bool:
    c = cfg.get(key, {}) or {}
    return bool(c.get("enabled", True)) and rng.random() < float(c.get("probability", 1.0))


def process_scene(
    scene_dir: Path, out_dir: Path, cfg: Mapping[str, Any], sensor: SensorData, seed: int = 0
) -> dict[str, Any]:
    """Read rgb.png/instance.png/meta.json (never modified) and write the processed scene."""
    meta = read_json(scene_dir / "meta.json")
    rgb = cv2.cvtColor(cv2.imread(str(scene_dir / "rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(scene_dir / "instance.png"), cv2.IMREAD_UNCHANGED)
    cam = meta["camera"]
    K = np.asarray(cam["K"], dtype=float)
    dist = np.asarray(cam.get("dist_coeffs", [0] * 5), dtype=float)
    h, w = mask.shape
    rng = random.Random(derive_seed(seed, "post", meta["scene_id"]))
    nrng = np.random.default_rng(derive_seed(seed, "noise", meta["scene_id"]))
    params: dict[str, Any] = {}

    # 1. lens distortion (RGB interpolated, mask nearest, keypoints with the same model)
    dc = cfg.get("distortion", {}) or {}
    if dc.get("enabled", True) and fx.has_distortion(dist):
        img = fx.distort_image(rgb, K, dist, dc.get("rgb_interpolation", "linear")).astype(
            np.float32
        )
        mask_out = fx.distort_mask(mask, K, dist)
        params["distortion"] = {
            "dist_coeffs": dist.tolist(),
            "interpolation": dc.get("rgb_interpolation", "linear"),
        }
        distorted = True
    else:
        img, mask_out, distorted = rgb.astype(np.float32), mask.copy(), False
        params["distortion"] = None

    # 2. vignetting / flat-field
    if _on(cfg, "vignetting", rng):
        vc = cfg.get("vignetting") or {}
        roi = cam.get("roi") if sensor.flatfield is not None else None
        ff = fx.flatfield_for_render(
            sensor.flatfield, roi, (w, h), float(vc.get("fallback_k", 0.25))
        )
        s = fx.uniform(rng, vc.get("strength", [0.8, 1.2]))
        img = fx.apply_vignetting(img, ff, s)
        params["vignetting"] = {
            "strength": s,
            "source": "calibration" if roi else "radial_fallback",
        }

    # 3. optics blur
    if _on(cfg, "blur", rng):
        sigma = fx.uniform(rng, (cfg.get("blur") or {}).get("sigma_px", [0.3, 0.9])) * (
            w / cam.get("roi", {}).get("w", w)
        )
        img = fx.apply_blur(img, sigma)
        params["blur"] = {"sigma_px": sigma}

    # 3b. chromatic aberration (off by default)
    cc = cfg.get("chromatic_aberration", {}) or {}
    if cc.get("enabled", False) and rng.random() < float(cc.get("probability", 0.5)):
        rs = [
            fx.uniform(rng, cc.get("scale", [0.9995, 1.0005])),
            1.0,
            fx.uniform(rng, cc.get("scale", [0.9995, 1.0005])),
        ]
        img = fx.apply_chromatic_aberration(img, rs, (K[0, 2], K[1, 2]))
        params["chromatic_aberration"] = {"scales_rgb": rs}

    # 4. exposure / white balance / gamma, optionally centred on the real empty-plate colour
    if _on(cfg, "colour", rng):
        col = cfg.get("colour", {}) or {}
        base = np.ones(3)
        background = (meta.get("lighting") or {}).get("background") or {}
        on_plate = background.get("kind", "calibrated_photo") == "calibrated_photo"
        # only scenes showing the real empty plate are pulled towards its colour; a craft mat
        # or a solid colour must keep its own
        if col.get("match_plate", True) and on_plate and sensor.plate_rgb is not None:
            bg = img[mask_out == 0]
            if len(bg) > 100:
                med = np.median(bg, axis=0)
                base = np.clip(np.asarray(sensor.plate_rgb) / np.maximum(med, 1.0), 0.5, 2.0)
        gain = fx.uniform(rng, col.get("gain", [0.9, 1.1]))
        wb = [fx.uniform(rng, col.get("wb", [0.96, 1.04])) for _ in range(3)]
        gains = (base * gain * np.asarray(wb)).tolist()
        gamma = fx.uniform(rng, col.get("gamma", [0.95, 1.05]))
        img = fx.apply_colour(img, gains, gamma)
        params["colour"] = {"gains_rgb": gains, "gamma": gamma, "plate_match": base.tolist()}

    # 5. sensor noise
    if _on(cfg, "noise", rng):
        nc = cfg.get("noise", {}) or {}
        model = sensor.noise or nc.get("fallback") or {ch: {"a": 0.5, "b": 2.0} for ch in "rgb"}
        if "a" in model:  # single fallback for all channels
            model = {ch: model for ch in "rgb"}
        scale = fx.uniform(rng, nc.get("scale", [0.8, 1.2]))
        fpn = None
        if float(nc.get("fixed_pattern", 0.0)) > 0:
            fpn = fx.fixed_pattern_map((w, h), float(nc["fixed_pattern"]), derive_seed(seed, "fpn"))
        img = fx.apply_noise(img, model, scale, nrng, fpn)
        params["noise"] = {
            "scale": scale,
            "model": model,
            "source": "sensor_profile" if sensor.noise else "fallback",
        }

    # 6. compression
    comp = cfg.get("compression", {}) or {}
    fmt = str(comp.get("production_format", "png"))
    quality = None
    if comp.get("enabled", True) and rng.random() < float(comp.get("jpeg_probability", 0.2)):
        fmt = "jpg"
        quality = int(round(fx.uniform(rng, comp.get("quality", [80, 97]))))
    params["compression"] = {"format": fmt, "quality": quality}

    out_dir.mkdir(parents=True, exist_ok=True)
    image_name = f"image.{fmt}"
    (out_dir / image_name).write_bytes(fx.encode(img, fmt, quality or 95))
    write_png(out_dir / "instance.png", mask_out.astype(np.uint16))

    new = copy.deepcopy(meta)
    new["image"] = image_name
    new["camera"]["distorted"] = distorted
    new["postprocess"] = params
    new["source_scene"] = str(scene_dir)
    for obj in new["objects"]:
        area = int((mask_out == obj["instance_id"]).sum())
        obj["mask_area_px"] = area
        if obj.get("alone_area_px"):
            obj["visible_pixel_fraction"] = round(min(1.0, area / obj["alone_area_px"]), 4)
        uv = np.array([obj["keypoints"][k]["uv"] for k in obj["keypoints"]])
        duv = fx.distort_points(uv, K, dist) if distorted else uv
        for kp, p in zip(obj["keypoints"].values(), duv, strict=True):
            kp["uv_undistorted"] = kp["uv"]
            kp["uv"] = np.round(p, 4).tolist()
            if not (0 <= p[0] <= w - 1 and 0 <= p[1] <= h - 1):
                kp["visibility"] = 0
    write_json(out_dir / "meta.json", new)
    return params
