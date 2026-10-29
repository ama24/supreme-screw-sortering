"""Lock exposure, gain, white balance and focus of the Pi Camera 3 for all future captures.

Run on the Raspberry Pi with the rig frozen: lights on, shroud closed, Y homed, empty TPU plate.
  screwrig lock --out camera_settings.json
Steps: full-resolution still mode -> let AE/AWB settle -> freeze exposure/gain/colour gains ->
manual focus sweep (variance of Laplacian on the centre) -> save camera_settings.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from .camera import FULL_RES, best_focus, focus_metric

log = logging.getLogger("screwrig.lock")


def _gray(picam2) -> np.ndarray:
    rgb = picam2.capture_array("main")
    return rgb.mean(axis=2) if rgb.ndim == 3 else rgb


def sweep(picam2, positions: list[float], settle_s: float) -> tuple[list[float], list[float]]:
    scores = []
    for lp in positions:
        picam2.set_controls({"LensPosition": float(lp)})
        time.sleep(settle_s)
        picam2.capture_array("main")  # discard one frame after the lens move
        scores.append(focus_metric(_gray(picam2)))
        log.info("lens %.3f dioptre: focus %.1f", lp, scores[-1])
    return positions, scores


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", default="camera_settings.json")
    ap.add_argument("--settle", type=float, default=3.0, help="seconds for AE/AWB to settle")
    ap.add_argument(
        "--distance-mm", type=float, default=300.0, help="approx. camera-to-plate distance"
    )
    ap.add_argument("--exposure-us", type=int, help="force this exposure instead of the AE value")


def run(args: argparse.Namespace) -> int:

    from libcamera import controls
    from picamera2 import Picamera2

    picam2 = Picamera2()
    mode = max(picam2.sensor_modes, key=lambda m: m["size"][0] * m["size"][1])
    config = picam2.create_still_configuration(
        main={"size": FULL_RES, "format": "RGB888"},
        sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]},
        buffer_count=2,
    )
    picam2.configure(config)
    picam2.set_controls(
        {
            "AeEnable": True,
            "AwbEnable": True,
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": 1000.0 / args.distance_mm,
        }
    )
    picam2.start()
    time.sleep(args.settle)
    md = picam2.capture_metadata()
    exposure = int(args.exposure_us or md["ExposureTime"])
    locked = {
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": exposure,
        "AnalogueGain": float(md["AnalogueGain"]),
        "ColourGains": list(md["ColourGains"]),
    }
    picam2.set_controls(locked)
    time.sleep(1.0)

    centre = 1000.0 / args.distance_mm  # dioptres
    coarse = list(np.round(np.arange(centre - 1.0, centre + 1.0001, 0.1), 3))
    pos, sc = sweep(picam2, coarse, 0.3)
    peak = best_focus(pos, sc)
    fine = list(np.round(np.arange(peak - 0.15, peak + 0.1501, 0.02), 3))
    pos, sc = sweep(picam2, fine, 0.3)
    lens = best_focus(pos, sc)
    picam2.set_controls({"LensPosition": lens})
    time.sleep(0.5)
    md = picam2.capture_metadata()

    settings = {
        "ExposureTime": exposure,
        "AnalogueGain": locked["AnalogueGain"],
        "ColourGains": locked["ColourGains"],
        "LensPosition": float(lens),
        "AfMode": "Manual",
        "size": list(FULL_RES),
        "sensor_mode": {"size": list(mode["size"]), "bit_depth": mode["bit_depth"]},
        "format": "RGB888 (BGR byte order, saved as PNG)",
        "focus_sweep": {"positions": [float(p) for p in pos], "scores": sc},
        "metadata_after_lock": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in md.items()
            if isinstance(v, (int, float, str, tuple, list))
        },
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    picam2.stop()
    log.info(
        "locked: exposure %d us, gain %.2f, colour gains %s, lens %.3f dioptre -> %s",
        exposure,
        settings["AnalogueGain"],
        settings["ColourGains"],
        lens,
        args.out,
    )
    return 0
