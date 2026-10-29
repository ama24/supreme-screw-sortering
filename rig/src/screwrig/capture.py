"""Calibration captures with exactly the locked camera settings (refuses to run if they cannot be applied).

  screwrig calib-capture --out captures/intrinsics --mode intrinsics --board board.json
  screwrig calib-capture --out captures/extrinsic  --mode extrinsic
  screwrig calib-capture --out captures/sensor     --mode sensor --frames 20
  screwrig calib-capture --out captures/misc       --mode single --name img_0001

Images are lossless PNG (BGR) with a JSON sidecar of the real frame metadata; --dng also saves raw.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from .camera import Camera, Coverage, focus_metric, save_capture

log = logging.getLogger("screwrig.capture")


def _save(cam: Camera, out: Path, name: str, dng: bool, extra: dict | None = None) -> Path:
    img, md = cam.grab(out / f"{name}.dng" if dng else None)
    return save_capture(out, name, img, md, {"settings": cam.settings, **(extra or {})})


def mode_intrinsics(cam: Camera, out: Path, board_json: str | None, dng: bool) -> None:
    import cv2

    detector, board = None, None
    if board_json:
        spec = json.loads(Path(board_json).read_text())
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
        board = cv2.aruco.CharucoBoard(
            (spec["squares_x"], spec["squares_y"]), spec["square_mm"], spec["marker_mm"], d
        )
        detector = cv2.aruco.CharucoDetector(board)
    cov = Coverage()
    sharp_ref: list[float] = []
    i = len(list(out.glob("view_*.png")))
    print(
        "Intrinsic views: move/tilt the board (max ~10 deg) to cover the WHOLE frame, corners too."
    )
    print("Press Enter to capture, q + Enter to finish. Aim for 25-40 views.")
    while input("> ").strip().lower() != "q":
        img, md = cam.grab()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = focus_metric(gray, 0.9)
        n = 0
        if detector is not None:
            corners, ids, _, _ = detector.detectBoard(gray)
            n = 0 if ids is None else len(ids)
            if n:
                cov.add(corners.reshape(-1, 2))
        if sharp_ref and sharp < 0.5 * float(np.median(sharp_ref)):
            print(
                f"WARNING: blurry (sharpness {sharp:.0f} vs median {np.median(sharp_ref):.0f}); keep still"
            )
        sharp_ref.append(sharp)
        name = f"view_{i:03d}"
        save_capture(
            out, name, img, md, {"settings": cam.settings, "corners": n, "sharpness": sharp}
        )
        i += 1
        print(f"saved {name}: {n} corners, coverage {cov.fraction:.0%}\n{cov.ascii()}")
        cv2.imwrite(str(out / "coverage.png"), cov.image())


def mode_sensor(cam: Camera, out: Path, frames: int, dng: bool) -> None:
    input(
        f"Static scene (e.g. a few parts on the plate, nothing moving). Enter to capture {frames} frames: "
    )
    for k in range(frames):
        _save(cam, out, f"frame_{k:03d}", dng)
    input("Empty plate (remove everything, Y homed). Enter: ")
    _save(cam, out, "empty", dng)
    input("White sheet (matte white paper covering the whole plate). Enter: ")
    _save(cam, out, "white", dng)


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--settings", default="camera_settings.json")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode", choices=["intrinsics", "extrinsic", "sensor", "single"], required=True
    )
    ap.add_argument(
        "--board", help="board.json (from screwsynth-calib board) for live corner coverage"
    )
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--name", default=None, help="file name for --mode single")
    ap.add_argument("--note", default="", help="operator note stored in the sidecar")
    ap.add_argument("--dng", action="store_true", help="also save the raw DNG")


def run(args: argparse.Namespace) -> int:
    settings = json.loads(Path(args.settings).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = Camera(settings)
    try:
        if args.mode == "intrinsics":
            mode_intrinsics(cam, out, args.board, args.dng)
        elif args.mode == "extrinsic":
            input("Board flat on the TPU plate, Y homed. Enter to capture: ")
            print(_save(cam, out, "board_flat", args.dng))
        elif args.mode == "sensor":
            mode_sensor(cam, out, args.frames, args.dng)
        else:
            name = args.name or time.strftime("img_%Y%m%d_%H%M%S")
            print(_save(cam, out, name, args.dng, {"note": args.note}))
    finally:
        cam.close()
    return 0
