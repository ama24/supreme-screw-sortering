#!/usr/bin/env python3
"""Capture with exactly the locked camera settings (refuses to run if they cannot be applied).

  python3 capture.py --settings camera_settings.json --out captures/intrinsics --mode intrinsics --board board.json
  python3 capture.py --settings camera_settings.json --out captures/extrinsic  --mode extrinsic
  python3 capture.py --settings camera_settings.json --out captures/sensor     --mode sensor --frames 20
  python3 capture.py --settings camera_settings.json --out captures/session1   --mode single --name img_0001

Images are lossless PNG (BGR) with a JSON sidecar of the real frame metadata; --dng also saves raw.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from pi_common import (
    FULL_RES,
    Coverage,
    controls_from_settings,
    focus_metric,
    save_capture,
    settings_mismatch,
)

log = logging.getLogger("capture")


class Camera:
    def __init__(self, settings: dict) -> None:
        from picamera2 import Picamera2

        self.settings = settings
        self.picam2 = Picamera2()
        mode = settings.get("sensor_mode") or {}
        cfg = self.picam2.create_still_configuration(
            main={"size": tuple(settings.get("size", FULL_RES)), "format": "RGB888"},
            raw={} if mode else None,
            sensor={"output_size": tuple(mode["size"]), "bit_depth": mode["bit_depth"]}
            if mode
            else {},
            buffer_count=2,
        )
        self.picam2.configure(cfg)
        self.picam2.set_controls(controls_from_settings(settings))
        self.picam2.start()
        time.sleep(1.5)
        self.verify()

    def verify(self) -> None:
        md = dict(self.picam2.capture_metadata())
        md["size"] = list(self.settings.get("size", FULL_RES))
        problems = settings_mismatch(self.settings, md)
        if problems:
            self.picam2.stop()
            raise SystemExit(
                "REFUSING TO CAPTURE: camera does not run with the locked settings:\n  "
                + "\n  ".join(problems)
            )

    def grab(self, dng_path: Path | None = None) -> tuple[np.ndarray, dict]:
        request = self.picam2.capture_request()
        try:
            img = request.make_array("main")
            md = request.get_metadata()
            if dng_path is not None:
                request.save_dng(str(dng_path))
        finally:
            request.release()
        problems = settings_mismatch(self.settings, md)
        if problems:
            raise SystemExit("settings drifted during capture: " + "; ".join(problems))
        return img, md

    def close(self) -> None:
        self.picam2.stop()


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
        "Intrinsic views: move/tilt the board (max ~15 deg) to cover the WHOLE frame, corners too."
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
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


if __name__ == "__main__":
    sys.exit(main())
