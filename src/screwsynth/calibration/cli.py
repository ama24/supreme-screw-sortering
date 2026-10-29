"""screwsynth-calib: ChArUco board, intrinsics, extrinsics, sensor profile, validation, recheck."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from screwsynth.common.config import load_config
from screwsynth.common.io import read_json, write_json
from screwsynth.common.log import setup_logging
from screwsynth.common.paths import resolve

from . import board as board_mod
from . import extrinsic, intrinsics, measure, report, sensor, synthetic
from .board import BoardSpec
from .geometry import Camera

log = logging.getLogger("screwsynth.calibration")
CALIB = "calibration.json"
PROFILE = "sensor_profile.json"
IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.dng")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


class _Ctx:
    def __init__(self, args: argparse.Namespace) -> None:
        self.cfg = load_config(args.config)
        self.section = self.cfg.get("calibration", {}) or {}
        self.dir = resolve(
            args.calib_dir or self.cfg.get("paths.calibration_dir", "data/calibration")
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        board_json = self.dir / "board.json"
        base = (
            read_json(board_json) if board_json.is_file() else dict(self.section.get("board", {}))
        )
        self.spec = BoardSpec.from_mapping(base).with_square(getattr(args, "square_mm", None))

    def calib(self) -> dict[str, Any]:
        p = self.dir / CALIB
        return read_json(p) if p.is_file() else {}

    def save_calib(self, data: dict[str, Any]) -> None:
        data["date"] = dt.datetime.now().isoformat(timespec="seconds")
        data["board"] = self.spec.to_dict()
        write_json(self.dir / CALIB, data)


def _images(src: str) -> list[str]:
    p = Path(src)
    if p.is_dir():
        files = sorted(str(f) for g in IMAGE_GLOBS for f in p.glob(g))
    else:
        files = sorted(glob.glob(src))  # noqa: PTH207 - user glob may be absolute
    if not files:
        raise SystemExit(f"no images found in {src}")
    return files


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return img


def cmd_board(ctx: _Ctx, args: argparse.Namespace) -> int:
    spec = BoardSpec.from_mapping(ctx.section.get("board", {}))
    out = resolve(args.out) if args.out else ctx.dir / "charuco_board.pdf"
    board_mod.write_board_pdf(spec, out, args.dpi)
    write_json(ctx.dir / "board.json", spec.to_dict())
    _out(f"wrote {out} and {ctx.dir / 'board.json'}")
    ids = list(ctx.section.get("corner_marker_ids", []) or [])
    if args.markers and ids:
        files = board_mod.corner_marker_svgs(
            ids,
            ctx.section.get("corner_marker_dictionary", "DICT_4X4_50"),
            float(ctx.section.get("corner_marker_mm", 20)),
            ctx.dir,
        )
        _out("corner markers: " + ", ".join(str(f) for f in files))
    _out("Print at 100% (actual size), glue flat, measure the squares, then pass --square-mm.")
    return 0


def cmd_intrinsics(ctx: _Ctx, args: argparse.Namespace) -> int:
    files = _images(args.images)
    res = intrinsics.calibrate(
        files,
        ctx.spec,
        model=args.model,
        min_corners=int(ctx.section.get("min_corners", 12)),
        blur_ratio=float(ctx.section.get("blur_ratio", 0.4)),
    )
    calib = ctx.calib()
    calib.update(res.to_json())
    calib.pop("extrinsic", None)  # intrinsics changed: the extrinsic must be redone
    ctx.save_calib(calib)
    np.save(ctx.dir / "residuals.npy", res.residuals)
    target = float(ctx.section.get("target_rms_px", 0.5))
    used = sum(v.used for v in res.views)
    _out(
        f"views used {used}/{len(res.views)}, model {res.model}, RMS {res.rms:.3f} px "
        f"(holdout {res.holdout_rms[res.model]:.3f})"
    )
    for v in res.views:
        if not v.used:
            _out(f"  rejected {v.file}: {v.reason}")
    worst = sorted((v for v in res.views if v.used), key=lambda v: -(v.rms or 0))[:3]
    _out("  worst views: " + ", ".join(f"{v.file} {v.rms:.3f}" for v in worst))
    if res.rms > target:
        log.warning(
            "RMS %.3f px is above the %.2f px target: retake blurry/poor views", res.rms, target
        )
    return 0


def cmd_extrinsic(ctx: _Ctx, args: argparse.Namespace) -> int:
    calib = ctx.calib()
    if "camera_matrix" not in calib:
        raise SystemExit("run `screwsynth-calib intrinsics` first")
    cam = Camera.from_calibration(calib)
    area = dict(ctx.section.get("working_area", {}) or {})
    if args.area_origin:
        area["origin_mm"] = args.area_origin
    if args.area_size:
        area["size_mm"] = args.area_size
    thickness = (
        args.board_thickness_mm
        if args.board_thickness_mm is not None
        else float(ctx.section.get("board_thickness_mm", 0.0))
    )
    ext = extrinsic.solve(
        _read(args.image),
        ctx.spec,
        cam,
        thickness,
        area,
        corner_ids=list(ctx.section.get("corner_marker_ids", []) or []),
        corner_dictionary=ctx.section.get("corner_marker_dictionary", "DICT_4X4_50"),
        roi_multiple=int(ctx.section.get("roi_multiple", 32)),
    )
    ext["image"] = Path(args.image).name
    calib["extrinsic"] = ext
    calib["roi"] = ext["roi"]
    calib["px_per_mm"] = ext["px_per_mm"]
    calib["plate_homography"] = ext["plate_homography"]
    ctx.save_calib(calib)
    _out(
        f"camera height {ext['camera_height_mm']:.2f} mm, {ext['px_per_mm']:.3f} px/mm, "
        f"ROI {ext['roi']}, RMS {ext['reprojection_rms_px']:.3f} px, area from {ext['working_area_mm']['source']}"
    )
    return 0


def cmd_sensor(ctx: _Ctx, args: argparse.Namespace) -> int:
    frames = [_read(f) for f in _images(args.frames)]
    calib = ctx.calib()
    cam = Camera.from_calibration(calib) if "camera_matrix" in calib else None
    roi = calib.get("roi")
    black = None
    sidecar = Path(
        args.frames if Path(args.frames).is_file() else _images(args.frames)[0]
    ).with_suffix(".json")
    if sidecar.is_file():
        levels = read_json(sidecar).get("metadata", {}).get("SensorBlackLevels")
        if levels:
            black = float(np.mean(levels)) / 256.0  # 16-bit sensor units -> 8-bit
    profile, _, _ = sensor.build_profile(
        frames, _read(args.white), _read(args.empty), cam, roi, black, ctx.dir
    )
    write_json(ctx.dir / PROFILE, profile)
    n = profile["noise"]
    _out(
        "noise var = a*I + b: " + ", ".join(f"{c}: a={n[c]['a']:.3f} b={n[c]['b']:.2f}" for c in n)
    )
    _out(f"wrote {ctx.dir / PROFILE}, flatfield.png, plate_empty.png")
    return 0


def cmd_validate(ctx: _Ctx, args: argparse.Namespace) -> int:
    calib = ctx.calib()
    cam = Camera.from_calibration(calib)
    res = measure.validate(_read(args.image), cam, args.lengths, args.height_mm, args.order)
    for o in res["objects"]:
        _out(
            f"  measured {o['length_mm']:8.3f}  true {o['true_mm']:8.3f}  error {o['error_mm']:+.3f} mm"
            f"  at ({o['centre_mm'][0]:.1f}, {o['centre_mm'][1]:.1f})"
        )
    target = float(ctx.section.get("validate_target_mm", 0.2))
    _out(
        f"mean |error| {res['mean_abs_error_mm']:.3f} mm, max {res['max_abs_error_mm']:.3f} mm "
        f"(target < {target} mm)"
    )
    calib["validation"] = res
    ctx.save_calib(calib)
    return 0 if res["max_abs_error_mm"] <= target else 1


def cmd_recheck(ctx: _Ctx, args: argparse.Namespace) -> int:
    r = measure.recheck(
        _read(args.image),
        ctx.spec,
        ctx.calib(),
        float(ctx.section.get("recheck_max_tilt_deg", 0.15)),
        float(ctx.section.get("recheck_max_height_mm", 0.3)),
    )
    _out(
        f"plane tilt change {r['plane_tilt_change_deg']:.3f} deg, camera height change "
        f"{r['camera_height_change_mm']:+.3f} mm"
    )
    _out(
        "RECALIBRATE the extrinsic (screwsynth-calib extrinsic)"
        if r["recalibrate_extrinsic"]
        else "OK: pose unchanged"
    )
    return 1 if r["recalibrate_extrinsic"] else 0


def cmd_report(ctx: _Ctx, args: argparse.Namespace) -> int:
    calib = ctx.calib()
    profile = read_json(ctx.dir / PROFILE) if (ctx.dir / PROFILE).is_file() else None
    res = np.load(ctx.dir / "residuals.npy") if (ctx.dir / "residuals.npy").is_file() else None
    ext_img = None
    ext = calib.get("extrinsic")
    if ext and args.extrinsic_image:
        ext_img = _read(args.extrinsic_image)
    path = report.write_report(ctx.dir, calib, profile, res, ext_img)
    _out(f"wrote {path}")
    return 0


def cmd_make_sample(ctx: _Ctx, args: argparse.Namespace) -> int:
    truth = synthetic.make_sample(resolve(args.out), args.seed, ctx.spec)  # the configured board
    _out(
        f"wrote synthetic sample set to {resolve(args.out)} (truth.json: K {truth['camera_matrix'][0][0]:.0f})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config")
    common.add_argument("--calib-dir", help="default paths.calibration_dir")
    common.add_argument("-v", "--verbose", action="store_true")
    sq = argparse.ArgumentParser(add_help=False)
    sq.add_argument("--square-mm", type=float, help="MEASURED square size of the printed board")

    p = argparse.ArgumentParser(prog="screwsynth-calib", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("board", parents=[common], help="printable ChArUco PDF (+ corner markers)")
    sp.add_argument("--out")
    sp.add_argument("--dpi", type=int, default=600)
    sp.add_argument("--markers", action="store_true", help="also write plate-corner ArUco SVGs")
    sp.set_defaults(func=cmd_board)

    sp = sub.add_parser("intrinsics", parents=[common, sq], help="calibrate K and distortion")
    sp.add_argument("--images", required=True, help="folder or glob of intrinsic views")
    sp.add_argument("--model", choices=["auto", "standard5", "rational8"], default="auto")
    sp.set_defaults(func=cmd_intrinsics)

    sp = sub.add_parser(
        "extrinsic", parents=[common, sq], help="plate frame from a flat board image"
    )
    sp.add_argument("--image", required=True)
    sp.add_argument("--board-thickness-mm", type=float, help="paper + glass; default from config")
    sp.add_argument(
        "--area-origin",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="working-area corner in the plate frame (mm, relative to the board origin)",
    )
    sp.add_argument("--area-size", type=float, nargs=2, metavar=("W", "H"))
    sp.set_defaults(func=cmd_extrinsic)

    sp = sub.add_parser("sensor", parents=[common], help="noise model, flat-field, empty plate")
    sp.add_argument(
        "--frames", required=True, help="folder/glob of repeated frames of a static scene"
    )
    sp.add_argument("--white", required=True)
    sp.add_argument("--empty", required=True)
    sp.set_defaults(func=cmd_sensor)

    sp = sub.add_parser("validate", parents=[common], help="measure objects of known length")
    sp.add_argument("--image", required=True)
    sp.add_argument("--lengths", type=float, nargs="+", required=True, help="caliper lengths (mm)")
    sp.add_argument("--height-mm", type=float, default=0.0, help="height of the measured outline")
    sp.add_argument(
        "--order",
        choices=["size", "x"],
        default="size",
        help="match by size (default) or left-to-right in the plate frame",
    )
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("recheck", parents=[common], help="has the camera/plate pose drifted?")
    sp.add_argument("--image", required=True, help="flat-board image")
    sp.set_defaults(func=cmd_recheck)

    sp = sub.add_parser("report", parents=[common], help="report.html with plots")
    sp.add_argument("--extrinsic-image", help="draw the plate axes and ROI on this image")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser(
        "make-sample", parents=[common], help="synthetic capture set with ground truth"
    )
    sp.add_argument("out")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_make_sample)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    try:
        return int(args.func(_Ctx(args), args))
    except ValueError as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
