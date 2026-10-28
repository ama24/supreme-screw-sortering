"""screwsynth-post: make renders look like real Pi Camera 3 captures (labels stay consistent)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import numpy as np

from screwsynth.common.config import load_config
from screwsynth.common.io import read_json, write_json
from screwsynth.common.log import setup_logging
from screwsynth.common.paths import resolve

from . import compare
from .process import SensorData, process_scene

log = logging.getLogger("screwsynth.postprocess")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _job(args: tuple) -> tuple[str, str]:
    scene_dir, out_dir, cfg, calib_dir, seed = args
    try:
        process_scene(Path(scene_dir), Path(out_dir), cfg, SensorData.load(Path(calib_dir)), seed)
        return Path(scene_dir).name, ""
    except Exception as exc:  # report per scene, keep going
        return Path(scene_dir).name, f"{type(exc).__name__}: {exc}"


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    pc = dict(cfg.get("postprocess", {}) or {})
    run_dir = resolve(cfg.get("paths.output_dir", "output")) / args.run
    scenes = sorted(d for d in (run_dir / "scenes").iterdir() if (d / "meta.json").is_file())
    calib_dir = resolve(cfg.get("paths.calibration_dir", "data/calibration"))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    jobs = []
    for d in scenes:
        out = run_dir / "processed" / d.name
        if args.resume and (out / "meta.json").is_file():
            continue
        jobs.append((str(d), str(out), pc, str(calib_dir), seed))
    workers = args.workers or int(pc.get("workers", 0)) or min(4, os.cpu_count() or 1)
    failed = []
    if workers <= 1 or len(jobs) <= 1:
        results = [_job(j) for j in jobs]
    else:
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        results = []
        try:
            with ProcessPoolExecutor(workers) as pool:
                futs = [pool.submit(_job, j) for j in jobs]
                for f in as_completed(futs):
                    results.append(f.result())
        except BrokenProcessPool:
            log.warning("worker processes failed; continuing in this process")
            done = {r[0] for r in results}
            results += [_job(j) for j in jobs if Path(j[0]).name not in done]
    for name, err in results:
        if err:
            failed.append(name)
            log.error("%s: %s", name, err)
    _out(
        f"scenes: {len(scenes)}  processed: {len(results) - len(failed)}  skipped: {len(scenes) - len(jobs)}"
        f"  failed: {len(failed)}  -> {run_dir / 'processed'}"
    )
    return 1 if failed else 0


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    run_dir = resolve(cfg.get("paths.output_dir", "output")) / args.run
    items = []
    for d in sorted((run_dir / "processed").iterdir())[: args.n]:
        meta = read_json(d / "meta.json")
        img = cv2.cvtColor(cv2.imread(str(d / meta["image"])), cv2.COLOR_BGR2RGB)
        items.append((img, cv2.imread(str(d / "instance.png"), cv2.IMREAD_UNCHANGED)))
    real = [
        cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        for p in compare.list_images(resolve(args.real), args.n)
    ]
    if not items or not real:
        raise SystemExit("need processed synthetic scenes and real images")
    calib = resolve(cfg.get("paths.calibration_dir", "data/calibration")) / "calibration.json"
    if args.crop_roi and calib.is_file():
        roi = read_json(calib).get("roi")
        if roi:
            real = [
                r[roi["y0"] : roi["y0"] + roi["h"], roi["x0"] : roi["x0"] + roi["w"]]
                if r.shape[1] > roi["w"]
                else r
                for r in real
            ]
    out = run_dir / "compare.html"
    summary = compare.report(items, real, out)
    write_json(run_dir / "compare.json", summary)
    for s in summary["suggestions"]:
        _out(f"- {s}")
    _out(f"wrote {out}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    run_dir = resolve(cfg.get("paths.output_dir", "output")) / args.run
    raw, proc = run_dir / "scenes" / args.scene, run_dir / "processed" / args.scene
    tiles = []
    for d in (raw, proc):
        meta = read_json(d / "meta.json")
        img = cv2.imread(str(d / meta.get("image", "rgb.png")))
        mask = cv2.imread(str(d / "instance.png"), cv2.IMREAD_UNCHANGED)
        edges = cv2.morphologyEx(
            (mask > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        boundary = cv2.Canny(mask.astype(np.float32).astype(np.uint8), 0, 1) | edges
        img[boundary > 0] = (0, 255, 255)
        for o in meta["objects"]:
            for name, kp in o["keypoints"].items():
                if kp["visibility"]:
                    col = {"head_top": (0, 0, 255), "bearing": (0, 200, 0), "tip": (255, 0, 0)}[
                        name
                    ]
                    cv2.circle(img, tuple(int(round(v)) for v in kp["uv"]), 4, col, 1, cv2.LINE_AA)
        cv2.putText(
            img,
            "render" if d == raw else "processed",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        tiles.append(img)
    out = run_dir / f"show_{args.scene}.png"
    cv2.imwrite(str(out), np.hstack(tiles))
    _out(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config")
    common.add_argument("-v", "--verbose", action="store_true")
    p = argparse.ArgumentParser(prog="screwsynth-post", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("run", parents=[common], help="process output/<run>/scenes -> processed")
    sp.add_argument("--run", required=True)
    sp.add_argument("--workers", type=int)
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--seed", type=int)
    sp.set_defaults(func=cmd_run)
    sp = sub.add_parser("compare", parents=[common], help="synthetic vs real statistics (HTML)")
    sp.add_argument("--run", required=True)
    sp.add_argument("--real", required=True, help="folder of real captures, e.g. data/real/<set>")
    sp.add_argument("--n", type=int, default=20)
    sp.add_argument(
        "--crop-roi", action="store_true", help="crop full-frame real images to the calibrated ROI"
    )
    sp.set_defaults(func=cmd_compare)
    sp = sub.add_parser(
        "show", parents=[common], help="before/after with mask outline and keypoints"
    )
    sp.add_argument("--run", required=True)
    sp.add_argument("--scene", required=True)
    sp.set_defaults(func=cmd_show)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
