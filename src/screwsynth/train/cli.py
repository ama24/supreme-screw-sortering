"""screwsynth-train / screwsynth-infer / screwsynth-eval: stage 1 and 2 training, measurement,
inference and evaluation (tool 08)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from screwsynth.common.config import load_config
from screwsynth.common.io import read_json, write_json, write_text
from screwsynth.common.log import setup_logging
from screwsynth.common.paths import resolve

log = logging.getLogger("screwsynth.train")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _runs_dir(cfg) -> Path:
    return resolve(cfg.get("train.runs_dir", "runs"))


def _calibration(cfg, args: argparse.Namespace) -> dict | None:
    if getattr(args, "nominal_camera", False):
        return None
    path = resolve(
        getattr(args, "calibration", None)
        or Path(cfg.get("paths.calibration_dir", "data/calibration")) / "calibration.json"
    )
    if not path.is_file():
        log.warning("%s not found: using the nominal camera (calibrate with tool 03)", path)
        return None
    return read_json(path)


# --------------------------------------------------------------------------- screwsynth-train


def cmd_run(args: argparse.Namespace) -> int:
    from .experiments import Experiment
    from .trainer import train

    cfg = load_config(args.config)
    exp = Experiment.load(args.experiment, args.dataset)
    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.dry_run:
        _out(json.dumps(exp.ultralytics_args(_runs_dir(cfg) / exp.stage), indent=2))
        return 0
    run_dir = train(exp, _runs_dir(cfg) / exp.stage, args.device, **overrides)
    _out(f"weights: {run_dir / 'weights' / 'best.pt'}")
    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    from .experiments import Experiment
    from .trainer import finetune

    cfg = load_config(args.config)
    exp = Experiment.load(args.experiment, args.dataset)
    res = finetune(
        resolve(args.weights),
        exp,
        resolve(args.data),
        resolve(args.holdout),
        _runs_dir(cfg) / exp.stage,
        args.freeze,
        args.lr0,
        args.epochs,
        args.device,
    )
    _out(json.dumps({"before": res["before"], "after": res["after"]}, indent=2))
    return 0


def cmd_curve(args: argparse.Namespace) -> int:
    from .experiments import Experiment
    from .trainer import learning_curve

    cfg = load_config(args.config)
    exp = Experiment.load(args.experiment, args.dataset)
    fractions = [float(f) / 100 for f in args.fractions.split(",")]
    res = learning_curve(exp, _runs_dir(cfg) / exp.stage, fractions, args.epochs, args.device)
    _out(json.dumps(res["points"], indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .trainer import export_onnx

    res = export_onnx(resolve(args.weights), args.imgsz, args.runs)
    _out(json.dumps(res, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .evaluate import compare_measurement, comparison_markdown
    from .geometry import Geometry
    from .infer import Settings

    cfg = load_config(args.config)
    ds = resolve(args.dataset)
    res = compare_measurement(
        ds,
        Geometry.from_config(cfg),
        args.split,
        resolve(args.stage2) if args.stage2 else None,
        Settings.from_config(cfg.to_dict()),
    )
    out = resolve(args.out) if args.out else ds / "measurement_comparison"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "comparison.json", res)
    md = comparison_markdown(res)
    write_text(out / "comparison.md", md)
    _out(md)
    return 0


def build_train_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config")
    common.add_argument("-v", "--verbose", action="store_true")
    p = argparse.ArgumentParser(prog="screwsynth-train", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    exp_help = "configs/train/<name>.yaml or a path"

    sp = sub.add_parser(
        "run", parents=[common], help="train an experiment (stage 1 seg or stage 2 pose)"
    )
    sp.add_argument("experiment", help=exp_help)
    sp.add_argument("--dataset", help="tool-07 dataset dir (overrides the experiment)")
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--device", help="e.g. 0 or cpu")
    sp.add_argument("--dry-run", action="store_true", help="print the Ultralytics arguments only")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser(
        "finetune", parents=[common], help="fine-tune on real data, compare before/after"
    )
    sp.add_argument("experiment", help=exp_help)
    sp.add_argument("--weights", required=True, help="best synthetic model (best.pt)")
    sp.add_argument("--data", required=True, help="real fine-tuning dataset yaml (tool 09)")
    sp.add_argument(
        "--holdout", required=True, help="held-out real dataset yaml (never trained on)"
    )
    sp.add_argument("--dataset")
    sp.add_argument("--freeze", type=int, default=10, help="frozen early layers")
    sp.add_argument("--lr0", type=float, default=0.0005)
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--device")
    sp.set_defaults(func=cmd_finetune)

    sp = sub.add_parser("curve", parents=[common], help="learning curve: mAP vs dataset size")
    sp.add_argument("experiment", help=exp_help)
    sp.add_argument("--dataset")
    sp.add_argument("--fractions", default="10,25,50,100", help="percent of the training layouts")
    sp.add_argument("--epochs", type=int, default=30)
    sp.add_argument("--device")
    sp.set_defaults(func=cmd_curve)

    sp = sub.add_parser("export", parents=[common], help="ONNX export + CPU latency")
    sp.add_argument("--weights", required=True)
    sp.add_argument("--imgsz", type=int, default=1024)
    sp.add_argument("--runs", type=int, default=10)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser(
        "compare-measure", parents=[common], help="method A (keypoints) vs B (classical)"
    )
    sp.add_argument(
        "--dataset", required=True, help="tool-07 dataset exported with the crops format"
    )
    sp.add_argument(
        "--stage2", help="keypoint model weights (otherwise only A with the true keypoints)"
    )
    sp.add_argument("--split", default="val", help="train | val | all")
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_compare)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_train_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return int(args.func(args))


# --------------------------------------------------------------------------- screwsynth-infer


def infer_main(argv: Sequence[str] | None = None) -> int:
    import cv2

    from .geometry import Geometry
    from .infer import Pipeline, Settings, draw

    p = argparse.ArgumentParser(
        prog="screwsynth-infer", description="photo -> parts, sizes, lengths (JSON)"
    )
    p.add_argument("--image", required=True, nargs="+", help="full frame or ROI image(s)")
    p.add_argument("--stage1", help="stage-1 weights (default train.stage1_weights)")
    p.add_argument(
        "--stage2",
        help="stage-2 keypoint weights (default train.stage2_weights; 'none' = classical only)",
    )
    p.add_argument("--calibration", help="calibration.json (default paths.calibration_dir)")
    p.add_argument("--nominal-camera", action="store_true")
    p.add_argument("--method", choices=["auto", "keypoints", "classical"])
    p.add_argument("--out", help="output folder (default next to the image)")
    p.add_argument("--save-vis", action="store_true", help="also write <image>_vis.jpg")
    p.add_argument("--config")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    cfg = load_config(args.config)
    st = Settings.from_config(cfg.to_dict())
    if args.method:
        st.method = args.method
    stage1 = args.stage1 or cfg.get("train.stage1_weights")
    if not stage1:
        raise SystemExit("no stage-1 weights: pass --stage1 or set train.stage1_weights")
    stage2 = args.stage2 or cfg.get("train.stage2_weights")
    stage2 = None if stage2 in (None, "none") else resolve(stage2)
    pipe = Pipeline(resolve(stage1), stage2, Geometry.from_config(cfg), st, _calibration(cfg, args))
    for img in args.image:
        path = resolve(img)
        result, scene, _ = pipe.run(path)
        out_dir = resolve(args.out) if args.out else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / f"{path.stem}_result.json", result)
        if args.save_vis:
            cv2.imwrite(str(out_dir / f"{path.stem}_vis.jpg"), draw(scene, result))
        n = len(result["objects"])
        measured = [o for o in result["objects"] if o.get("length")]
        _out(
            f"{path.name}: {n} object(s), {len(measured)} measured, {result['timing_s']['total']:.2f} s -> {out_dir}"
        )
    return 0


# ---------------------------------------------------------------------------- screwsynth-eval


def eval_main(argv: Sequence[str] | None = None) -> int:
    from .evaluate import detection_metrics, length_accuracy, write_report
    from .geometry import Geometry
    from .infer import Pipeline, Settings

    p = argparse.ArgumentParser(
        prog="screwsynth-eval", description="mAP, size confusion, length accuracy"
    )
    p.add_argument(
        "--dataset", required=True, help="tool-07 dataset (synthetic) or tool-09 real set"
    )
    p.add_argument("--stage1", required=True)
    p.add_argument("--stage2", help="keypoint weights; omit for classical measurement only")
    p.add_argument("--split", default="val")
    p.add_argument(
        "--compare-with", help="a second dataset (e.g. synthetic val) for the sim-to-real gap"
    )
    p.add_argument("--out", help="report folder (default <dataset>/eval_<weights name>)")
    p.add_argument("--config")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    cfg = load_config(args.config)
    st = Settings.from_config(cfg.to_dict())
    geo = Geometry.from_config(cfg)
    pipe = Pipeline(
        resolve(args.stage1), resolve(args.stage2) if args.stage2 else None, geo, st, None
    )

    def evaluate(ds: Path) -> dict:
        out = {
            "name": ds.name,
            "detection": detection_metrics(resolve(args.stage1), ds, args.split, st.stage1_imgsz),
        }
        if (ds / "crops.csv").is_file():
            out["length"] = length_accuracy(pipe, ds, args.split, out_dir / ds.name)
        return out

    ds = resolve(args.dataset)
    out_dir = resolve(args.out) if args.out else ds / f"eval_{Path(args.stage1).stem}"
    res = evaluate(ds)
    other = evaluate(resolve(args.compare_with)) if args.compare_with else None
    write_report(out_dir, ds.name, res, other)
    _out((out_dir / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
