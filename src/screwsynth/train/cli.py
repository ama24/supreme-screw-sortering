"""screwsynth-train: stage 1 and 2 training, fine-tuning, learning curve, ONNX export (tool 08).

Generic training is moving to the separate model-trainer app; measurement and inference live in
screwsynth.measure (screwsynth-infer, screwsynth-eval)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from screwsynth.common.config import load_config
from screwsynth.common.log import setup_logging
from screwsynth.common.paths import resolve

log = logging.getLogger("screwsynth.train")


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _runs_dir(cfg) -> Path:
    return resolve(cfg.get("train.runs_dir", "runs"))


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

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_train_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
