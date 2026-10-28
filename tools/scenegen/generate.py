# BlenderProc requires its import to be the first non-comment line, before any docstring or other import.
# isort: skip_file
import blenderproc as bproc  # noqa: F401

import argparse
import sys
from pathlib import Path

USAGE = """Render synthetic scenes of fasteners on the TPU plate.

  blenderproc run tools/scenegen/generate.py --run NAME --scenes N [--seed S] [--start K] [--resume]
                                             [--nominal-camera] [--preview] [--config PATH]
  blenderproc debug tools/scenegen/generate.py --run debug --scenes 1 --nominal-camera   (Blender UI)
"""

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0, str(ROOT / "src")
)  # pure-Python repo modules only (stdlib + numpy inside Blender)


def ensure_packages() -> None:
    """PyYAML is needed to read the configs; install it into Blender's Python on first use."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        from blenderproc.python.utility.SetupUtility import SetupUtility

        SetupUtility.setup_pip(["pyyaml"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True, help="output/<run>")
    ap.add_argument("--scenes", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="index of the first scene (batch chunks)")
    ap.add_argument("--seed", type=int, default=None, help="default: config seed")
    ap.add_argument("--resume", action="store_true", help="skip scenes whose meta.json exists")
    ap.add_argument(
        "--nominal-camera", action="store_true", help="nominal Pi Camera 3, no calibration needed"
    )
    ap.add_argument("--preview", action="store_true", help="10 low-res, low-sample scenes")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    ensure_packages()
    from screwsynth.common.config import load_config
    from screwsynth.scenegen.runner import RunArgs, run

    cfg = load_config(args.config)
    scenes = args.scenes
    if args.preview:
        scenes = int((cfg.get("scenegen.preview", {}) or {}).get("scenes", 10))
    stats = run(
        RunArgs(
            args.run,
            scenes,
            cfg.get("seed", 0) if args.seed is None else args.seed,
            args.start,
            args.resume,
            args.nominal_camera,
            args.preview,
        ),
        cfg,
    )
    return 1 if stats["failed"] and not stats["done"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))  # blenderproc forwards everything after the script path
