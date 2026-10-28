"""Render a large run in chunks, restarting Blender every chunk so memory leaks cannot accumulate.

  python tools/scenegen/batch.py --run NAME --total 3000 --chunk 200 [--seed S] [--nominal-camera]
  python tools/scenegen/batch.py --run bench --benchmark [--nominal-camera]

Each chunk is `blenderproc run tools/scenegen/generate.py --start K --scenes CHUNK --resume`. Scene k
always uses the seed derived from (seed, k), so chunking/restarting does not change the output and
an interrupted batch continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATE = ROOT / "tools" / "scenegen" / "generate.py"
log = logging.getLogger("batch")

BENCH_SETTINGS = [
    {"samples": 16, "denoiser": "OPTIX", "resolution_scale": 1.0},
    {"samples": 32, "denoiser": "OPTIX", "resolution_scale": 1.0},
    {"samples": 64, "denoiser": "OPTIX", "resolution_scale": 1.0},
    {"samples": 128, "denoiser": "OPTIX", "resolution_scale": 1.0},
    {"samples": 64, "denoiser": "none", "resolution_scale": 1.0},
    {"samples": 64, "denoiser": "OPTIX", "resolution_scale": 0.5},
]


def chunk_commands(
    run: str, total: int, chunk: int, seed: int | None, extra: list[str]
) -> list[list[str]]:
    cmds = []
    for start in range(0, total, chunk):
        n = min(chunk, total - start)
        cmd = [
            "blenderproc",
            "run",
            str(GENERATE),
            "--run",
            run,
            "--start",
            str(start),
            "--scenes",
            str(n),
            "--resume",
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        cmds.append(cmd + extra)
    return cmds


def _write_bench_config(base: Path, setting: dict, out: Path) -> Path:
    """Copy the config with scenegen.render overridden (simple YAML append; later keys win)."""
    import yaml

    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg.setdefault("scenegen", {}).setdefault("render", {}).update(setting)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def benchmark(run: str, scenes: int, extra: list[str], config: Path) -> list[dict]:
    results = []
    out_root = ROOT / "output" / run
    out_root.mkdir(parents=True, exist_ok=True)
    for i, setting in enumerate(BENCH_SETTINGS):
        cfg = _write_bench_config(config, setting, out_root / f"bench_config_{i}.yaml")
        sub = f"{run}/setting_{i}"
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                "blenderproc",
                "run",
                str(GENERATE),
                "--run",
                sub,
                "--scenes",
                str(scenes),
                "--config",
                str(cfg),
                *extra,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        wall = time.perf_counter() - t0
        log_txt = ROOT / "output" / sub / "log.txt"
        per = (
            re.findall(r"([\d.]+) s/scene", log_txt.read_text(encoding="utf-8"))
            if log_txt.is_file()
            else []
        )
        results.append(
            {
                **setting,
                "scenes": scenes,
                "wall_s": round(wall, 1),
                "s_per_scene": float(per[-1]) if per else None,
                "ok": proc.returncode == 0,
            }
        )
        log.info("benchmark %s -> %s", setting, results[-1])
    (out_root / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--total", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--nominal-camera", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--benchmark", action="store_true", help="time several render settings")
    ap.add_argument("--bench-scenes", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    extra = ["--nominal-camera"] if args.nominal_camera else []
    extra_cfg = ["--config", args.config]
    if args.benchmark:
        for r in benchmark(args.run, args.bench_scenes, extra, Path(args.config)):
            print(json.dumps(r))
        return 0
    cmds = chunk_commands(args.run, args.total, args.chunk, args.seed, extra + extra_cfg)
    for i, cmd in enumerate(cmds, 1):
        log.info("chunk %d/%d: %s", i, len(cmds), " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            log.error(
                "chunk %d exited with %d; continuing (rerun the batch to retry failed scenes)",
                i,
                rc,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
