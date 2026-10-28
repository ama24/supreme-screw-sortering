"""Scene generation loop (runs inside `blenderproc run tools/scenegen/generate.py`)."""

from __future__ import annotations

import json
import logging
import math
import random
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from screwsynth.common.config import Config
from screwsynth.common.io import read_json, read_yaml, write_json, write_yaml
from screwsynth.common.paths import repo_root, resolve
from screwsynth.common.png import write_png
from screwsynth.common.seed import derive_seed

from . import camera as camera_mod
from .composition import BalancedSampler, BalanceState, parts_from_assets, plan_scene
from .groundtruth import discard_reason, object_meta, silhouette_area, transform_points
from .materials import pick_insert_finish, sample_material
from .placement import convex_hull_2d, place

log = logging.getLogger("screwsynth.scenegen")

STEEL_DENSITY_KG_MM3 = 7.85e-6
PLASTIC_DENSITY_KG_MM3 = 1.15e-6


@dataclass
class RunArgs:
    run: str
    scenes: int
    seed: int = 0
    start: int = 0
    resume: bool = False
    nominal_camera: bool = False
    preview: bool = False
    benchmark: bool = False


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Part:
    """Asset data needed per instance: part.json, rest poses, local footprints and collision vertices."""

    def __init__(self, assets_dir: Path, rest_dir: Path, part_id: str, collision_reader) -> None:
        self.meta = read_json(assets_dir / part_id / "part.json")
        self.rest = read_json(rest_dir / f"{part_id}.json")
        self.pieces = [
            np.asarray(p.vertices) for p in collision_reader(assets_dir / part_id / "collision.obj")
        ]
        self.verts = np.vstack(self.pieces)
        self.hulls = []
        for pose in self.rest["poses"]:
            v = transform_points(np.asarray(pose["transform"]), self.verts)
            self.hulls.append(convex_hull_2d(v[:, :2]))

    def mass_kg(self, finish: str) -> float:
        density = PLASTIC_DENSITY_KG_MM3 if finish.startswith("nylon") else STEEL_DENSITY_KG_MM3
        return max(1e-5, float(self.meta["mesh"]["volume_mm3"]) * density)

    def sample_pose(self, rng: random.Random) -> int:
        return rng.choices(
            range(len(self.rest["poses"])), weights=[p["probability"] for p in self.rest["poses"]]
        )[0]


def _read_collision(path: Path):
    """Collision OBJ reader without trimesh (inside Blender)."""
    pieces, verts = [], None

    class _P:
        def __init__(self, v):
            self.vertices = np.asarray(v, dtype=float)

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("o "):
            if verts:
                pieces.append(_P(verts))
            verts = []
        elif line.startswith("v ") and verts is not None:
            verts.append([float(x) for x in line.split()[1:4]])
    if verts:
        pieces.append(_P(verts))
    return pieces


def _yaw_matrix(yaw: float, xy: tuple[float, float], lift: float) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3], T[1, 3], T[2, 3] = xy[0], xy[1], lift
    return T


def run(args: RunArgs, cfg: Config) -> dict[str, Any]:
    from . import blender_scene as bs  # bpy only from here on

    sg = dict(cfg.get("scenegen", {}) or {})
    ws = float(cfg.get("world_scale", 10))
    s = 0.001 * ws
    paths = {
        k: resolve(cfg.get(f"paths.{k}", d))
        for k, d in (
            ("assets_dir", "data/assets"),
            ("rest_poses_dir", "data/rest_poses"),
            ("calibration_dir", "data/calibration"),
            ("output_dir", "output"),
            ("classes", "configs/classes.yaml"),
            ("catalogue", "configs/catalogue.yaml"),
        )
    }
    out = paths["output_dir"] / args.run
    scenes_dir = out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    from screwsynth.common.log import setup_logging

    setup_logging(cfg.get("logging.level", "INFO"), out / "log.txt")

    render_cfg = dict(sg.get("render", {}) or {})
    scale = float(render_cfg.get("resolution_scale", 1.0))
    if args.preview:
        pv = dict(sg.get("preview", {}) or {})
        scale = float(pv.get("resolution_scale", 0.25))
        render_cfg["samples"] = int(pv.get("samples", 16))
    if args.nominal_camera:
        nc = dict(sg.get("nominal_camera", {}) or {})
        cam = camera_mod.nominal(
            nc.get("focal_length_mm", 4.74),
            nc.get("pixel_size_um", 1.4),
            tuple(nc.get("resolution", (4608, 2592))),
            nc.get("height_mm", 300.0),
            scale=scale,
        )
        area = (0.0, 0.0, 220.0, 220.0)
    else:
        calib = read_json(paths["calibration_dir"] / "calibration.json")
        cam = camera_mod.from_calibration(calib, scale)
        wa = calib["extrinsic"]["working_area_mm"]
        area = (
            wa["origin"][0],
            wa["origin"][1],
            wa["origin"][0] + wa["size"][0],
            wa["origin"][1] + wa["size"][1],
        )
    vis = cam.visible_plate_rect(-float(sg.get("area_margin_mm", 3.0)))
    area = (max(area[0], vis[0]), max(area[1], vis[1]), min(area[2], vis[2]), min(area[3], vis[3]))

    classes_doc = read_yaml(paths["classes"])
    class_ids = {name: int(i) for i, name in classes_doc["names"].items()}
    from screwsynth.catalogue.schema import load_catalogue

    catalogue = load_catalogue(paths["catalogue"], cfg.get("catalogue.max_length_mm"))
    enabled = {p.id for p in catalogue.enabled_parts}
    infos = parts_from_assets(
        paths["assets_dir"], list(class_ids), enabled, paths["rest_poses_dir"]
    )
    missing = set(class_ids) - {p.class_name for p in infos}
    if missing:
        log.warning("classes without built assets/rest poses (not rendered): %s", sorted(missing))
    state_path = out / "balance_state.json"
    sampler = BalancedSampler(
        infos,
        BalanceState.load(state_path),
        int(sg.get("balance_slack", 1)),
        sg.get("finish_weights"),
    )

    run_cfg = {
        "run": args.run,
        "args": vars(args),
        "seed": args.seed,
        "git_commit": git_commit(),
        "config": cfg.to_dict(),
        "camera": cam.to_json(),
        "area_mm": list(area),
        "classes_hash": classes_doc.get("catalogue_hash"),
        "world_scale": ws,
    }
    if not (out / "run_config.yaml").is_file() or not args.resume:
        write_yaml(out / "run_config.yaml", json.loads(json.dumps(run_cfg)))

    import blenderproc as bproc

    bproc.init()
    bs.init_renderer(render_cfg)
    bs.set_camera(cam, ws)
    bs.check_camera(cam, ws)
    static = bs.StaticScene(cam, sg, paths["calibration_dir"], ws)
    library = bs.AssetLibrary(paths["assets_dir"], ws)
    parts: dict[str, Part] = {}
    physics = dict(sg.get("physics", {}) or {})
    discard_cfg = dict(sg.get("discard", {}) or {})
    variants = max(1, int(sg.get("variants_per_layout", 1)))
    measurable_tilt = float(cfg.get("restposes.measurable_tilt_deg", 15))
    window = int(sg.get("keypoint_window_px", 2))
    stats = {"done": 0, "skipped": 0, "failed": 0, "seconds": []}

    for idx in range(args.start, args.start + args.scenes):
        names = [f"{idx:06d}"] if variants == 1 else [f"{idx:06d}_{v}" for v in range(variants)]
        if args.resume and all((scenes_dir / n / "meta.json").is_file() for n in names):
            stats["skipped"] += 1
            continue
        t0 = time.perf_counter()
        rng = random.Random(derive_seed(args.seed, "scene", idx))
        objs, mats = [], []
        try:
            plan = plan_scene(sampler, sg, rng)
            chosen = []
            for info, finish in plan.parts:
                if info.id not in parts:
                    parts[info.id] = Part(
                        paths["assets_dir"], paths["rest_poses_dir"], info.id, _read_collision
                    )
                pose_i = parts[info.id].sample_pose(rng)
                chosen.append((info, finish, pose_i))
            hulls = [parts[i.id].hulls[p] for i, _, p in chosen]
            placed = place(
                hulls,
                area,
                rng,
                plan.gap_mm,
                plan.touch_probability,
                int(sg.get("max_place_tries", 60)),
            )
            instances = []
            for pl in placed:
                info, finish, pose_i = chosen[pl.index]
                part = parts[info.id]
                T0 = _yaw_matrix(pl.yaw, pl.xy, float(physics.get("drop_mm", 0.05))) @ np.asarray(
                    part.rest["poses"][pose_i]["transform"]
                )
                obj, children = library.instance(info.id, part.meta)
                objs.append(obj)  # registered at once so a failure below still cleans it up
                bs.place_instance(obj, children, T0, s, part.mass_kg(finish), physics)
                material = {
                    "body": sample_material(finish, sg.get("materials"), rng, sg.get("wear"))
                }
                if "insert" in part.meta["mesh"]["material_slots"]:
                    material["insert"] = sample_material(
                        pick_insert_finish(sg.get("insert_colours"), rng),
                        sg.get("materials"),
                        rng,
                        {"scratches": 0, "grime": 0, "fingerprints": 0},
                    )
                bmats = {
                    k: bs.make_material(f"{info.id}_{k}_{len(objs)}", v)
                    for k, v in material.items()
                }
                mats += [m.name for m in bmats.values()]
                bs.assign_materials(obj, bmats)
                instances.append(
                    {
                        "info": info,
                        "finish": finish,
                        "pose": pose_i,
                        "T0": T0,
                        "obj": obj,
                        "material": material,
                    }
                )
            if instances:
                bs.simulate(physics)
            kept, dropped = [], []
            for inst in instances:
                T = bs.final_pose_mm(inst["obj"], s)
                why = discard_reason(
                    T,
                    inst["T0"],
                    parts[inst["info"].id].verts,
                    area,
                    float(discard_cfg.get("max_lift_mm", 0.3)),
                    float(discard_cfg.get("max_move_mm", 15.0)),
                    float(discard_cfg.get("below_plate_mm", 1.0)),
                )
                if why:
                    dropped.append({"part_id": inst["info"].id, "reason": why})
                    inst["obj"].hide(True)
                    for c in inst["obj"].get_children():
                        c.hide(True)
                else:
                    inst["T"] = T
                    kept.append(inst)
            for k, inst in enumerate(kept, start=1):
                inst["obj"].set_cp("category_id", k)
                inst["id"] = k
                rp = inst["obj"].get_rigidbody()
                if rp is not None:
                    inst["obj"].disable_rigidbody()
            for v, name in enumerate(names):
                light = static.jitter(random.Random(derive_seed(args.seed, "light", idx, v)))
                rgb, seg, dep = bs.render(bool(render_cfg.get("depth", False)))
                objects = []
                for inst in kept:
                    part = parts[inst["info"].id]
                    pieces = [transform_points(inst["T"], p) for p in part.pieces]
                    objects.append(
                        object_meta(
                            inst_id=inst["id"],
                            part=part.meta,
                            class_id=class_ids[part.meta["class_name"]],
                            finish=inst["finish"],
                            material=inst["material"]["body"],
                            T_plate=inst["T"],
                            rest_pose_index=inst["pose"],
                            measurable_tilt_deg=measurable_tilt,
                            cam=cam,
                            instance=seg,
                            alone_area=silhouette_area(cam, pieces),
                            window=window,
                        )
                    )
                ids_in_mask = sorted(int(i) for i in np.unique(seg) if i)
                missing_ids = sorted({o["instance_id"] for o in objects} - set(ids_in_mask))
                d = scenes_dir / name
                d.mkdir(parents=True, exist_ok=True)
                write_png(d / "rgb.png", rgb)
                write_png(d / "instance.png", seg)
                if dep is not None:
                    np.save(d / "depth.npy", dep)
                write_json(
                    d / "meta.json",
                    {
                        "scene_id": name,
                        "layout_id": f"{idx:06d}",
                        "variant": v,
                        "seed": args.seed,
                        "scene_kind": plan.kind,
                        "units": "mm",
                        "image": "rgb.png",
                        "instance_mask": "instance.png",
                        "camera": cam.to_json(),
                        "world_scale": ws,
                        "area_mm": list(area),
                        "lighting": light,
                        "objects": objects,
                        "dropped": dropped,
                        "fully_hidden_ids": missing_ids,
                        "render": {
                            "samples": render_cfg.get("samples"),
                            "denoiser": render_cfg.get("denoiser"),
                            "resolution": list(cam.resolution),
                        },
                    },
                )
            sampler.record([inst["info"].class_name for inst in kept])
            sampler.state.save(state_path)
            dt = time.perf_counter() - t0
            stats["done"] += 1
            stats["seconds"].append(dt)
            log.info(
                "scene %s: %d objects (%d dropped), %s, %.1f s",
                names[0],
                len(kept),
                len(dropped),
                plan.kind,
                dt,
            )
        except Exception as exc:  # one bad scene must not stop the run
            stats["failed"] += 1
            log.error("scene %06d failed: %s\n%s", idx, exc, traceback.format_exc())
            with (out / "failed.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"scene": idx, "error": str(exc)}) + "\n")
        finally:
            bs.remove(objs, mats)
            log.debug("datablocks after scene %06d: %s", idx, bs.datablock_counts())
    if stats["seconds"]:
        log.info(
            "rendered %d scenes, %.1f s/scene (mean), skipped %d, failed %d",
            stats["done"],
            float(np.mean(stats["seconds"])),
            stats["skipped"],
            stats["failed"],
        )
    return stats
