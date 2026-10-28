import math
import random
import shutil
import subprocess
from collections import Counter

import cv2
import numpy as np
import pytest

from screwsynth.calibration.geometry import Camera
from screwsynth.common.paths import repo_root
from screwsynth.common.png import read_png, write_png
from screwsynth.scenegen import camera as cammod
from screwsynth.scenegen.composition import BalancedSampler, BalanceState, PartInfo, plan_scene
from screwsynth.scenegen.groundtruth import (
    OCCLUDED,
    VISIBLE,
    discard_reason,
    keypoint_visibility,
    silhouette_area,
)
from screwsynth.scenegen.materials import DEFAULT_PRESETS, pick_insert_finish, sample_material
from screwsynth.scenegen.placement import convex_hull_2d, inside_area, place, separation

# ----------------------------------------------------------------------------------- PNG


def test_png_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    mask = rng.integers(0, 65535, (37, 53), dtype=np.uint16)
    gray = rng.integers(0, 256, (10, 7), dtype=np.uint8)
    write_png(tmp_path / "rgb.png", rgb)
    write_png(tmp_path / "mask.png", mask)
    write_png(tmp_path / "gray.png", gray)
    assert np.array_equal(read_png(tmp_path / "rgb.png"), rgb)
    assert np.array_equal(read_png(tmp_path / "mask.png"), mask)
    assert np.array_equal(read_png(tmp_path / "gray.png"), gray)
    # OpenCV reads them too (BGR order, 16-bit preserved)
    assert np.array_equal(cv2.imread(str(tmp_path / "rgb.png"))[..., ::-1], rgb)
    assert np.array_equal(cv2.imread(str(tmp_path / "mask.png"), cv2.IMREAD_UNCHANGED), mask)


def test_png_reads_opencv_files(tmp_path):
    img = np.random.default_rng(1).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "cv.png"), img)  # OpenCV uses adaptive filters
    assert np.array_equal(read_png(tmp_path / "cv.png"), img[..., ::-1])


# -------------------------------------------------------------------------------- camera


def test_nominal_camera_geometry():
    cam = cammod.nominal()
    assert cam.resolution == (2560, 2560)
    assert cam.K[0, 2] == pytest.approx((4608 - 1) / 2 - cam.roi["x0"])
    centre = cam.project(np.array([[110.0, 110.0, 0.0]]))[0]
    assert np.allclose(centre, [(2560 - 1) / 2, (2560 - 1) / 2], atol=1e-6)
    # plate x to the right, plate y up in the image
    p = cam.project(np.array([[120.0, 110, 0], [110, 120, 0]]))
    assert p[0, 0] > centre[0] and p[1, 1] < centre[1]
    assert cam.visible_plate_rect()[2] - cam.visible_plate_rect()[0] == pytest.approx(
        2560 / (3385.7 / 300), rel=1e-3
    )


def test_camera_matches_calibration_geometry():
    """The render camera (ROI crop) projects like the full calibrated camera minus the ROI offset."""
    full = Camera(
        np.array([[3400.0, 0, 2310], [0, 3405, 1290], [0, 0, 1]]), np.zeros(5), (4608, 2592)
    )
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array([math.pi + 0.01, 0.02, 0.3]))[0]
    T[:3, 3] = [-100.0, 110.0, 301.0]
    full.T_cam_plate = T
    calib = {
        "camera_matrix": full.K.tolist(),
        "dist_coeffs": [0] * 5,
        "image_size": [4608, 2592],
        "roi": {"x0": 1000, "y0": 16, "w": 2560, "h": 2560},
        "extrinsic": {"T_cam_plate": T.tolist()},
    }
    cam = cammod.from_calibration(calib)
    pts = np.array([[10.0, -20.0, 0.0], [150.0, -100.0, 5.0], [60.0, -60.0, 2.5]])
    assert np.allclose(cam.project(pts), full.project(pts, distort=False) - [1000, 16], atol=1e-9)
    half = cammod.from_calibration(calib, scale=0.5)
    assert np.allclose(half.project(pts), (cam.project(pts) + 0.5) * 0.5 - 0.5, atol=1e-9)


def test_blender_pose_convention():
    """cam2world (Blender: camera looks along -z) maps the optical axis onto the camera's view ray."""
    cam = cammod.nominal()
    M = cam.cam2world_blender(world_scale=10)
    look = M[:3, :3] @ np.array([0, 0, -1.0])
    assert np.allclose(look, [0, 0, -1])  # straight down
    assert np.allclose(M[:3, 3], np.array([110, 110, 300]) * 0.01)
    up_image = M[:3, :3] @ np.array([0, 1.0, 0])  # Blender camera +y = image up = plate +y here
    assert np.allclose(up_image, [0, 1, 0])


# ------------------------------------------------------------------------- composition


def _parts(n_classes=20, lengths=4):
    out = []
    for c in range(n_classes):
        for k in range(lengths if c % 2 == 0 else 1):
            out.append(
                PartInfo(
                    f"C{c:02d}_{k}",
                    f"C{c:02d}",
                    "screw",
                    ("zinc_clear", "black_oxide"),
                    float(k + 1),
                )
            )
    return out


def test_sampler_balances_classes():
    sampler = BalancedSampler(_parts(), slack=1)
    rng = random.Random(0)
    for _ in range(1000):
        part, finish = sampler.draw(rng)
        sampler.record([part.class_name])
    counts = list(sampler.state.counts.values())
    assert sum(counts) == 1000
    assert max(counts) - min(counts) <= 2  # 50 per class, spread <= slack + 1


def test_sampler_recovers_from_imbalance_and_lengths_uniform():
    state = BalanceState(counts={"C00": 300})
    sampler = BalancedSampler(_parts(), state=state)
    rng = random.Random(1)
    drawn = Counter()
    lengths = Counter()
    for _ in range(800):
        part, _ = sampler.draw(rng)
        sampler.record([part.class_name])
        drawn[part.class_name] += 1
        if part.class_name == "C02":
            lengths[part.length] += 1
    assert drawn["C00"] < 10  # the over-represented class is avoided
    assert set(lengths) == {1.0, 2.0, 3.0, 4.0}


def test_balance_state_persists(tmp_path):
    s = BalanceState(counts={"a": 3}, scenes=2)
    s.save(tmp_path / "b.json")
    assert BalanceState.load(tmp_path / "b.json") == s
    assert BalanceState.load(tmp_path / "missing.json") == BalanceState()


def test_plan_scene_kinds_and_determinism():
    cfg = {
        "objects": [15, 40],
        "few_objects": [1, 5],
        "few_fraction": 0.2,
        "crowded_fraction": 0.3,
        "crowded_objects": [30, 50],
    }
    kinds = Counter()
    for i in range(300):
        plan = plan_scene(BalancedSampler(_parts()), cfg, random.Random(i))
        kinds[plan.kind] += 1
        lo, hi = {"few": (1, 5), "crowded": (30, 50), "normal": (15, 40)}[plan.kind]
        assert lo <= len(plan.parts) <= hi
    assert 30 < kinds["few"] < 90 and 60 < kinds["crowded"] < 130
    a = plan_scene(BalancedSampler(_parts()), cfg, random.Random(7))
    b = plan_scene(BalancedSampler(_parts()), cfg, random.Random(7))
    assert [(p.id, f) for p, f in a.parts] == [(p.id, f) for p, f in b.parts]


# ---------------------------------------------------------------------------- placement


def _rect(w, h):
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])


def test_separation():
    a = _rect(2, 2)
    assert separation(a, a + [5, 0]) == pytest.approx(3)
    assert separation(a, a + [1.5, 0]) == pytest.approx(-0.5)
    tri = convex_hull_2d(np.array([[0, 0], [1, 0], [0, 1], [0.2, 0.2]]))
    assert len(tri) == 3


def test_place_no_overlap_inside_area():
    rng = random.Random(0)
    hulls = [_rect(rng.uniform(4, 40), rng.uniform(3, 8)) for _ in range(40)]
    area = (0, 0, 220, 220)
    placed = place(hulls, area, rng, gap_mm=1.0)
    assert len(placed) >= 35
    for i, a in enumerate(placed):
        assert inside_area(a.footprint, area)
        for b in placed[i + 1 :]:
            assert separation(a.footprint, b.footprint) >= 1.0 - 1e-9


def test_place_touching():
    rng = random.Random(3)
    hulls = [_rect(20, 4) for _ in range(20)]
    placed = place(hulls, (0, 0, 120, 120), rng, gap_mm=-0.3, touch_probability=1.0)
    seps = [min(separation(a.footprint, b.footprint) for b in placed if b is not a) for a in placed]
    assert len(placed) >= 15
    assert (
        sum(abs(s + 0.3) < 0.05 for s in seps) >= len(placed) // 2
    )  # most parts touch a neighbour
    assert min(seps) >= -0.3 - 1e-6


def test_discard_rules():
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 2, 2.0]])
    T = np.eye(4)
    T[:3, 3] = [50, 50, 0]
    area = (0, 0, 100, 100)
    assert discard_reason(T, T, verts, area) is None
    lifted = T.copy()
    lifted[2, 3] = 2.0
    assert "stacked" in discard_reason(lifted, T, verts, area)
    moved = T.copy()
    moved[0, 3] += 30
    assert "moved" in discard_reason(moved, T, verts, area)
    off = T.copy()
    off[0, 3] = 150
    assert "left" in discard_reason(off, T, verts, area)
    sunk = T.copy()
    sunk[2, 3] = -5
    assert "fell" in discard_reason(sunk, T, verts, area)


# ----------------------------------------------------------------------- ground truth


def test_keypoint_visibility():
    inst = np.zeros((20, 20), np.uint16)
    inst[5:10, 5:15] = 3
    inst[5:10, 12:15] = 4  # part 4 covers the right end of part 3
    uv = np.array([[7.0, 7.0], [13.0, 7.0], [4.0, 7.0], [30.0, 7.0]])
    assert keypoint_visibility(uv, inst, 3) == [VISIBLE, OCCLUDED, VISIBLE, 0]


def test_silhouette_area_matches_rasterised_box():
    cam = cammod.nominal(scale=0.25)
    box = np.array([[x, y, z] for x in (100, 120) for y in (105, 109) for z in (0, 4.0)])
    total, inside = silhouette_area(cam, [box])
    px = cam.project(box)
    w = px[:, 0].max() - px[:, 0].min()
    h = px[:, 1].max() - px[:, 1].min()
    assert total == pytest.approx(w * h, rel=0.12)
    assert inside == total
    far = box + [400, 0, 0]
    t2, i2 = silhouette_area(cam, [far])
    assert t2 > 0 and i2 == 0


# --------------------------------------------------------------------------- materials


def test_materials_sampled_in_range():
    rng = random.Random(0)
    for finish, preset in DEFAULT_PRESETS.items():
        m = sample_material(finish, None, rng)
        lo, hi = (
            preset["roughness"]
            if isinstance(preset["roughness"], list)
            else (preset["roughness"],) * 2
        )
        assert lo <= m["roughness"] <= hi
        assert all(0 <= c <= 1 for c in m["base_color"])
    override = sample_material("brass", {"brass": {"base_color": [1, 0, 0], "roughness": 0.5}}, rng)
    assert override["roughness"] == 0.5 and override["base_color"][0] > 0.9
    with pytest.raises(KeyError):
        sample_material("aluminium", None, rng)


def test_generate_script_imports_blenderproc_first():
    """`blenderproc run` refuses scripts whose first non-comment line is not the blenderproc import."""
    text = (repo_root() / "tools" / "scenegen" / "generate.py").read_text(encoding="utf-8")
    lines = [s.strip() for s in text.splitlines()]
    first = next(s for s in lines if s and not s.startswith("#"))
    assert first.startswith("import blenderproc")


def test_insert_colour_weights():
    rng = random.Random(0)
    picks = Counter(pick_insert_finish(None, rng) for _ in range(2000))
    assert set(picks) == {"insert_blue", "insert_white"}
    assert 0.65 < picks["insert_blue"] / 2000 < 0.75
    assert {pick_insert_finish({"insert_white": 1, "insert_blue": 0}, rng) for _ in range(50)} == {
        "insert_white"
    }
    a = [pick_insert_finish(None, random.Random(5)) for _ in range(3)]
    assert a == [pick_insert_finish(None, random.Random(5)) for _ in range(3)]
    with pytest.raises(ValueError):
        pick_insert_finish({"insert_blue": 0}, rng)


# ------------------------------------------------------------------------- Blender smoke


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("blenderproc") is None, reason="blenderproc not installed")
def test_blender_smoke(tmp_path):
    """One tiny scene with the nominal camera; instance ids in meta.json match instance.png."""
    import json

    cfg = tmp_path / "config.yaml"
    base = (repo_root() / "configs" / "config.yaml").read_text(encoding="utf-8")
    cfg.write_text(
        base.replace("paths:\n", f"paths:\n  output_dir: {tmp_path.as_posix()}\n", 1),
        encoding="utf-8",
    )
    r = subprocess.run(
        [
            "blenderproc",
            "run",
            str(repo_root() / "tools/scenegen/generate.py"),
            "--run",
            "smoke",
            "--preview",
            "--nominal-camera",
            "--config",
            str(cfg),
        ],
        cwd=repo_root(),
        timeout=1800,
    )
    assert r.returncode == 0
    scene = sorted((tmp_path / "smoke" / "scenes").iterdir())[0]
    meta = json.loads((scene / "meta.json").read_text())
    mask = read_png(scene / "instance.png")
    ids = {o["instance_id"] for o in meta["objects"]}
    assert set(np.unique(mask)) - {0} <= ids
