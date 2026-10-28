import math

import numpy as np
import pytest

from screwsynth.catalogue.schema import load_catalogue
from screwsynth.common.paths import repo_root
from screwsynth.partgen.assets import Settings as PartSettings
from screwsynth.partgen.assets import part_json
from screwsynth.partgen.builders import Lod, build
from screwsynth.restposes import simulate
from screwsynth.restposes.cli import main
from screwsynth.restposes.compute import (
    Pose,
    Settings,
    _rz,
    canonical_yaw,
    compute_poses,
    equivalent,
    rest_pose_json,
)

LOW = Lod(24, 8, 32)


@pytest.fixture(scope="module")
def catalogue():
    return load_catalogue(repo_root() / "configs" / "catalogue.yaml", max_length_mm=80)


@pytest.fixture(scope="module")
def asset(catalogue):
    cache = {}

    def get(pid):
        if pid not in cache:
            part = catalogue.part(pid)
            b = build(part, LOW)
            meta = part_json(part, b, PartSettings("low", LOW), [])
            cache[pid] = (b.mesh, meta)
        return cache[pid]

    return get


def test_washer_one_dominant_flat_pose(asset):
    poses = compute_poses(*asset("DIN125A_M4"), Settings())
    assert poses[0].category == "flat"
    assert poses[0].probability > 0.85
    assert sum(p.category == "flat" for p in poses) == 1  # top/bottom merged (flip symmetric)


def test_hex_nut_flat_and_edge(asset):
    poses = compute_poses(*asset("DIN934_M4"), Settings())
    cats = {p.category for p in poses}
    assert cats == {"flat", "edge"}


@pytest.mark.parametrize("pid", ["DIN912_M4x16", "DIN912_M6x30", "DIN912_M3x12"])
def test_din912_tilt_matches_analytic(asset, pid):
    mesh, meta = asset(pid)
    d, L = meta["dims"], meta["length"]
    analytic = math.degrees(math.atan((d["dk"] - d["d"]) / 2 / L))
    side = [p for p in compute_poses(mesh, meta, Settings()) if p.category == "side"]
    assert side[0].probability > 0.5
    assert side[0].axis_tilt_deg == pytest.approx(analytic, abs=1.5)
    assert side[0].measurable


def test_short_screw_can_stand_on_head(asset):
    cats = {p.category for p in compute_poses(*asset("DIN912_M3x5"), Settings())}
    assert "standing_on_head" in cats


def test_countersunk_rocks_on_cone(asset):
    poses = compute_poses(*asset("DIN7991_M4x20"), Settings())
    assert poses[0].category == "side" and 2 < poses[0].axis_tilt_deg < 15


@pytest.mark.parametrize(
    "pid", ["DIN912_M4x16", "DIN933_M6x30", "ISO7380_M5x12", "DIN985_M6", "TNUT_DROP_S8_M5"]
)
def test_lowest_vertex_on_plate_and_heights(asset, pid):
    mesh, meta = asset(pid)
    poses = compute_poses(mesh, meta, Settings())
    assert sum(p.probability for p in poses) == pytest.approx(1.0)
    for p in poses:
        v = mesh.vertices @ p.transform[:3, :3].T + p.transform[:3, 3]
        assert abs(v[:, 2].min()) < 1e-6
        assert np.allclose(p.transform[:3, :3] @ p.transform[:3, :3].T, np.eye(3), atol=1e-9)
        kp = np.asarray(meta["keypoints"]["tip"]["xyz"])
        assert p.keypoint_heights["tip"] == pytest.approx(
            (p.transform[:3, :3] @ kp + p.transform[:3, 3])[2]
        )
        assert all(h >= -1e-6 for h in p.keypoint_heights.values())


def test_overrides_and_threshold(asset):
    mesh, meta = asset("DIN912_M3x5")
    s = Settings(overrides=({"match": "DIN912*", "category": "standing_on_head", "weight": 0},))
    poses = compute_poses(mesh, meta, s)
    assert all(p.category != "standing_on_head" for p in poses)
    assert sum(p.probability for p in poses) == pytest.approx(1.0)
    strict = compute_poses(mesh, meta, Settings(min_probability=0.3))
    assert all(p.probability >= 0.3 or p.raw_probability >= 0.3 for p in strict)


def test_equivalence_rules():
    R = _rz(0.3) @ np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])  # axis horizontal
    assert equivalent(R, _rz(1.2) @ R, "hex_nut", 1.0)  # yaw only
    roll = np.array(
        [[math.cos(0.4), -math.sin(0.4), 0], [math.sin(0.4), math.cos(0.4), 0], [0, 0, 1]]
    )
    assert equivalent(R, R @ roll, "socket_cap", 1.0)  # continuous roll symmetry
    assert not equivalent(R, R @ roll, "hex_nut", 1.0)  # 0.4 rad is not a 60 deg multiple
    up, down = np.eye(3), np.diag([1.0, -1, -1])
    assert equivalent(up, down, "washer", 1.0) and not equivalent(up, down, "nyloc_nut", 1.0)


def test_canonical_yaw_points_axis_along_x():
    T = np.eye(4)
    T[:3, :3] = _rz(1.0) @ np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])
    R = canonical_yaw(T)[:3, :3]
    assert R[1, 2] == pytest.approx(0, abs=1e-12) and R[0, 2] > 0


def test_json_schema(asset):
    mesh, meta = asset("DIN912_M4x16")
    poses = compute_poses(mesh, meta, Settings())
    doc = rest_pose_json(meta, poses, Settings())
    pose = doc["poses"][0]
    assert {
        "transform",
        "probability",
        "axis_tilt_deg",
        "measurable",
        "axis_height_head_mm",
        "axis_height_tip_mm",
        "keypoint_heights_mm",
    } <= set(pose)
    assert np.asarray(pose["transform"]).shape == (4, 4)
    assert doc["roll_symmetry"] == "continuous"


def test_simulation_comparison_logic():
    poses = [
        Pose(np.eye(4), 0.7, 0.7, "flat", 90.0),
        Pose(
            np.c_[np.r_[np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]), [[0, 0, 0]]], [0, 0, 0, 1]],
            0.3,
            0.3,
            "edge",
            0.0,
        ),
    ]
    flat = _rz(0.5)
    edge = _rz(2.0) @ poses[1].transform[:3, :3]
    assert simulate.classify(flat, poses, "hex_nut") == 0
    assert simulate.classify(edge, poses, "hex_nut") == 1
    tilted = np.array(
        [[1.0, 0, 0], [0, math.cos(0.7), -math.sin(0.7)], [0, math.sin(0.7), math.cos(0.7)]]
    )
    assert simulate.classify(tilted, poses, "hex_nut") is None
    ok = simulate.compare([0] * 7 + [1] * 3, poses)
    assert ok.disagreements == []
    bad = simulate.compare([1] * 8 + [None] * 2, poses)
    assert len(bad.disagreements) == 3  # both probabilities off, 20% unmatched


@pytest.mark.slow
@pytest.mark.skipif(not simulate.available(), reason="pybullet not installed")
def test_drop_validation(asset):
    mesh, meta = asset("DIN125A_M4")
    poses = compute_poses(mesh, meta, Settings())
    cmp = simulate.validate(mesh, poses, meta["shape"], 20)
    assert cmp is not None and cmp.empirical[0] > 0.6


def test_cli(tmp_path, catalogue, capsys):
    from screwsynth.partgen.assets import Settings as PS
    from screwsynth.partgen.assets import build_asset

    assets_dir = tmp_path / "assets"
    for pid in ("DIN934_M3", "DIN912_M3x8"):
        assert (
            build_asset(catalogue.part(pid), assets_dir, PS("low", LOW), preview=False).status
            == "built"
        )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "seed: 0\ncatalogue: {max_length_mm: 80}\n"
        f"paths: {{assets_dir: {assets_dir.as_posix()}, rest_poses_dir: {(tmp_path / 'rp').as_posix()},"
        f" catalogue: {(repo_root() / 'configs' / 'catalogue.yaml').as_posix()}}}\n"
        "restposes: {min_probability: 0.02}\n",
        encoding="utf-8",
    )
    args = ["--config", str(cfg)]
    assert main(["compute", "--id", "DIN934_M3", "DIN912_M3x8", *args]) == 0
    assert "computed: 2" in capsys.readouterr().out
    assert main(["compute", "--id", "DIN934_M3", *args]) == 0
    assert "skipped: 1" in capsys.readouterr().out
    assert main(["show", "--id", "DIN912_M3x8", *args]) == 0
    assert (tmp_path / "rp" / "DIN912_M3x8.png").is_file()
    assert main(["summary", *args]) == 0
    assert "DIN934" in capsys.readouterr().out
