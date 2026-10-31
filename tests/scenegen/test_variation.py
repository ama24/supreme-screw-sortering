import math
import subprocess
from collections import Counter

import numpy as np
import pytest

from screwsynth.common.paths import repo_root
from screwsynth.scenegen import camera as cammod
from screwsynth.scenegen import environments as envs
from screwsynth.scenegen.variation import (
    CameraSettings,
    EnvironmentSettings,
    layout_environment,
    layout_view,
    parse_value,
    sequence_point,
    steepest_ray_deg,
    view_envelope,
    view_pose,
)

TARGET = (110.0, 110.0)


def _spread(values, lo, hi, bins=10):
    counts = Counter(min(bins - 1, int((v - lo) / (hi - lo) * bins)) for v in values)
    return [counts.get(b, 0) for b in range(bins)]


@pytest.mark.parametrize("n", [50, 412, 1000])
def test_every_range_is_covered_evenly_for_any_number_of_layouts(n):
    settings = CameraSettings.from_config(
        {
            "height_mm": [250, 400],
            "tilt_deg": [0, 30],
            "tilt_direction_deg": [0, 360],
            "roll_deg": [0, 360],
        }
    )
    base = cammod.nominal(scale=0.1)
    views = [layout_view(base, settings, TARGET, i, seed=7).values for i in range(n)]
    for key, lo, hi in (
        ("height_mm", 250, 400),
        ("tilt_deg", 0, 30),
        ("tilt_direction_deg", 0, 360),
        ("roll_deg", 0, 360),
    ):
        counts = _spread([v[key] for v in views], lo, hi)
        expected = n / 10
        assert max(abs(c - expected) for c in counts) <= max(2, 0.15 * expected), (key, counts)
    # combinations: steep tilts are not bunched at one height
    steep = [v["height_mm"] for v in views if v["tilt_deg"] > 20]
    assert min(steep) < 290 and max(steep) > 360


def test_layouts_added_later_fill_the_gaps():
    """500 layouts, then 500 more: the second batch alone is even too (no repeats, no clumps)."""
    first = [sequence_point(i, 3)["tilt"] for i in range(500)]
    second = [sequence_point(i, 3)["tilt"] for i in range(500, 1000)]
    for batch in (first, second, first + second):
        counts = _spread(batch, 0, 1)
        assert max(counts) - min(counts) <= max(3, 0.1 * len(batch) / 10)
    assert not set(first) & set(second)


def test_same_seed_same_view_other_seed_other_view():
    s = CameraSettings.from_config({"tilt_deg": [0, 30], "roll_deg": [0, 360]})
    base = cammod.nominal()
    a = layout_view(base, s, TARGET, 41, seed=1)
    b = layout_view(base, s, TARGET, 41, seed=1)
    c = layout_view(base, s, TARGET, 41, seed=2)
    assert np.array_equal(a.camera.T_cam_plate, b.camera.T_cam_plate)
    assert not np.allclose(a.camera.T_cam_plate, c.camera.T_cam_plate)


def test_defaults_are_the_rig_camera_unchanged():
    s = CameraSettings.from_config({})
    assert s.is_rig_view and s.height == "calibrated" and s.tilt == 0 and s.roll == 0
    base = cammod.nominal()
    v = layout_view(base, s, TARGET, 5, seed=0)
    assert v.rig_view and v.camera is base
    # a fixed value of 0 and the base height computed explicitly give the same pose too
    T = view_pose(base, TARGET, 300.0, 0.0, 123.0, 0.0)
    assert np.allclose(T, base.T_cam_plate, atol=1e-9)


def test_fixed_values_and_ranges():
    assert parse_value(20, "t") == 20.0
    assert parse_value([0, 30], "t") == (0.0, 30.0)
    assert parse_value([15, 15], "t") == 15.0  # an empty range is a fixed value
    assert parse_value("calibrated", "h", ("calibrated",)) == "calibrated"
    with pytest.raises(ValueError):
        parse_value([30, 0], "t")
    with pytest.raises(ValueError):
        parse_value("high", "t")
    s = CameraSettings.from_config({"tilt_deg": 20, "tilt_direction_deg": 90})
    views = [layout_view(cammod.nominal(), s, TARGET, i, 0).values for i in range(20)]
    assert {v["tilt_deg"] for v in views} == {20.0} and {
        v["tilt_direction_deg"] for v in views
    } == {90.0}


@pytest.mark.parametrize(
    "tilt,direction,roll", [(0, 0, 45), (25, 0, 0), (25, 90, 0), (35, 210, 300)]
)
def test_view_pose_looks_at_the_target_from_the_set_height(tilt, direction, roll):
    base = cammod.nominal()
    cam = layout_view(
        base,
        CameraSettings.from_config(
            {"height_mm": 350, "tilt_deg": tilt, "tilt_direction_deg": direction, "roll_deg": roll}
        ),
        TARGET,
        0,
        0,
    ).camera
    c = cam.centre_in_plate
    assert c[2] == pytest.approx(350.0)
    # the camera stands on the tilt-direction side of the target
    off = math.hypot(c[0] - TARGET[0], c[1] - TARGET[1])
    assert off == pytest.approx(350.0 * math.tan(math.radians(tilt)), abs=1e-6)
    if tilt:
        assert math.degrees(math.atan2(c[1] - TARGET[1], c[0] - TARGET[0])) % 360 == pytest.approx(
            direction % 360, abs=1e-6
        )
    # the optical axis goes through the target: it lands on the principal point
    uv = cam.project(np.array([[*TARGET, 0.0]]))[0]
    assert np.allclose(uv, cam.K[:2, 2], atol=1e-6)
    # a proper rotation (no mirroring)
    assert np.linalg.det(cam.T_cam_plate[:3, :3]) == pytest.approx(1.0)


def test_envelope_covers_every_view_and_refuses_the_horizon():
    base = cammod.nominal(scale=0.1)
    s = CameraSettings.from_config(
        {"height_mm": [250, 400], "tilt_deg": [0, 30], "roll_deg": [0, 360]}
    )
    env = view_envelope(base, s, TARGET)
    x0, y0, x1, y1 = env["rect"]
    for i in range(0, 200, 7):
        r = layout_view(base, s, TARGET, i, 0).camera.visible_plate_rect(0.0)
        assert x0 - 1e-6 <= r[0] and y0 - 1e-6 <= r[1] and r[2] <= x1 + 1e-6 and r[3] <= y1 + 1e-6
    assert env["max_height_mm"] == pytest.approx(400.0)
    assert env["max_offset_mm"] == pytest.approx(400 * math.tan(math.radians(30)), rel=1e-6)
    assert steepest_ray_deg(base) < 30
    with pytest.raises(ValueError, match="horizon"):
        view_envelope(base, CameraSettings.from_config({"tilt_deg": 75}), TARGET)


def test_hdri_files_are_used_equally_often(tmp_path):
    for name in ("a.hdr", "b.exr", "c.hdr", "d.hdr", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert envs.files(tmp_path) == ["a.hdr", "b.exr", "c.hdr", "d.hdr"]
    envs.set_enabled(tmp_path, "c.hdr", False)
    files = envs.usable(tmp_path)
    assert files == ["a.hdr", "b.exr", "d.hdr"]
    s = EnvironmentSettings.from_config({"enabled": True, "strength": [0.5, 1.0]})
    picks = [layout_environment(files, s, i, 0) for i in range(300)]
    counts = Counter(p["file"] for p in picks)
    assert set(counts) == set(files) and max(counts.values()) - min(counts.values()) <= 1
    assert all(0.5 <= p["strength"] <= 1.0 and 0 <= p["rotation_deg"] < 360 for p in picks)
    assert layout_environment(files, EnvironmentSettings.from_config({}), 0, 0) is None  # off
    assert layout_environment([], s, 0, 0) is None  # no files: as usual


def test_config_defaults_are_the_usual_dataset():
    from screwsynth.common.config import load_config

    cfg = load_config(repo_root() / "configs" / "config.yaml")
    assert CameraSettings.from_config(cfg.get("scenegen.camera")).is_rig_view
    assert not EnvironmentSettings.from_config(cfg.get("scenegen.environment")).enabled


@pytest.mark.slow
def test_blender_varied_views_and_hdri(tmp_path):
    """Tiny preview with tilted views and an HDRI: views recorded, nothing measurable."""
    import json

    import cv2

    hdri = tmp_path / "hdri"
    hdri.mkdir()
    sky = np.zeros((64, 128, 3), np.float32)
    sky[:32] = (2.0, 1.5, 1.0)  # a bright upper half
    cv2.imwrite(str(hdri / "test.hdr"), sky)
    cfg = tmp_path / "config.yaml"
    base = (repo_root() / "configs" / "config.yaml").read_text(encoding="utf-8")
    base = base.replace("  output_dir: output\n", f"  output_dir: {tmp_path.as_posix()}\n", 1)
    base = base.replace("  hdri_dir: data/hdri", f"  hdri_dir: {hdri.as_posix()}", 1)
    base = base.replace("    tilt_deg: 0 ", "    tilt_deg: [10, 30] ", 1).replace(
        "    roll_deg: 0 ", "    roll_deg: [0, 360] ", 1
    )
    base = base.replace(
        "    enabled: false             # true = an HDRI", "    enabled: true  # true = an HDRI", 1
    )
    cfg.write_text(base, encoding="utf-8")
    r = subprocess.run(
        [
            "blenderproc",
            "run",
            str(repo_root() / "tools/scenegen/generate.py"),
            "--run",
            "views",
            "--preview",
            "--nominal-camera",
            "--scenes",
            "2",
            "--config",
            str(cfg),
        ],
        cwd=repo_root(),
        timeout=1800,
    )
    assert r.returncode == 0
    metas = [
        json.loads((d / "meta.json").read_text())
        for d in sorted((tmp_path / "views" / "scenes").iterdir())
    ]
    assert metas
    for m in metas:
        assert not m["camera_view"]["rig_view"] and 10 <= m["camera_view"]["tilt_deg"] <= 30
        assert m["lighting"]["environment"]["file"] == "test.hdr"
        assert not any(o["measurable"] for o in m["objects"])
