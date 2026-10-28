import importlib.util
import math
import sys

import cv2
import numpy as np
import pytest

from screwsynth.calibration import extrinsic, intrinsics, measure, sensor
from screwsynth.calibration.board import BoardSpec, detect, write_board_pdf
from screwsynth.calibration.cli import main
from screwsynth.calibration.geometry import Camera, nominal_camera, roi_camera_matrix, roi_for_area
from screwsynth.calibration.synthetic import (
    intrinsic_views,
    make_sample,
    pose_matrix,
    render_board,
    render_plane,
    small_camera,
)
from screwsynth.common.io import read_json
from screwsynth.common.paths import repo_root


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    out = tmp_path_factory.mktemp("calib_sample")
    truth = make_sample(out, seed=1)
    return out, truth


def _undistort_field(K, dist, size, step=40):
    w, h = size
    pts = (
        np.stack(np.meshgrid(np.arange(0, w, step), np.arange(0, h, step)), -1)
        .reshape(-1, 1, 2)
        .astype(float)
    )
    return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def test_intrinsics_recover_ground_truth(sample):
    out, truth = sample
    files = sorted((out / "intrinsics").glob("*.png"))
    res = intrinsics.calibrate(files, BoardSpec())
    K_true = np.asarray(truth["camera_matrix"])
    assert res.rms < 0.3
    assert res.K[0, 0] == pytest.approx(K_true[0, 0], rel=5e-3)
    assert res.K[1, 1] == pytest.approx(K_true[1, 1], rel=5e-3)
    assert abs(res.K[0, 2] - K_true[0, 2]) < 8 and abs(res.K[1, 2] - K_true[1, 2]) < 8
    # what matters: the undistortion maps agree wherever the board was observed
    f_est = _undistort_field(res.K, res.dist, res.image_size)
    f_true = _undistort_field(K_true, np.asarray(truth["dist_coeffs"]), res.image_size)
    w, h = res.image_size
    grid = (
        np.stack(np.meshgrid(np.arange(0, w, 40), np.arange(0, h, 40)), -1)
        .reshape(-1, 2)
        .astype(np.float32)
    )
    hull = cv2.convexHull(res.residuals[:, :2].astype(np.float32))
    inside = np.array(
        [cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in grid]
    )
    assert inside.mean() > 0.6
    assert np.abs(f_est - f_true)[inside].max() < 0.5
    assert all(v.used for v in res.views)


def test_view_rejection(sample, tmp_path):
    out, _ = sample
    files = sorted((out / "intrinsics").glob("*.png"))[:6]
    blurry = tmp_path / "blurry.png"
    cv2.imwrite(str(blurry), cv2.GaussianBlur(cv2.imread(str(files[0])), (0, 0), 1.6))
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((960, 1280), 128, np.uint8))
    views, size = intrinsics.load_views([*files, blurry, blank], BoardSpec())
    used = intrinsics.select_views(views, 12, 0.4)
    reasons = {v.file: v.reason for v in views if not v.used}
    assert len(used) == 6
    assert "blurry" in reasons["blurry.png"] and "not found" in reasons["blank.png"]


def test_extrinsic_plate_frame(sample):
    out, truth = sample
    cam = Camera(
        np.asarray(truth["camera_matrix"]),
        np.asarray(truth["dist_coeffs"]),
        tuple(truth["image_size"]),
    )
    img = cv2.imread(str(out / "extrinsic" / "board_flat.png"))
    ext = extrinsic.solve(
        img,
        BoardSpec(),
        cam,
        thickness_mm=2.0,
        area={"origin_mm": [0, -105], "size_mm": [135, 105]},
    )
    assert ext["reprojection_rms_px"] < 0.4
    assert ext["camera_height_mm"] == pytest.approx(
        300 + 2.0, abs=0.3
    )  # z = 0 is 2 mm below the board
    T = np.asarray(ext["T_cam_plate"])
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    assert T[:3, 2] @ np.array([0, 0, -1.0]) > 0.99  # plate z points towards the camera
    full = Camera(cam.K, cam.dist, cam.image_size, T)
    # board corner (plate origin) projects onto the detected first chessboard corner
    d = detect(img, BoardSpec())
    first = d.corners[np.argmin(d.ids)]
    obj = d.object_points[np.argmin(d.ids)]
    proj = full.project(np.array([[obj[0], -obj[1], 2.0]]))[0]
    assert np.linalg.norm(proj - first) < 0.5
    roi = ext["roi"]
    assert roi["w"] % 32 == 0 and roi["h"] % 32 == 0
    assert ext["px_per_mm"] == pytest.approx(1100 / 302, rel=0.02)


def test_homography_and_parallax():
    cam = nominal_camera()
    pts = np.array([[10.0, 20.0, 0], [200.0, 150.0, 0], [110.0, 110.0, 0]])
    px = cam.project(pts)
    H = cam.homography()
    hp = (H @ np.c_[pts[:, :2], np.ones(3)].T).T
    assert np.allclose(hp[:, :2] / hp[:, 2:], px, atol=1e-6)
    back = cam.pixel_to_plate(px, 0.0)
    assert np.allclose(back, pts, atol=1e-6)
    high = np.array([[30.0, 40.0, 5.0]])
    assert np.allclose(cam.pixel_to_plate(cam.project(high), 5.0), high, atol=1e-6)
    assert cam.px_per_mm((110, 110)) == pytest.approx(4.74 / 1.4e-3 / 300, rel=1e-6)
    roi = roi_for_area(cam, (0, 0), (220, 220), 5)
    # 230 mm at 11.3 px/mm exceeds the 2592 px sensor height: the ROI is clipped to the image
    assert (
        roi["w"] % 32 == 0 and roi["h"] % 32 == 0 and 2560 <= roi["w"] <= 2624 and roi["h"] <= 2592
    )
    Kr = roi_camera_matrix(cam.K, roi)
    assert Kr[0, 2] == pytest.approx(cam.K[0, 2] - roi["x0"])


def test_measurement_on_synthetic_image():
    """Dark bars of known length on a green plate, seen through a distorted camera."""
    cam = small_camera()
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = -T[:3, :3] @ np.array([70.0, -50.0, 150.0])  # ~7 px/mm
    cam.T_cam_plate = T
    ppm = 20.0
    origin = (-10.0, -110.0)
    tex = np.zeros((int(130 * ppm), int(170 * ppm), 3), np.uint8)
    tex[:] = (60, 150, 70)
    bars = [(5, -95, 20.0), (5, -60, 35.5), (40, -30, 50.0), (100, -90, 12.0)]
    for x, y, L in bars:
        r0, c0 = int((y - origin[1]) * ppm), int((x - origin[0]) * ppm)
        tex[r0 : r0 + int(4 * ppm), c0 : c0 + int(L * ppm)] = (40, 40, 40)
    img = render_plane(cam, T, tex, ppm, (60, 150, 70), origin)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    res = measure.validate(img, cam, [b[2] for b in bars])
    assert res["max_abs_error_mm"] < 0.15, res


def test_sensor_profile(sample, tmp_path):
    out, truth = sample
    frames = [cv2.imread(str(p)) for p in sorted((out / "sensor").glob("frame_*.png"))]
    white, empty = (
        cv2.imread(str(out / "sensor" / "white.png")),
        cv2.imread(str(out / "sensor" / "empty.png")),
    )
    profile, ff, plate = sensor.build_profile(frames, white, empty, None, None, out_dir=tmp_path)
    for ch in "rgb":
        assert profile["noise"][ch]["a"] == pytest.approx(truth["noise"]["a"], rel=0.15)
        assert profile["noise"][ch]["b"] == pytest.approx(truth["noise"]["b"] + 1 / 12, abs=0.8)
    h, w = ff.shape
    yy, xx = np.mgrid[0:h, 0:w]
    expected = 1 - 0.25 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w / 2) ** 2
    assert np.abs(ff - expected / expected.max()).max() < 0.02
    assert (tmp_path / "flatfield.png").is_file() and (tmp_path / "plate_empty.png").is_file()
    assert profile["plate_empty"]["mean_rgb"][1] > profile["plate_empty"]["mean_rgb"][0]


def test_recheck_detects_drift(sample):
    out, truth = sample
    cam = Camera(
        np.asarray(truth["camera_matrix"]),
        np.asarray(truth["dist_coeffs"]),
        tuple(truth["image_size"]),
    )
    img = cv2.imread(str(out / "extrinsic" / "board_flat.png"))
    calib = {
        "camera_matrix": cam.K.tolist(),
        "dist_coeffs": cam.dist.tolist(),
        "image_size": list(cam.image_size),
        "extrinsic": extrinsic.solve(img, BoardSpec(), cam),
    }
    same = measure.recheck(img, BoardSpec(), calib)
    assert not same["recalibrate_extrinsic"] and same["plane_tilt_change_deg"] < 0.01
    rng = np.random.default_rng(3)
    higher = render_board(cam, BoardSpec(), pose_matrix(0, 0, 3, (-60, -50, 301.5)), rng)
    r = measure.recheck(higher, BoardSpec(), calib)
    assert r["recalibrate_extrinsic"]
    assert r["camera_height_change_mm"] == pytest.approx(1.5, abs=0.3)
    tilted = render_board(cam, BoardSpec(), pose_matrix(0.6, 0, 3, (-60, -50, 300)), rng)
    r = measure.recheck(tilted, BoardSpec(), calib)
    assert r["recalibrate_extrinsic"] and r["plane_tilt_change_deg"] == pytest.approx(0.6, abs=0.15)
    shifted = render_board(cam, BoardSpec(), pose_matrix(0, 0, 20, (-40, -30, 300)), rng)
    assert not measure.recheck(shifted, BoardSpec(), calib)[
        "recalibrate_extrinsic"
    ]  # moved board, same plane


def test_board_pdf(tmp_path):
    write_board_pdf(BoardSpec(), tmp_path / "b.pdf", dpi=200)
    assert (tmp_path / "b.pdf").read_bytes()[:4] == b"%PDF"


def test_board_spec_measured_square():
    s = BoardSpec().with_square(14.9)
    assert s.square_mm == 14.9 and s.marker_mm == pytest.approx(11 * 14.9 / 15)


def test_cli_end_to_end(sample, tmp_path, capsys):
    out, _ = sample
    cd = ["--calib-dir", str(tmp_path)]
    assert main(["board", "--dpi", "150", *cd]) == 0
    assert main(["intrinsics", "--images", str(out / "intrinsics"), *cd]) == 0
    assert (
        main(
            [
                "extrinsic",
                "--image",
                str(out / "extrinsic" / "board_flat.png"),
                "--area-origin",
                "0",
                "-105",
                "--area-size",
                "135",
                "105",
                *cd,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "sensor",
                "--frames",
                str(out / "sensor" / "frame_*.png"),
                "--white",
                str(out / "sensor" / "white.png"),
                "--empty",
                str(out / "sensor" / "empty.png"),
                *cd,
            ]
        )
        == 0
    )
    assert main(["recheck", "--image", str(out / "extrinsic" / "board_flat.png"), *cd]) == 0
    assert (
        main(["report", "--extrinsic-image", str(out / "extrinsic" / "board_flat.png"), *cd]) == 0
    )
    calib = read_json(tmp_path / "calibration.json")
    for key in (
        "camera_matrix",
        "dist_coeffs",
        "image_size",
        "rms_px",
        "extrinsic",
        "roi",
        "px_per_mm",
        "plate_homography",
        "board",
    ):
        assert key in calib
    assert np.asarray(calib["extrinsic"]["T_cam_plate"]).shape == (4, 4)
    for f in (
        "sensor_profile.json",
        "flatfield.png",
        "plate_empty.png",
        "report.html",
        "charuco_board.pdf",
    ):
        assert (tmp_path / f).is_file()
    pe = cv2.imread(str(tmp_path / "plate_empty.png"))
    assert pe.shape[:2] == (calib["roi"]["h"], calib["roi"]["w"])


# ------------------------------------------------------------------------- Pi helper scripts


@pytest.fixture(scope="module")
def pi_common():
    path = repo_root() / "tools" / "calibration" / "pi" / "pi_common.py"
    spec = importlib.util.spec_from_file_location("pi_common", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pi_common"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pi_focus(pi_common):
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (200, 300)).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2)
    assert pi_common.focus_metric(sharp) > pi_common.focus_metric(blurred)
    pos = [2.8, 2.9, 3.0, 3.1, 3.2, 3.3]
    scores = [-((p - 3.07) ** 2) for p in pos]
    assert pi_common.best_focus(pos, scores) == pytest.approx(3.07, abs=1e-6)


def test_pi_settings_check(pi_common):
    locked = {
        "ExposureTime": 10000,
        "AnalogueGain": 1.5,
        "ColourGains": [2.0, 1.6],
        "LensPosition": 3.3,
        "size": [4608, 2592],
    }
    good = {
        "ExposureTime": 10020,
        "AnalogueGain": 1.51,
        "ColourGains": (2.0, 1.605),
        "LensPosition": 3.3,
        "size": [4608, 2592],
    }
    assert pi_common.settings_mismatch(locked, good) == []
    bad = dict(good, ExposureTime=12000, ColourGains=(2.3, 1.6))
    problems = pi_common.settings_mismatch(locked, bad)
    assert len(problems) == 2 and any("ExposureTime" in p for p in problems)
    c = pi_common.controls_from_settings(locked)
    assert c["AeEnable"] is False and c["AwbEnable"] is False and c["AfMode"] == 0


def test_pi_coverage_and_save(pi_common, tmp_path):
    cov = pi_common.Coverage((4608, 2592), (16, 9))
    cov.add(np.array([[10.0, 10.0], [4600, 2590], [2300, 1300]]))
    assert cov.fraction == pytest.approx(3 / 144)
    assert cov.ascii().count("\n") == 8
    p = pi_common.save_capture(
        tmp_path, "x", np.zeros((4, 4, 3), np.uint8), {"ExposureTime": 1, "ColourGains": (1.0, 2.0)}
    )
    assert p.is_file() and read_json(p.with_suffix(".json"))["metadata"]["ColourGains"] == [
        1.0,
        2.0,
    ]


def test_intrinsic_views_cover_tilts():
    tilts = [math.degrees(math.acos(abs(T[2, 2]))) for T in intrinsic_views(14)]
    assert max(tilts) <= 22
