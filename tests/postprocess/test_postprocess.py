import cv2
import numpy as np
import pytest

from screwsynth.calibration.geometry import Camera
from screwsynth.common.io import read_json, write_json
from screwsynth.common.png import write_png
from screwsynth.postprocess import compare
from screwsynth.postprocess import effects as fx
from screwsynth.postprocess.cli import main
from screwsynth.postprocess.process import SensorData, process_scene

DIST = np.array([-0.12, 0.05, 0.0005, -0.0003, 0.0])
FULL_K = np.array([[1100.0, 0, 640.0], [0, 1100.0, 480.0], [0, 0, 1]])
ROI = {"x0": 200, "y0": 100, "w": 800, "h": 760}


def render_K():
    K = FULL_K.copy()
    K[0, 2] -= ROI["x0"]
    K[1, 2] -= ROI["y0"]
    return K


def test_straight_line_becomes_calibrated_curve():
    """A straight plate line rendered undistorted, then distorted, follows the OpenCV projection."""
    K = render_K()
    cam = Camera(FULL_K, DIST, (1280, 960))
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1, -1])
    T[:3, 3] = -T[:3, :3] @ np.array([0.0, 0.0, 300.0])
    cam.T_cam_plate = T
    line = np.array([[x, 20.0, 0.0] for x in np.linspace(-100, 100, 400)])
    und = Camera(FULL_K, np.zeros(5), (1280, 960), T).project(line) - [ROI["x0"], ROI["y0"]]
    img = np.zeros((ROI["h"], ROI["w"]), np.uint8)
    pts = np.round(und * 16).astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, 255, 3, cv2.LINE_AA, shift=4)
    dimg = fx.distort_image(img, K, DIST)
    expected = cam.project(line) - [ROI["x0"], ROI["y0"]]
    # the line's centroid in each column of the distorted image matches the calibrated projection
    errs = []
    for u, v in expected[50:-50]:
        col = dimg[:, int(round(u))].astype(float)
        if col.sum() > 0:
            errs.append(abs((col * np.arange(len(col))).sum() / col.sum() - v))
    assert np.median(errs) < 0.2 and max(errs) < 0.6
    # and the points function agrees with OpenCV projectPoints exactly
    assert np.allclose(fx.distort_points(und, K, DIST), expected, atol=1e-3)


def test_mask_and_rgb_stay_aligned():
    K = render_K()
    rgb = np.full((ROI["h"], ROI["w"], 3), 60, np.uint8)
    mask = np.zeros((ROI["h"], ROI["w"]), np.uint16)
    for k, (x, y) in enumerate([(30, 40), (600, 50), (350, 380), (40, 680), (700, 690)], start=1):
        rgb[y : y + 50, x : x + 70] = 220
        mask[y : y + 50, x : x + 70] = k
    drgb = fx.distort_image(rgb, K, DIST)
    dmask = fx.distort_mask(mask, K, DIST)
    assert set(np.unique(dmask)) <= set(np.unique(mask))  # ids never blend
    bright = drgb[..., 0] > 140
    fg = dmask > 0
    # object edges coincide within 1 px
    dist_to_fg = cv2.distanceTransform((~fg).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_bright = cv2.distanceTransform((~bright).astype(np.uint8), cv2.DIST_L2, 3)
    assert dist_to_fg[bright].max() <= 1.0 + 1e-6
    assert dist_to_bright[fg].max() <= 1.0 + 1e-6


def test_keypoint_stays_on_feature():
    K = render_K()
    img = np.zeros((ROI["h"], ROI["w"]), np.float32)
    kps = np.array([[25.0, 30.0], [400.0, 380.0], [770.0, 735.0], [120.5, 600.25]])
    for u, v in kps:
        cv2.circle(
            img, (int(round(u * 16)), int(round(v * 16))), 3 * 16, 255, -1, cv2.LINE_AA, shift=4
        )
    d = fx.distort_image(img, K, DIST)
    for p in fx.distort_points(kps, K, DIST):
        x, y = int(round(p[0])), int(round(p[1]))
        patch = d[y - 6 : y + 7, x - 6 : x + 7]
        yy, xx = np.mgrid[y - 6 : y + 7, x - 6 : x + 7]
        c = np.array([(patch * xx).sum(), (patch * yy).sum()]) / patch.sum()
        assert np.linalg.norm(c - p) < 0.3


def test_no_distortion_is_identity():
    img = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    assert np.array_equal(fx.distort_image(img, render_K(), np.zeros(5)), img)


def test_noise_statistics():
    rng = np.random.default_rng(0)
    img = np.full((300, 300, 3), 100.0)
    model = {ch: {"a": 0.5, "b": 2.0} for ch in "rgb"}
    noisy = fx.apply_noise(img, model, 1.0, rng)
    assert noisy[..., 1].var() == pytest.approx(0.5 * 100 + 2.0, rel=0.05)


def test_flatfield_crop_and_vignetting():
    full = np.ones((960, 1280), np.float32)
    full[:, :640] = 0.5
    ff = fx.flatfield_for_render(full, ROI, (400, 380))
    assert ff.shape == (380, 400) and ff[:, :100].mean() == pytest.approx(0.5, abs=0.02)
    v = fx.apply_vignetting(np.full((380, 400, 3), 200.0), ff, 1.0)
    assert v[:, :100].mean() == pytest.approx(100, abs=3) and v[:, -50:].mean() == pytest.approx(
        200, abs=1
    )
    assert fx.radial_flatfield((100, 100), 0.25)[50, 50] == pytest.approx(1.0, abs=1e-3)


def _scene(tmp_path, dist=DIST):
    d = tmp_path / "scenes" / "000000"
    d.mkdir(parents=True)
    rgb = np.full((ROI["h"], ROI["w"], 3), (40, 90, 160), np.uint8)
    mask = np.zeros((ROI["h"], ROI["w"]), np.uint16)
    rgb[300:340, 200:500] = 200
    mask[300:340, 200:500] = 1
    write_png(d / "rgb.png", rgb)
    write_png(d / "instance.png", mask)
    kp = lambda u, v, vis: {"xyz": [0, 0, 0], "uv": [u, v], "visibility": vis}  # noqa: E731
    write_json(
        d / "meta.json",
        {
            "scene_id": "000000",
            "layout_id": "000000",
            "image": "rgb.png",
            "camera": {
                "K": render_K().tolist(),
                "dist_coeffs": list(dist),
                "roi": ROI,
                "distorted": False,
            },
            "objects": [
                {
                    "instance_id": 1,
                    "alone_area_px": 12000,
                    "mask_area_px": 12000,
                    "visible_pixel_fraction": 1.0,
                    "keypoints": {
                        "head_top": kp(495, 320, 2),
                        "bearing": kp(490, 320, 2),
                        "tip": kp(799.6, 759.5, 2),
                    },
                }
            ],
        },
    )
    return d


def test_process_scene(tmp_path):
    d = _scene(tmp_path)
    cfg = {"compression": {"jpeg_probability": 1.0, "quality": [90, 90]}}
    out = tmp_path / "processed" / "000000"
    sensor = SensorData(noise={ch: {"a": 0.4, "b": 1.5} for ch in "rgb"}, plate_rgb=[50, 100, 150])
    params = process_scene(d, out, cfg, sensor, seed=3)
    meta = read_json(out / "meta.json")
    assert meta["image"] == "image.jpg" and (out / "image.jpg").is_file()
    assert meta["camera"]["distorted"] is True
    assert set(meta["postprocess"]) >= {
        "distortion",
        "vignetting",
        "blur",
        "colour",
        "noise",
        "compression",
    }
    kp = meta["objects"][0]["keypoints"]
    assert kp["bearing"]["uv_undistorted"] == [490, 320]
    assert np.allclose(
        kp["bearing"]["uv"],
        fx.distort_points(np.array([[490, 320.0]]), render_K(), DIST)[0],
        atol=1e-3,
    )
    # visibility becomes 0 exactly when the distorted keypoint leaves the frame
    for k in kp.values():
        u, v = k["uv"]
        inside = 0 <= u <= ROI["w"] - 1 and 0 <= v <= ROI["h"] - 1
        assert (k["visibility"] == 0) == (not inside)
    mask = cv2.imread(str(out / "instance.png"), cv2.IMREAD_UNCHANGED)
    assert mask.dtype == np.uint16 and meta["objects"][0]["mask_area_px"] == int((mask == 1).sum())
    assert (d / "rgb.png").is_file() and read_json(d / "meta.json")[
        "image"
    ] == "rgb.png"  # raw untouched
    # deterministic per seed
    process_scene(d, tmp_path / "again", cfg, sensor, seed=3)
    assert read_json(tmp_path / "again" / "meta.json")["postprocess"] == params


def test_compare_report(tmp_path):
    rng = np.random.default_rng(0)
    syn = []
    for _ in range(3):
        img = np.full((200, 240, 3), (40, 90, 160), np.uint8)
        img[80:120, 60:180] = 210
        m = np.zeros((200, 240), np.uint16)
        m[80:120, 60:180] = 1
        syn.append((img, m))
    real = []
    for _ in range(3):
        img = np.full((200, 240, 3), (45, 95, 150), np.float32)
        img[70:115, 50:170] = 205
        img = cv2.GaussianBlur(img, (0, 0), 1.5) + rng.normal(0, 2, img.shape)
        real.append(np.clip(img, 0, 255).astype(np.uint8))
    summary = compare.report(syn, real, tmp_path / "r.html")
    assert (tmp_path / "r.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert summary["synthetic"]["edge_width_px"] < summary["real"]["edge_width_px"]
    assert any("too sharp" in s for s in summary["suggestions"])


def test_cli_run_show_compare(tmp_path, capsys):
    run = tmp_path / "output" / "r1"
    _scene(run)
    real = tmp_path / "real"
    real.mkdir()
    cv2.imwrite(str(real / "a.png"), np.full((200, 200, 3), 120, np.uint8))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"seed: 1\npaths: {{output_dir: {(tmp_path / 'output').as_posix()}, "
        f"calibration_dir: {(tmp_path / 'calib').as_posix()}}}\npostprocess: {{workers: 1}}\n",
        encoding="utf-8",
    )
    assert main(["run", "--run", "r1", "--config", str(cfg)]) == 0
    assert main(["run", "--run", "r1", "--resume", "--config", str(cfg)]) == 0
    assert "skipped: 1" in capsys.readouterr().out
    assert main(["show", "--run", "r1", "--scene", "000000", "--config", str(cfg)]) == 0
    assert (run / "show_000000.png").is_file()
    assert main(["compare", "--run", "r1", "--real", str(real), "--config", str(cfg)]) == 0
    assert (run / "compare.html").is_file()


@pytest.mark.parametrize(
    "kind,matched",
    [("calibrated_photo", True), ("images", False), ("colours", False), (None, True)],
)
def test_plate_colour_match_only_on_the_real_plate(tmp_path, kind, matched):
    """A craft mat or a solid colour keeps its own colour; only the real empty plate is matched."""
    d = _scene(tmp_path)
    if kind is not None:
        meta = read_json(d / "meta.json")
        meta["lighting"] = {"background": {"kind": kind}}
        write_json(d / "meta.json", meta)
    cfg = {"colour": {"probability": 1.0, "gain": [1, 1], "wb": [1, 1], "gamma": [1, 1]}}
    sensor = SensorData(plate_rgb=[50, 100, 150])
    params = process_scene(d, tmp_path / "out", cfg, sensor, seed=1)
    plate_match = params["colour"]["plate_match"]
    assert (plate_match != [1.0, 1.0, 1.0]) == matched
