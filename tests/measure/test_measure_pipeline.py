import shutil

import pytest

from fake_scenes import make_run
from screwsynth.common.config import load_config


@pytest.fixture
def fake_dataset(tmp_path):
    """Processed fake scenes exported by tool 07 with crops (measurable DIN912 screws)."""
    from screwsynth.labels import export as ex

    make_run(tmp_path / "out", "fake", n=6)
    scenes = sorted((tmp_path / "out" / "fake" / "processed").iterdir())
    names = {i: f"C{i}" for i in range(4)}
    st = ex.Settings.from_config(load_config().get("labels", {}), 0)
    ds = tmp_path / "ds"
    ex.export(scenes, ds, names, ["seg", "crops"], st)
    return ds


def test_compare_measurement_runs_on_a_dataset(fake_dataset):
    from screwsynth.measure.evaluate import compare_measurement, comparison_markdown
    from screwsynth.measure.geometry import Geometry

    res = compare_measurement(fake_dataset, Geometry.from_config(load_config()), split="all")
    assert res["screws"] > 0
    assert set(res["methods"]) == {"A_true", "B"}
    assert res["methods"]["A_true"]["n"] == res["screws"]
    md = comparison_markdown(res)
    assert "| Method |" in md and "classical" in md


def test_measure_screw_on_a_true_mask(fake_dataset):
    from screwsynth.measure.evaluate import _gt_detection, _gt_mask_window, scene_from_dir
    from screwsynth.measure.geometry import Geometry
    from screwsynth.measure.infer import Settings, measure_screw, plate_pose

    geo = Geometry.from_config(load_config())
    crops = (fake_dataset / "crops.csv").read_text(encoding="utf-8").splitlines()
    scene_dir = crops[1].split(",")[2]
    scene, meta, inst = scene_from_dir(__import__("pathlib").Path(scene_dir))
    o = next(o for o in meta["objects"] if o["kind"] == "screw" and o["class_name"] in geo)
    det = _gt_detection(o, inst)
    res = measure_screw(scene, det, geo[o["class_name"]], Settings(), mask=_gt_mask_window(o, inst))
    assert res["length"]["method"] == "classical" and res["length_classical_mm"] > 0
    assert res["keypoints"]["classical"] is not None and "thread_diameter_mm" in res
    pos, angle = plate_pose(scene, det, 2.0)
    assert len(pos) == 2 and 0 <= angle < 180


@pytest.mark.slow
def test_inference_end_to_end_with_dummy_model(tmp_path, fake_dataset):
    """A tiny untrained stage-1 model (built from its yaml, no download) through the pipeline."""
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    from screwsynth.measure.evaluate import scene_from_dir
    from screwsynth.measure.geometry import Geometry
    from screwsynth.measure.infer import Pipeline, Settings, draw

    model = YOLO("yolo26n-seg.yaml")
    weights = tmp_path / "dummy.pt"
    model.save(str(weights))
    pipe = Pipeline(weights, None, Geometry.from_config(load_config()), Settings(conf=0.01), None)
    scene_dir = next((tmp_path / "out" / "fake" / "processed").iterdir())
    scene, _, _ = scene_from_dir(scene_dir)
    result, dets = pipe.run_scene(scene, "fake")
    assert isinstance(result["objects"], list) and result["timing_s"]["total"] > 0
    assert draw(scene, result).ndim == 3
    shutil.rmtree(tmp_path / "out", ignore_errors=True)
