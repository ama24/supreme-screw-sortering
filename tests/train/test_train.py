import json
import shutil

import pytest

from fake_scenes import make_run
from screwsynth.common.config import load_config
from screwsynth.common.io import read_yaml, write_yaml
from screwsynth.train import cli
from screwsynth.train.experiments import DEFAULT_AUGMENT, Experiment
from screwsynth.train.trainer import _layout_of, subset_yaml


def test_experiment_args(tmp_path):
    exp = Experiment.load("stage1_nano", dataset=tmp_path)
    assert exp.stage == "seg" and exp.model.startswith("yolo26n-seg") and exp.imgsz == 1024
    args = exp.ultralytics_args(tmp_path / "runs")
    assert args["data"] == str(tmp_path / "seg.yaml")
    assert args["degrees"] == 180 and args["scale"] == 0.1 and args["perspective"] == 0.0
    assert args["shear"] == 0.0 and args["batch"] == -1 and args["name"] == "stage1_nano"
    pose = Experiment.load("stage2_pose", dataset=tmp_path)
    assert pose.data_yaml.name == "crops.yaml" and pose.imgsz == 384
    assert pose.ultralytics_args(tmp_path)["mosaic"] == 0.0


def test_experiment_overrides_and_validation(tmp_path):
    e = Experiment.from_dict(
        {"stage": "seg", "dataset": str(tmp_path), "train": {"lr0": 0.002, "degrees": 90}}
    )
    a = e.ultralytics_args(tmp_path)
    assert a["lr0"] == 0.002 and a["degrees"] == 90  # `train:` wins over the augment defaults
    assert set(DEFAULT_AUGMENT) <= set(a)
    with pytest.raises(ValueError):
        Experiment.from_dict({"stage": "detect", "dataset": "x"})
    with pytest.raises(ValueError):
        Experiment.from_dict({"stage": "seg"})  # no dataset


def test_layout_of():
    assert _layout_of("run01_000012") == "run01_000012"
    assert _layout_of("run01_000012_r2") == "run01_000012"
    assert _layout_of("run01_000012_r2_1") == "run01_000012"
    assert _layout_of("dataset_v1_000003_0") == "dataset_v1_000003"


def test_subset_keeps_layouts_together(tmp_path):
    root = tmp_path / "seg"
    tr = root / "images" / "train"
    tr.mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    for k in range(20):
        for suffix in ("", "_r1", "_r2"):
            (tr / f"run_{k:06d}{suffix}.jpg").write_bytes(b"")
    data = tmp_path / "seg.yaml"
    write_yaml(
        data,
        {"path": root.as_posix(), "train": "images/train", "val": "images/val", "names": {0: "a"}},
    )
    sub = subset_yaml(data, 0.25, tmp_path / "subsets", seed=1)
    lines = (tmp_path / "subsets" / "train_25.txt").read_text(encoding="utf-8").splitlines()
    layouts = {_layout_of(line.rsplit("/", 1)[-1].rsplit(".", 1)[0]) for line in lines}
    assert len(layouts) == 5 and len(lines) == 15  # 5 of 20 layouts, each with all 3 photos
    cfg = read_yaml(sub)
    assert cfg["train"].endswith("train_25.txt") and cfg["val"].endswith("images/val")
    again = subset_yaml(data, 0.25, tmp_path / "again", seed=1)
    assert read_yaml(again)["train"] != cfg["train"]  # different folder, same choice:
    assert (tmp_path / "again" / "train_25.txt").read_text(encoding="utf-8").splitlines() == lines


def test_cli_dry_run(tmp_path, capsys):
    assert cli.main(["run", "stage2_pose", "--dataset", str(tmp_path), "--dry-run"]) == 0
    args = json.loads(capsys.readouterr().out)
    assert args["imgsz"] == 384 and args["data"].endswith("crops.yaml")


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
    from screwsynth.train.evaluate import compare_measurement, comparison_markdown
    from screwsynth.train.geometry import Geometry

    res = compare_measurement(fake_dataset, Geometry.from_config(load_config()), split="all")
    assert res["screws"] > 0
    assert set(res["methods"]) == {"A_true", "B"}
    assert res["methods"]["A_true"]["n"] == res["screws"]
    md = comparison_markdown(res)
    assert "| Method |" in md and "classical" in md


def test_measure_screw_on_a_true_mask(fake_dataset):
    from screwsynth.train.evaluate import _gt_detection, _gt_mask_window, scene_from_dir
    from screwsynth.train.geometry import Geometry
    from screwsynth.train.infer import Settings, measure_screw, plate_pose

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

    from screwsynth.train.evaluate import scene_from_dir
    from screwsynth.train.geometry import Geometry
    from screwsynth.train.infer import Pipeline, Settings, draw

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
