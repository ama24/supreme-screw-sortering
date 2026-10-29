import csv

import cv2
import numpy as np
import pytest

from screwsynth.common.io import read_json, read_yaml, write_json, write_yaml
from screwsynth.common.paths import repo_root
from screwsynth.realdata import cli, cvat, matching
from screwsynth.realdata.build import roi_camera
from screwsynth.realdata.session import (
    InventoryRow,
    SessionError,
    new_session,
    read_inventory,
)

# ------------------------------------------------------------------------------- sessions


def _write_inventory(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("# comment line\n")
        w = csv.writer(f)
        w.writerow(["part_id", "finish", "length_mm", "count", "notes"])
        for r in rows:
            w.writerow(r)


def test_session_and_inventory(tmp_path):
    d = new_session(tmp_path, "s1", operator="me", notes="first")
    assert (d / "images").is_dir() and (d / "inventory.csv").is_file()
    assert read_json(d / "session.json")["operator"] == "me"
    with pytest.raises(SessionError):
        new_session(tmp_path, "s1")  # exists
    with pytest.raises(SessionError):
        new_session(tmp_path, "bad name/")
    _write_inventory(
        d / "inventory.csv",
        [
            ["DIN912_M4x16", "black_oxide", "15.9", "3", ""],
            ["DIN912_M4x20", "black_oxide", "", "", ""],
            ["DIN934_M4", "zinc_clear", "", "5", ""],
            ["UNKNOWN", "", "", "2", "wood screws"],
        ],
    )
    rows = read_inventory(d / "inventory.csv", ["black_oxide", "zinc_clear"])
    assert [r.class_name for r in rows] == ["DIN912_M4", "DIN912_M4", "DIN934_M4", "UNKNOWN"]
    assert rows[0].length_mm == 15.9 and rows[0].count == 3
    assert rows[1].length_mm == 20.0 and rows[1].count == 1  # nominal length when no caliper value


@pytest.mark.parametrize(
    "bad",
    [
        ["DIN912_M4", "black_oxide", "x", "1", ""],
        ["NOT_A_PART", "", "", "1", ""],
        ["DIN934_M4", "gold", "", "1", ""],
    ],
)
def test_inventory_errors(tmp_path, bad):
    p = tmp_path / "inv.csv"
    _write_inventory(p, [bad])
    with pytest.raises(SessionError):
        read_inventory(p, ["black_oxide"])


# ----------------------------------------------------------------------------------- CVAT


def _anns():
    screw = cvat.Obj(
        "DIN912_M4",
        [(10, 10), (60, 10), (60, 20), (10, 20)],
        {"head_top": (12.0, 15.0), "bearing": (20.5, 15.0), "tip": (58.25, 15.0)},
        0.9,
        1,
    )
    nut = cvat.Obj("DIN934_M4", [(100, 100), (120, 100), (120, 120), (100, 120)], {}, None, 2)
    return [
        cvat.ImageAnn("img_0001.png", 200, 150, [screw, nut]),
        cvat.ImageAnn("img_0002.png", 200, 150, []),
    ]


def test_cvat_round_trip(tmp_path):
    anns = _anns()
    path = tmp_path / "cvat.xml"
    cvat.export_cvat(anns, ["DIN912_M4", "DIN934_M4"], path)
    back = cvat.import_cvat(path)
    assert [a.name for a in back] == ["img_0001.png", "img_0002.png"]
    s, n = back[0].objects
    assert s.label == "DIN912_M4" and n.label == "DIN934_M4" and not n.keypoints
    assert np.allclose(s.polygon, anns[0].objects[0].polygon)
    assert s.keypoints == anns[0].objects[0].keypoints  # 2-decimal values survive exactly
    assert cvat.from_json(cvat.to_json(back))[0].objects[0].keypoints == s.keypoints
    labels = cvat.cvat_labels(["DIN912_M4"])
    assert {"name": "tip", "type": "points", "attributes": []} in labels


def test_cvat_points_without_group_attach_by_position(tmp_path):
    xml = tmp_path / "hand.xml"
    xml.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations><version>1.1</version>
<image id="0" name="a.png" width="100" height="100">
  <polygon label="DIN912_M3" occluded="0" points="10,10;50,10;50,20;10,20" z_order="0"/>
  <points label="tip" occluded="0" points="48,15" z_order="0"/>
  <points label="head_top" occluded="0" points="80,80" z_order="0"/>
</image></annotations>""",
        encoding="utf-8",
    )
    (img,) = cvat.import_cvat(xml)
    assert img.objects[0].keypoints == {"tip": (48.0, 15.0)}  # the stray point outside is dropped


# ------------------------------------------------------------------------------- matching


def _inv(*rows):
    return [InventoryRow(f"{c}x{L:g}", c, "zinc_clear", L, n) for c, L, n in rows]


def test_matching_cases(tmp_path):
    inv = _inv(("DIN912_M4", 16, 2), ("DIN912_M4", 20, 1), ("DIN912_M3", 10, 3))
    ms = matching.match_image(
        "a.png",
        [
            (1, "DIN912_M4", 16.2),
            (2, "DIN912_M4", 19.7),
            (3, "DIN912_M4", 18.1),
            (4, "DIN912_M4", None),
            (5, "DIN912_M3", None),
            (6, "DIN912_M3", 13.0),
            (7, "DIN933_M5", 20.0),
        ],
        inv,
    )
    by = {m.object_id: m for m in ms}
    assert (by[1].length_mm, by[1].status) == (16, "auto")
    assert (by[2].length_mm, by[2].status) == (20, "auto")
    assert by[3].status == "ambiguous" and by[3].candidates == [16, 20]  # near the middle
    assert by[4].status == "ambiguous"  # no keypoints and two possible lengths
    assert (by[5].length_mm, by[5].status) == (10, "auto")  # one length for the class
    assert by[6].length_mm == 10  # one length: the inventory is the truth, not the measurement
    assert by[7].status == "not_in_inventory"
    over = matching.match_image("b.png", [(1, "DIN912_M4", 20.1), (2, "DIN912_M4", 19.9)], inv)
    assert sorted(m.status for m in over) == ["ambiguous", "auto"]  # only one M4x20 on the plate
    single = matching.match_image("c.png", [(i, "DIN912_M3", None) for i in range(4)], inv)
    assert {m.status for m in single} == {"over_count"}  # 4 found, 3 in the inventory


def test_manual_matches_survive(tmp_path):
    path = tmp_path / "matches.csv"
    first = matching.match_image(
        "a.png", [(1, "DIN912_M4", 18.0)], _inv(("DIN912_M4", 16, 1), ("DIN912_M4", 20, 1))
    )
    matching.write_matches(path, first)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["length_mm"], rows[0]["status"] = "20", "manual"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=matching.FIELDS)
        w.writeheader()
        w.writerows(rows)
    again = matching.write_matches(path, first, matching.read_matches(path))
    assert (again[0].length_mm, again[0].status) == (20, "manual")


# ----------------------------------------------------------------------- end to end (CLI)


def test_end_to_end_session_to_dataset(tmp_path, capsys):
    """Session -> CVAT annotations -> import -> match -> build-dataset (tool 07 formats)."""
    base = read_yaml(repo_root() / "configs" / "config.yaml")
    base["paths"].update(
        real_dir=str(tmp_path / "real"),
        datasets_dir=str(tmp_path / "datasets"),
        calibration_dir=str(tmp_path / "no_calibration"),
    )
    cfg = tmp_path / "config.yaml"
    write_yaml(cfg, base)
    c = ["--config", str(cfg)]
    assert cli.main(["session", "new", "--name", "s1", *c]) == 0
    d = tmp_path / "real" / "s1"
    cam, roi, _ = roi_camera(None)
    w, h = roi["w"], roi["h"]
    # a DIN 912 M4x16 lying on the plate (heights as when lying), drawn into the image
    a3, t3 = np.array([100.0, 100.0, 3.5]), np.array([116.0, 100.0, 1.9])
    a3[0] = t3[0] - np.sqrt(16.0**2 - (a3[2] - t3[2]) ** 2)
    head_top = a3 - [3.9, 0, 0]
    pa, pt, ph = cam.project(np.array([a3, t3, head_top]))
    img = np.full((h, w, 3), (150, 90, 40), np.uint8)
    poly = np.array([ph + [-2, -40], pt + [2, -25], pt + [2, 25], ph + [-2, 40]])
    cv2.fillPoly(img, [poly.astype(np.int32)], (200, 200, 205))
    nut = np.array([[1000, 1000], [1080, 1000], [1080, 1080], [1000, 1080]], float)
    cv2.fillPoly(img, [nut.astype(np.int32)], (180, 180, 180))
    cv2.imwrite(str(d / "images" / "img_0001.png"), img)  # already the ROI size
    _write_inventory(
        d / "inventory.csv",
        [
            ["DIN912_M4x16", "black_oxide", "16", "1", ""],
            ["DIN912_M4x20", "zinc_clear", "20", "1", ""],
            ["DIN934_M4", "zinc_clear", "", "1", ""],
        ],
    )
    assert cli.main(["export-cvat", "--session", "s1", *c]) == 0
    assert (d / "annotations" / "images" / "img_0001.png").is_file()
    # "correct in CVAT": the annotation with polygons and keypoints
    ann = cvat.ImageAnn(
        "img_0001.png",
        w,
        h,
        [
            cvat.Obj(
                "DIN912_M4",
                [tuple(p) for p in poly],
                {"head_top": tuple(ph), "bearing": tuple(pa), "tip": tuple(pt)},
                None,
                1,
            ),
            cvat.Obj("DIN934_M4", [tuple(p) for p in nut], {}, None, 2),
        ],
    )
    corrected = tmp_path / "from_cvat.xml"
    cvat.export_cvat([ann], ["DIN912_M4", "DIN934_M4"], corrected)
    assert cli.main(["import", "--session", "s1", "--xml", str(corrected), *c]) == 0
    assert cli.main(["match", "--session", "s1", *c]) == 0
    m = matching.read_matches(d / "matches.csv")[("img_0001.png", 1)]
    assert m.status == "auto" and m.length_mm == 16 and abs(m.measured_mm - 16) < 0.3
    assert cli.main(["build-dataset", "--name", "val_real", *c]) == 0
    ds = tmp_path / "datasets" / "val_real"
    stats = read_json(ds / "stats.json")
    assert stats["images"] == {"val": 1}  # a real validation set is only ever val
    crops = list(csv.DictReader((ds / "crops.csv").open(encoding="utf-8")))
    assert (
        len(crops) == 1
        and crops[0]["part_id"] == "DIN912_M4x16"
        and float(crops[0]["length_mm"]) == 16
    )
    meta = read_json(d / "scenes" / "img_0001" / "meta.json")
    assert meta["layout_id"] == "s1" and meta["camera"]["source"] == "nominal"
    finishes = {o["class_name"]: o["finish"] for o in meta["objects"]}
    assert (
        finishes["DIN934_M4"] == "zinc_clear" and finishes["DIN912_M4"] == "unknown"
    )  # two finishes
    write_json(tmp_path / "done.json", {"ok": True})
