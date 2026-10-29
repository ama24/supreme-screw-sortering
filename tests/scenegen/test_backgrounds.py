import random
from collections import Counter

import pytest

from screwsynth.scenegen import backgrounds as bg


def test_colour_parsing_and_hex_round_trip():
    lin = bg.parse_colour("#3A86D4")
    assert all(0 < c < 1 for c in lin) and lin[2] > lin[1] > lin[0]  # blue
    assert bg.to_hex(lin) == "#3A86D4"
    assert bg.parse_colour([0.12, 0.35, 0.65]) == (0.12, 0.35, 0.65)  # older linear lists
    for bad in ("#12345", "blue", [1.5, 0, 0]):
        with pytest.raises(ValueError):
            bg.parse_colour(bad)


def test_colour_list_forms():
    assert bg.colour_list({}) == [{"name": "silicone_blue", "hex": "#3A86D4"}]  # the default
    lst = bg.colour_list({"colours": [{"name": "green", "hex": "#2e8b57"}]})
    assert lst == [{"name": "green", "hex": "#2E8B57"}]
    mapping = bg.colour_list({"colours": {"mat": "#336699"}})
    assert mapping == [{"name": "mat", "hex": "#336699"}]
    legacy = bg.colour_list({"solid_colours": [[0.12, 0.35, 0.65], [0.5, 0.5, 0.5]]})
    assert [c["name"] for c in legacy] == ["colour_1", "colour_2"]
    assert bg.parse_colour(legacy[1]["hex"]) == pytest.approx((0.5, 0.5, 0.5), abs=0.005)


def test_weights_drop_unavailable_sources():
    w = bg.weights({}, have_photo=False, have_images=False)
    assert w == {"calibrated_photo": 0.0, "images": 0.0, "colours": 0.2}
    w = bg.weights(
        {"background_weights": {"calibrated_photo": 1, "images": 0, "colours": 0}}, False, True
    )
    assert w["colours"] == 1.0  # nothing left: colours
    legacy = bg.weights({"solid_colour_probability": 0.25}, True, False)
    assert legacy["calibrated_photo"] == pytest.approx(0.75) and legacy["colours"] == pytest.approx(
        0.25
    )


def test_choose_respects_weights_and_is_deterministic():
    images = [
        bg.BackgroundImage("mat.jpg", 450.0, weight=3),
        bg.BackgroundImage("table.jpg", 600.0, weight=1),
    ]
    plate = {"background_weights": {"calibrated_photo": 0.5, "images": 0.3, "colours": 0.2}}
    rng = random.Random(0)
    picks = [bg.choose(plate, images, True, rng) for _ in range(4000)]
    kinds = Counter(p["kind"] for p in picks)
    assert 0.45 < kinds["calibrated_photo"] / 4000 < 0.55 and 0.25 < kinds["images"] / 4000 < 0.35
    files = Counter(p["file"] for p in picks if p["kind"] == "images")
    assert 2.4 < files["mat.jpg"] / files["table.jpg"] < 3.8
    im = next(p for p in picks if p["kind"] == "images")
    assert 0 <= im["rotation_deg"] < 360 and all(0 <= o < 1 for o in im["offset"])
    col = next(p for p in picks if p["kind"] == "colours")
    assert col["name"] == "silicone_blue" and col["hex"].startswith("#")
    a = [bg.choose(plate, images, True, random.Random(5)) for _ in range(3)]
    b = [bg.choose(plate, images, True, random.Random(5)) for _ in range(3)]
    assert a == b


def test_library_round_trip_and_usable(tmp_path):
    (tmp_path / "mat.jpg").write_bytes(b"x")
    items = [
        bg.BackgroundImage("mat.jpg", 450.0),
        bg.BackgroundImage("gone.jpg", 300.0),  # file missing
        bg.BackgroundImage("mat.jpg", 450.0, enabled=False),
    ]
    bg.save_library(tmp_path, items)
    back = bg.load_library(tmp_path)
    assert back == items
    assert [i.file for i in bg.usable_images(tmp_path, back)] == ["mat.jpg"]
    assert bg.load_library(tmp_path / "nothing") == []


def test_width_from_ruler():
    # a 3000 px wide photo; the 0 and 200 mm marks are 1200 px apart -> 500 mm wide
    assert bg.width_from_ruler(3000, (100, 50), (1300, 50), 200) == pytest.approx(500.0)
    assert bg.width_from_ruler(3000, (0, 0), (600, 800), 100) == pytest.approx(300.0)  # diagonal
    with pytest.raises(ValueError):
        bg.width_from_ruler(3000, (5, 5), (5, 5), 100)
