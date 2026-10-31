import json

from screwsynth.common.io import write_yaml
from screwsynth.measure.modelcard import check_classes, find_card


def test_model_card_and_catalogue_checks(tmp_path):
    pkg = tmp_path / "run" / "package"
    pkg.mkdir(parents=True)
    weights = pkg / "best.pt"
    names = {0: "DIN912_M3", 1: "DIN912_M4"}
    (pkg / "model_card.json").write_text(json.dumps({"classes": ["DIN912_M3", "DIN912_M4"]}))
    classes = tmp_path / "classes.yaml"
    write_yaml(classes, {"names": {0: "DIN912_M3", 1: "DIN912_M4", 2: "DIN934_M4"}})
    errors, warnings = check_classes(weights, names, classes)
    assert errors == [] and len(warnings) == 1 and "DIN934_M4" in warnings[0]
    (pkg / "model_card.json").write_text(json.dumps({"classes": ["DIN912_M4", "DIN912_M3"]}))
    errors, _ = check_classes(weights, names, classes)
    assert errors  # swapped order: the package is mixed up
    errors, warnings = check_classes(tmp_path / "x.pt", {0: "FOO"}, classes)
    assert errors == [] and any("does not" in w for w in warnings)


def test_model_card_found_from_an_export(tmp_path):
    pkg = tmp_path / "stage1"
    export = pkg / "exports" / "ncnn_fp32" / "best_ncnn_model"
    export.mkdir(parents=True)
    (pkg / "model_card.json").write_text("{}")
    assert find_card(export) == pkg / "model_card.json"
    assert find_card(pkg / "best.pt") == pkg / "model_card.json"
