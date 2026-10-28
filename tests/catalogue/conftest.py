import copy
import shutil
from pathlib import Path

import pytest

from screwsynth.common.io import read_yaml
from screwsynth.common.paths import repo_root

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "catalogue"
MINIMAL = FIXTURES / "minimal.yaml"
SHIPPED = repo_root() / "configs" / "catalogue.yaml"


@pytest.fixture
def minimal_data():
    """A fresh, mutable copy of the minimal fixture catalogue as plain dicts."""
    return copy.deepcopy(read_yaml(MINIMAL))


@pytest.fixture
def minimal_file(tmp_path):
    dst = tmp_path / "catalogue.yaml"
    shutil.copy(MINIMAL, dst)
    return dst


@pytest.fixture
def cli_env(tmp_path, minimal_file):
    """Config + catalogue in a temp dir; returns (config path, catalogue path, classes path)."""
    classes = tmp_path / "classes.yaml"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "catalogue:\n  max_length_mm: 80\n"
        f"paths:\n  catalogue: {minimal_file.as_posix()}\n  classes: {classes.as_posix()}\n",
        encoding="utf-8",
    )
    return cfg, minimal_file, classes
