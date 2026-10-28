import math

import pytest

from screwsynth.common.units import (
    DEFAULT_WORLD_SCALE,
    bu_to_mm,
    bu_to_mm_vec,
    deg2rad,
    mm_to_bu,
    mm_to_bu_vec,
    rad2deg,
)


def test_default_scale():
    assert DEFAULT_WORLD_SCALE == 10
    # 100 mm = 0.1 m, scaled x10 -> 1 Blender unit
    assert mm_to_bu(100) == pytest.approx(1.0)
    assert bu_to_mm(1.0) == pytest.approx(100.0)


@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 2.5])
@pytest.mark.parametrize("mm", [0.0, 0.4, 2.5, 16.0, 220.0, -3.0])
def test_round_trip(mm, scale):
    assert bu_to_mm(mm_to_bu(mm, scale), scale) == pytest.approx(mm, abs=1e-12)


def test_scaling_is_linear():
    assert mm_to_bu(16, 1) == pytest.approx(0.016)
    assert mm_to_bu(16, 10) == pytest.approx(0.16)
    assert mm_to_bu(16, 100) == pytest.approx(1.6)


def test_vectors():
    assert mm_to_bu_vec([0, 100, 220]) == pytest.approx([0.0, 1.0, 2.2])
    assert bu_to_mm_vec((0.0, 1.0, 2.2)) == pytest.approx([0.0, 100.0, 220.0])


@pytest.mark.parametrize("scale", [0, -10])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        mm_to_bu(1.0, scale)
    with pytest.raises(ValueError):
        bu_to_mm(1.0, scale)


def test_angles():
    assert deg2rad(180) == pytest.approx(math.pi)
    assert rad2deg(math.pi / 2) == pytest.approx(90)
    assert rad2deg(deg2rad(15)) == pytest.approx(15)
