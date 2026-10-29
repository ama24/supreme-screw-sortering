import math

import cv2
import numpy as np
import pytest

from screwsynth.calibration.geometry import nominal_camera
from screwsynth.train.classical import edge_keypoints, rim_corrected, to_gray


def _screw_image(angle_deg=27.0, head=(14.0, 22.0), shank=(60.0, 11.0), ss=8, size=(160, 140)):
    """Grey image of a lying screw seen from above (head + shank rectangles), anti-aliased by
    supersampling, plus its true head_top / bearing / tip pixels."""
    w, h = size
    big = np.full((h * ss, w * ss), 0.35, np.float32)  # plate
    a = math.radians(angle_deg)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-u[1], u[0]])
    (hl, hw), (sl, sw) = head, shank
    top = np.array([w / 2, h / 2]) - (hl + sl) / 2 * u  # centred, inside the image at any angle

    def rect(start, length, width):
        corners = [
            start + width / 2 * v,
            start + length * u + width / 2 * v,
            start + length * u - width / 2 * v,
            start - width / 2 * v,
        ]
        return (np.array(corners) * ss).astype(np.int32)

    cv2.fillPoly(big, [rect(top, hl, hw)], 0.8)
    cv2.fillPoly(big, [rect(top + hl * u, sl, sw)], 0.7)
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    # the supersampled pixel grid: big pixel centre (i + 0.5) / ss - 0.5 in the small image
    shift = 0.5 / ss - 0.5
    truth = {
        "head_top": top + shift,
        "bearing": top + hl * u + shift,
        "tip": top + (hl + sl) * u + shift,
    }
    mask = img > 0.5
    return img, mask, truth


@pytest.mark.parametrize("angle", [0.0, 27.0, 118.0, 200.0])
def test_edge_keypoints_subpixel(angle):
    img, mask, truth = _screw_image(angle)
    ek = edge_keypoints(img, mask)
    assert ek is not None
    for k in ("head_top", "bearing", "tip"):
        err = np.linalg.norm(ek.keypoints[k] - truth[k])
        assert err < 0.4, (k, err)  # sub-pixel
    assert ek.head_width_px > ek.shank_widths_px[0]


def test_edge_keypoints_head_found_at_either_end():
    img, mask, truth = _screw_image(10.0)
    flipped = img[:, ::-1].copy()
    ek = edge_keypoints(flipped, flipped > 0.5)
    w = img.shape[1]
    mirrored_tip = np.array([w - 1 - truth["tip"][0], truth["tip"][1]])
    assert np.linalg.norm(ek.keypoints["tip"] - mirrored_tip) < 0.4


@pytest.mark.parametrize("outward_sign", [1, -1])
def test_rim_correction_recovers_axis_point(outward_sign):
    """The silhouette extremity is a rim of the end face; the axis point behind it is found."""
    cam = nominal_camera()
    axis_pt = np.array([190.0, 60.0, 2.0])  # far from the nadir (110, 110): strong parallax
    r = 1.5
    nadir = cam.centre_in_plate[:2]
    radial = (axis_pt[:2] - nadir) / np.linalg.norm(axis_pt[:2] - nadir)
    outward = outward_sign * radial  # this end points away from (or towards) the nadir
    z_rim = axis_pt[2] + r if outward_sign > 0 else axis_pt[2] - r
    rim_px = cam.project(np.array([[axis_pt[0], axis_pt[1], z_rim]]))[0]
    true_px = cam.project(np.array([axis_pt]))[0]
    got = rim_corrected(cam, rim_px, axis_pt[2], r, outward)
    assert np.linalg.norm(got - true_px) < 0.02
    assert np.linalg.norm(rim_px - true_px) > 0.5  # the uncorrected extremity is off


def test_to_gray():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 1] = 255
    g = to_gray(rgb)
    assert g.dtype == np.float32 and g.max() == pytest.approx(1 / 3)


def test_rim_correction_only_along_the_axis():
    """A screw lying across the radial direction: the rim shift is sideways, the tip stays put."""
    cam = nominal_camera()
    axis_pt = np.array([190.0, 110.0, 2.0])  # due +x of the nadir (110, 110)
    rim_px = cam.project(np.array([[axis_pt[0], axis_pt[1], axis_pt[2] + 1.5]]))[0]
    axis_dir_px = np.array([0.0, 1.0])  # the screw lies along the image y axis (across the radial)
    got = rim_corrected(cam, rim_px, axis_pt[2], 1.5, np.array([0.0, -1.0]), axis_dir_px)
    assert abs(got[1] - rim_px[1]) < 1e-6  # no shift along the axis
    full = rim_corrected(cam, rim_px, axis_pt[2], 1.5, np.array([0.0, -1.0]))
    assert np.linalg.norm(full - rim_px) > 0.5  # the unprojected correction would move it
