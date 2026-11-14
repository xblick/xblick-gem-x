"""The tracking, the skeleton mapping and the depth fit; on the CPU."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xblick_gem_x.depth import fit_depth, moved, visible_depth
from xblick_gem_x.gemx import square_boxes
from xblick_gem_x.skeleton import INDEX, K4ABT_BODY_32, SOMA77, bodies_message, check_soma77, k4abt_joints
from xblick_gem_x.tracking import Tracker, iou
from xblick_gem_x.weights import STORE

WEIGHTS = STORE.root  # the node's cache directory (XBLICK_HOME moves it)

K = np.array([[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]])


def test_iou_and_square_boxes():
    a = np.array([[0, 0, 10, 10]], float)
    assert iou(a, np.array([[5, 0, 15, 10], [20, 20, 30, 30]], float)).round(3).tolist() == [[0.333, 0.0]]
    np.testing.assert_allclose(square_boxes(np.array([[10, 20, 30, 120]])), [[20, 70, 120]])


def test_tracker_keeps_ids_and_ends_tracks():
    t = Tracker(min_iou=0.3, max_missing=1)
    assert t.update(np.array([[0, 0, 10, 10], [50, 0, 60, 10]])) == [1, 2]
    assert t.update(np.array([[51, 0, 61, 10], [1, 0, 11, 10]])) == [2, 1]  # moved a little, other order
    assert t.update(np.zeros((0, 4))) == []
    assert t.update(np.zeros((0, 4))) == []  # missing twice: both ended
    assert t.update(np.array([[0, 0, 10, 10]])) == [3]  # ids are not reused


def test_k4abt_mapping_and_derived_head_joints():
    joints = np.zeros((77, 3))
    joints[INDEX["Head"]] = [0, -0.6, 2.0]
    joints[INDEX["LeftEye"]], joints[INDEX["RightEye"]] = [0.03, -0.62, 1.9], [-0.03, -0.62, 1.9]
    joints[INDEX["LeftArm"]] = [0.2, -0.4, 2.0]
    out, confidence = k4abt_joints(joints, np.ones(77, bool))
    named = dict(zip(K4ABT_BODY_32, out, strict=True))
    np.testing.assert_allclose(named["ShoulderLeft"], [0.2, -0.4, 2.0])
    assert named["EarLeft"][0] == pytest.approx(0.075) and named["EarRight"][0] == pytest.approx(-0.075)
    assert named["Nose"][2] < 1.9  # ahead of the eyes, towards the camera here
    assert confidence[K4ABT_BODY_32.index("Nose")] == 1 and confidence[K4ABT_BODY_32.index("Pelvis")] == 2
    message = json.loads(bodies_message(5, "c", 1.0, [{"id": 3, "joints": joints, "valid": np.ones(77, bool),
                                                       "depth_fit": None}]))
    assert message["model"] == "K4ABT_BODY_32" and set(message["bodies"][0]["joints"]) == set(K4ABT_BODY_32)
    assert message["bodies"][0]["joints"]["ShoulderLeft"]["position_mm"] == [200.0, -400.0, 2000.0]
    assert len(message["bodies"][0]["soma77_mm"]) == 77


@pytest.mark.skipif(not (WEIGHTS / "soma" / "SOMA_template_rig.usda").is_file(), reason="needs weights/soma")
def test_soma77_is_the_rigs():
    check_soma77(WEIGHTS / "soma" / "SOMA_template_rig.usda")
    assert SOMA77[67:72] == ("LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd")


def test_bad_rig_is_refused(tmp_path: Path):
    (tmp_path / "rig.usda").write_text('uniform token[] joints = ["Root", "Root/Hips"]')
    with pytest.raises(ValueError, match="not the SOMA"):
        check_soma77(tmp_path / "rig.usda")


def test_depth_fit_recovers_a_shift_despite_occluders():
    # a 0.4 x 0.6 m surface of points 2 m away (a body's front, roughly), measured 0.25 m farther; a box in front
    # covering a third of it, the background at 5 m
    g = torch.linspace(-0.2, 0.2, 120)
    yy, xx = torch.meshgrid(torch.linspace(-0.3, 0.3, 180), g, indexing="ij")
    vertices = torch.stack([xx, yy, torch.full_like(xx, 2.0) + 0.05 * xx], -1).reshape(-1, 3)
    truth = vertices + torch.tensor([0.0, 0.0, 0.25])
    depth = visible_depth(truth, K, 480, 640)
    depth[depth == 0] = 5.0
    depth[:, :300] = torch.where(depth[:, :300] < 4, torch.full_like(depth[:, :300], 1.2), depth[:, :300])
    fit = fit_depth(vertices, depth, K)
    assert fit is not None and fit.shift == pytest.approx(0.25, abs=0.005) and fit.residual < 0.005
    centre = vertices.mean(0)
    after = moved(vertices, centre, fit.shift)
    assert float((after[:, 2] - vertices[:, 2]).mean()) == pytest.approx(fit.shift)
    assert fit_depth(vertices, torch.zeros(480, 640), K) is None  # no depth at all
