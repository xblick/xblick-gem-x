"""GEM-X's Body Pose and YOLOX loaded from weights/: a body for a crop, with the anatomy of a person (the mapped
joints where SOMA77 says: hips between the legs' roots, knees below them, a plausible size), and the persons' detector
on an empty image. Needs a CUDA device and weights/ (tools/setup.sh)."""

import numpy as np
import pytest
import torch

from durchblick_gem_x.gemx import WEIGHTS

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
              pytest.mark.skipif(not (WEIGHTS / "body_s240000.ckpt").is_file(), reason="needs weights/")]


def test_a_body_with_a_persons_anatomy():
    from durchblick_gem_x.gemx import GemX
    from durchblick_gem_x.skeleton import INDEX

    gemx = GemX("cuda")
    rgb = np.full((480, 640, 3), 127, np.uint8)  # GEM-X answers with a body for any crop: its shape is checked here
    K = np.array([[600.0, 0, 319.5], [0, 600.0, 239.5], [0, 0, 1]])
    b = gemx.predict(rgb, np.array([[320.0, 240.0, 400.0], [200.0, 240.0, 300.0]]), K)
    assert b.joints.shape == (2, 77, 3) and b.vertices.shape[0] == 2 and b.keypoints.shape == (2, 77, 2)
    j = b.joints[0].cpu().numpy()
    hips, left, right = j[INDEX["Hips"]], j[INDEX["LeftLeg"]], j[INDEX["RightLeg"]]
    assert 0.05 < np.linalg.norm(left - right) < 0.4  # the legs' roots: a pelvis' width apart
    assert np.linalg.norm((left + right) / 2 - hips) < 0.15
    thigh = np.linalg.norm(j[INDEX["LeftShin"]] - left)
    assert 0.25 < thigh < 0.6  # LeftShin is the knee
    height = np.ptp(b.vertices[0].cpu().numpy(), axis=0).max()
    assert 1.0 < height < 2.2
    assert (b.joints[:, :, 2] > 0).all()  # in front of the camera


def test_no_person_in_an_empty_image():
    from durchblick_gem_x.gemx import PersonDetector

    boxes, scores = PersonDetector(device="cuda:0").detect(np.full((480, 640, 3), 127, np.uint8))
    assert boxes.shape == (0, 4) and scores.shape == (0,)
