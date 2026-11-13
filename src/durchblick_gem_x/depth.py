"""The measured depth against GEM-X's bodies. GEM-X reconstructs a body from one image: its size from its learned
shape, its distance from the camera from that size and the image, so its depth is off where the person's size is
(a child, a crouching person). With the camera's depth a body is moved along the ray through its centre until its
visible surface lies on the measured one: its size stays GEM-X's, and so, nearly, does its fit in the image.

Visible surface: the body's vertices projected into the depth image, per pixel the nearest one (a z-buffer of points,
not of triangles: the mesh's 18k vertices at a body's usual size in a 640 x 480 image cover its pixels densely enough).
The shift: the median of measured minus predicted depth over those pixels, then again over the pixels within
``gate`` of that first median (occluders: other people, objects in front; the background behind the silhouette).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthFit:
    shift: float  # [m] along the camera's z (positive: the body was too near)
    pixels: int  # the body's visible pixels with measured depth
    inliers: int  # of them within the gate of the shift
    residual: float  # [m] median |measured - corrected| of the inliers

    def as_dict(self) -> dict:
        return {"shift_m": round(self.shift, 4), "pixels": self.pixels, "inliers": self.inliers,
                "residual_m": round(self.residual, 4)}


def visible_depth(vertices, K: np.ndarray, height: int, width: int):
    """Per pixel the nearest vertex's depth (0: none): ``vertices`` (V, 3) torch [m], camera frame."""
    import torch  # noqa: PLC0415

    z = vertices[:, 2]
    front = z > 0.05
    u = torch.round(vertices[:, 0] * K[0, 0] / z.clamp_min(1e-6) + K[0, 2]).long()
    v = torch.round(vertices[:, 1] * K[1, 1] / z.clamp_min(1e-6) + K[1, 2]).long()
    inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    index = v[inside] * width + u[inside]
    image = torch.full((height * width,), float("inf"), device=vertices.device)
    image.scatter_reduce_(0, index, z[inside], reduce="amin")
    image[torch.isinf(image)] = 0.0
    return image.reshape(height, width)


def fit_depth(vertices, depth, K: np.ndarray, gate: float = 0.08, min_pixels: int = 100) -> DepthFit | None:
    """How far to move a body (``vertices`` (V, 3) torch [m]) along its ray so that its visible surface lies on
    ``depth`` (H, W) torch [m] (0: none): None if fewer than ``min_pixels`` of its pixels have depth."""
    h, w = depth.shape
    predicted = visible_depth(vertices, K, h, w)
    both = (predicted > 0) & (depth > 0)
    pixels = int(both.sum())
    if pixels < min_pixels:
        return None
    diff = depth[both] - predicted[both]
    shift = float(diff.median())
    near = (diff - shift).abs() < gate
    if int(near.sum()) < min_pixels:
        return None
    shift = float(diff[near].median())
    near = (diff - shift).abs() < gate
    residual = float((diff[near] - shift).abs().median())
    return DepthFit(shift, pixels, int(near.sum()), residual)


def moved(points, centre, shift: float):
    """``points`` (..., 3) moved along the ray through ``centre`` (3,): every point's depth changes by ``shift``, the
    centre's projection stays (the others' move by a fraction of a pixel per cm of shift at a body's usual
    distance)."""
    return points + shift * centre / centre[2]
