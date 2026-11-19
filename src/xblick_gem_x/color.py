"""A camera's colour stream alone, for ``xblick-gem-x --no-depth``: the counterpart of xblick.rgbd's RGBDInputs and
FramePreparer without a depth stream. ``ColorInputs`` subscribes to ``<camera>/<color>`` only (never to a depth
stream); ``ColorPreparer`` turns one of its frames into a ``ColorFrame`` on the GPU, at the colour stream's own
resolution, undistorted into the pinhole camera of the colour CameraInfo's P (rgbd.Undistorter, as the RGB-D path).

GEM-X needs the camera's intrinsics in any case: the colour stream's camera_info gives them (``xblick play`` and
``xblick cam`` publish one from an assumed field of view unless a calibration is given).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from xblick import tensors
from xblick.camera import ImageSubscriber
from xblick.history import FrameHistory
from xblick.rgbd import CameraSpec, RGBDInputs, Undistorter


def color_only_spec(text: str) -> CameraSpec:
    """``CAMERA[:color=NAME]`` for ``--no-depth``: a ``depth=`` in it is an error, never ignored."""
    spec = CameraSpec.parse(text)
    named = {part.partition("=")[0] for part in text.strip().split(":")[1:]}
    if "depth" in named:
        raise ValueError(f"{text}: --no-depth reads no depth stream, so the camera names none (CAMERA[:color=NAME])")
    return spec


class ColorInputs(RGBDInputs):
    """The colour stream of ``spec`` alone, taken frame by frame (FrameHistory), with RGBDInputs' interface
    (``newest_complete``, ``take``, ``waiting``, ``check``, ``close``, ``readers``): a colour frame is complete by
    itself."""

    def __init__(self, session, spec: CameraSpec, history: int, local: str, wake: threading.Event,
                 max_hz: float | None = None, device=None):
        self.spec = spec
        self.readers = {"color": FrameHistory(ImageSubscriber(session, spec.keys["color"], transport="auto",
                                                              camera_info=True, local=local, max_hz=max_hz,
                                                              device=device),
                                              history, wake.set)}
        self.histories = dict(self.readers)
        self.detections = None


@dataclass
class ColorFrame:
    """One colour frame on the GPU, in the pinhole camera ``K``; ``depth`` None (no depth was read)."""

    rgb: object  # (H, W, 3) uint8
    depth: None
    K: np.ndarray
    frame_id: str


class ColorPreparer:
    """Turns a stamp of a camera's ColorInputs into a ColorFrame; keeps the undistortion of the last camera model."""

    def __init__(self, device):
        self.device = device
        self.undistorter = None
        self.model_key = None

    def __call__(self, inputs: ColorInputs, stamp: int) -> ColorFrame | None:
        """The frame of ``stamp``; None until the colour stream's camera_info came, and for a frame whose size is not
        that camera_info's (a camera that came back with another size)."""
        frames = inputs.take(stamp)
        if frames is None:
            return None
        color = frames["color"]
        info = inputs.readers["color"].subscriber.camera_info
        if info is None:
            return None
        height, width = color.image.shape[:2]
        if (info.height, info.width) != (height, width):
            return None
        key = (tuple(info.k), tuple(info.d), tuple(info.p), tuple(info.r), info.distortion_model, width, height)
        if self.model_key != key:
            self.undistorter = Undistorter(info, height, width, self.device)
            self.model_key = key
        rgb = tensors.rgb(color, self.device).contiguous()
        return ColorFrame(self.undistorter.apply(rgb, "bilinear"), None, self.undistorter.K,
                          info.header.frame_id or color.header.frame_id)
