"""``--no-depth``, the colour stream alone (color.py): the camera spec without depth, the bodies message's marker,
ColorInputs and ColorPreparer on a published colour stream (no depth stream anywhere), and the node itself (needs a
CUDA device and the weights): with --no-depth it publishes bodies messages that say ``"depth": "none"`` from a
colour stream alone; without it, it publishes nothing while no depth comes, as before. Each test with a router of
its own on a free loopback port."""

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import xblick as db

from xblick_gem_x.color import ColorInputs, ColorPreparer, color_only_spec
from xblick_gem_x.skeleton import bodies_message
from xblick_gem_x.weights import STORE

BIN = Path(sys.executable).parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def router():
    """``xblick router`` on a free loopback port; yields its endpoint."""
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    process = subprocess.Popen([str(BIN / "xblick"), "router", "--listen", endpoint], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    try:
        yield endpoint
    finally:
        process.terminate()
        process.wait(timeout=10)


class Colour:
    """A colour stream ``<camera>/color`` (640 x 480, its camera_info) published at 10 Hz while somebody reads it."""

    def __init__(self, session, camera: str, image: np.ndarray | None = None):
        info = db.CameraInfo.pinhole(640, 480, fx=500.0, fy=500.0, cx=319.5, cy=239.5, frame_id="test_optical")
        self.publisher = db.CameraPublisher(session, f"{camera}/color", encoding="rgb8", frame_id="test_optical",
                                            camera_info=info, source="test")
        self.image = np.full((480, 640, 3), 127, np.uint8) if image is None else image
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.done.is_set():
            if self.publisher.subscribed:
                self.publisher.publish(self.image, stamp_ns=time.time_ns())
            time.sleep(0.1)

    def close(self):
        self.done.set()
        self.thread.join(timeout=5)
        self.publisher.close()


def test_a_colour_only_camera_names_no_depth():
    assert color_only_spec("cam/video0").keys["color"] == "cam/video0/color"
    assert color_only_spec("cam/rc:color=left").keys["color"] == "cam/rc/left"
    with pytest.raises(ValueError, match="--no-depth reads no depth stream"):
        color_only_spec("cam/azurekinect:depth=depth_to_color")


def test_the_message_says_depth_none_only_without_depth():
    joints, valid = np.zeros((77, 3)) + [0.0, 0.0, 2.0], np.ones(77, bool)
    body = {"id": 1, "joints": joints, "valid": valid, "depth_fit": None}
    rgbd = json.loads(bodies_message(5, "cam_optical", 1.0, [body]))
    assert "depth" not in rgbd
    assert list(rgbd) == ["v", "model", "source", "stamp_ns", "frame_id", "latency_ms", "bodies"]
    colour = json.loads(bodies_message(5, "cam_optical", 1.0, [body], depth="none"))
    assert colour["depth"] == "none" and colour["bodies"][0]["depth_fit"] is None
    assert {k: v for k, v in colour.items() if k != "depth"} == rgbd


def test_colour_inputs_give_frames_without_any_depth_stream(router):
    session = db.open_session(connect=[router])
    colour = Colour(session, "test/gx")
    wake = threading.Event()
    inputs = ColorInputs(session, color_only_spec("test/gx"), 10, "auto", wake)
    try:
        assert list(inputs.readers) == ["color"]  # no subscriber of test/gx/depth
        prepare, frame = ColorPreparer(torch.device("cpu")), None
        deadline = time.monotonic() + 15
        while frame is None and time.monotonic() < deadline:
            wake.wait(0.2)
            wake.clear()
            stamp = inputs.newest_complete()
            if stamp is not None:
                frame = prepare(inputs, stamp)
        assert frame is not None, inputs.waiting()
        assert tuple(frame.rgb.shape) == (480, 640, 3) and frame.depth is None  # the colour stream's own size
        assert np.allclose(frame.K, [[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]])
        assert frame.frame_id == "test_optical"
    finally:
        inputs.close()
        colour.close()
        session.close()


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
needs_weights = pytest.mark.skipif(not (STORE.root / "body_s240000.ckpt").is_file(), reason="needs the weights")


def bodies_of(router, arguments: list[str], seconds: float) -> tuple[list[dict], str]:
    """The node started with ``arguments`` on the colour stream test/gx alone; its bodies messages within
    ``seconds`` after it printed its camera line, and its output."""
    session = db.open_session(connect=[router])
    colour = Colour(session, "test/gx")
    messages = []
    subscriber = session.declare_subscriber("test/gx/gem_x/bodies",
                                            lambda sample: messages.append(json.loads(sample.payload.to_bytes())))
    process = subprocess.Popen([str(BIN / "xblick-gem-x"), "--camera", "test/gx", "--connect", router, "--no-download",
                                *arguments], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    try:
        deadline = time.monotonic() + 240  # loading GEM-X and YOLOX
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            lines.append(line)
            if "-> test/gx/gem_x/bodies" in line:
                break
        time.sleep(seconds)
    finally:
        process.terminate()
        out, _ = process.communicate(timeout=30)
        lines.append(out)
        subscriber.undeclare()
        colour.close()
        session.close()
    return messages, "".join(lines)


@needs_cuda
@needs_weights
def test_the_node_without_depth_publishes_from_colour_alone(router):
    messages, out = bodies_of(router, ["--no-depth"], 15)
    assert "colour only (--no-depth)" in out and "test/gx (colour only) -> test/gx/gem_x/bodies" in out, out
    assert messages, out
    assert all(m["depth"] == "none" and m["frame_id"] == "test_optical" for m in messages)
    assert all(b["depth_fit"] is None for m in messages for b in m["bodies"])


@needs_cuda
@needs_weights
def test_the_node_with_depth_still_waits_for_it(router):
    messages, out = bodies_of(router, [], 15)
    assert messages == [], out  # no depth stream: no frame is complete, as before


def test_no_depth_and_no_depth_fit_exclude_each_other():
    done = subprocess.run([str(BIN / "xblick-gem-x"), "--camera", "test/gx", "--no-depth", "--no-depth-fit"],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "not allowed with argument" in done.stderr


def test_no_depth_refuses_a_camera_with_depth(router):
    done = subprocess.run([str(BIN / "xblick-gem-x"), "--camera", "cam/a:depth=depth_to_color", "--no-depth",
                           "--connect", router], capture_output=True, text=True, timeout=60)
    assert done.returncode == 2 and "--no-depth reads no depth stream" in done.stderr
