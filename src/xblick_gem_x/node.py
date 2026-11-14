"""xblick-gem-x: people's 3D bodies from cameras with depth by GEM-X (gemx.py), an alternative to the Azure
Kinect's body tracking (xblick-azure-kinect's xblick-k4abt) that runs on any colour camera with depth.

    xblick-gem-x --camera CAMERA[:color=NAME][:depth=NAME] [--camera ...] [--node gem_x]
                     [--bodies-key KEY] [--no-depth-fit] [--autocast none] [--no-download]

A camera is ``<CAMERA>/<color>`` and ``<CAMERA>/<depth>`` (16UC1 mm in the colour camera's geometry), paired by stamp,
at the depth's resolution, undistorted (xblick.rgbd). Per frame: YOLOX's persons (boxes at least --min-height px
tall, score over --min-score), matched to the persons of the frames before by their boxes' overlap (tracking.py: an id
per person); GEM-X's body of each (one batch); each body moved along its ray until its visible surface lies on the
measured depth (depth.py; ``--no-depth-fit``: GEM-X's monocular distance as it is).

Published per camera, stamped as the frame: ``--bodies-key`` (default ``<CAMERA>/<node>/bodies``; ``{camera}`` and
``{node}`` are filled in), JSON as xblick-azure-kinect's bodies (skeleton.py: model K4ABT_BODY_32, positions in mm
in the colour camera's frame; also each body's SOMA joints and its depth fit). With ``--bodies-key {camera}/bodies``
it stands in for the Azure Kinect's tracker for its readers (Durchblick's Rerun bridge, ...): run one of the two.

Weights (weights.py): missing ones are downloaded at the start, before the models load; --no-download ends the node
instead. ``xblick-gem-x --download-only`` only downloads and checks them (no camera, no Zenoh, no GPU), for
preparation scripts.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import zenoh
from xblick import node, topics
from xblick.camera import CountedPublisher
from xblick.rgbd import CameraSpec, FramePreparer, RGBDInputs
from xblick.session import add_arguments, open_from_args

from . import weights

NODE = "gem_x"


@dataclass
class Camera:
    spec: CameraSpec
    inputs: RGBDInputs
    prepare: FramePreparer
    publisher: object
    tracker: object
    report: object  # xblick.node.Reporter
    last_stamp: int | None = None


# the parameters that change while the node runs (Parameters.declare_options); every other option applies at the start
LIVE = {"min_score": {"range": (0.0, 1.0, 0.01)},
        "min_height": {"range": (0, 4000, 10), "unit": "pixels"},
        "max_persons": {"range": (1, 100, 1)},
        "gate": {"range": (0.0, 1.0, 0.01), "unit": "m"}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="xblick-gem-x", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", action="append", required=True, metavar="SPEC",
                        help="CAMERA[:color=NAME][:depth=NAME] (repeated)")
    parser.add_argument("--node", default=NODE)
    parser.add_argument("--bodies-key", default="{camera}/{node}/bodies",
                        help="where the bodies go ({camera}/bodies: in the Azure Kinect tracker's place)")
    parser.add_argument("--no-depth-fit", action="store_true", help="GEM-X's monocular distance, not fitted")
    parser.add_argument("--autocast", choices=["bfloat16", "none"], default="bfloat16")
    parser.add_argument("--detector-device", choices=["cuda", "cpu"], default="cuda",
                        help="YOLOX on ONNX Runtime's CUDA build (16 ms) or on the CPU (200-450 ms)")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--min-height", type=int, default=60, help="smaller persons' boxes [px] are left out")
    parser.add_argument("--max-persons", type=int, default=6, help="the largest this many per frame")
    parser.add_argument("--gate", type=float, default=0.08, help="[m] of the depth fit (depth.py)")
    parser.add_argument("--max-offset-ms", type=float, default=None,
                        help="pair a colour frame with the nearest depth frame this close (default: half the "
                             "depth's frame period, measured: each colour frame with its nearest depth frame, as a "
                             "camera whose depth is stamped apart from its colour needs, one Azure Kinect we measured: "
                             "11-17 ms); 0: only equal stamps")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    weights.add_arguments(parser)  # --download-only (a mode of the command line), --no-download
    add_arguments(parser)
    # --max-hz: at most 20 colour frames/s (about 10 measured; 0: every frame)
    # --local: auto (the CUDA ring on this GPU, else shared memory), shm or cuda
    node.add_arguments(parser, max_hz=20.0, local=("auto", "shm", "cuda"))
    node.Parameters.add_arguments(parser)  # --param NAME:=VALUE, --params-file, --dump-params, --params-read-only
    args = weights.parse_args(parser, argv)  # --download-only: --camera not required
    if args.download_only:
        return weights.download_only(weights.needed())
    # the options as parameters (the terminal interface shows them): read-only, but LIVE
    session = open_from_args(args)
    params = node.Parameters(session, args.node)
    params.configure(args)
    params.declare_options(parser, args, live=LIVE, leave_out=("node", "download_only"))
    params.check_overrides()
    try:
        specs = [CameraSpec.parse(c) for c in args.camera]
        for spec in specs:
            topics.check_name(spec.camera)
            if spec.segmenter is not None:
                raise ValueError(f"{spec}: a segmenter is not used here")
            topics.check_name(args.bodies_key.format(camera=spec.camera, node=args.node))
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if len({s.camera for s in specs}) != len(specs):
        parser.error("--camera: a camera given twice")

    # SOMA's skinning: torch's notes on sparse tensors (beta, invariant checks) at the first frame
    node.start("xblick-gem-x", ignore=["Sparse CSR tensor support is in beta", "Sparse invariant checks",
                                      "Validating sparse tensor invariants"], max_hz=args.max_hz, local=args.local)
    try:
        weights.ensure(weights.needed(), download=not args.no_download)
    except weights.WeightsError as error:
        params.close()
        session.close()
        return weights.fail(error)

    import onnxruntime  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import warp  # noqa: PLC0415

    warp.config.quiet = True  # no banner and kernel-cache lines (SOMA's skinning)
    onnxruntime.set_default_logger_severity(3)  # errors only: "no registered plugin EP device" for CUDA is a notice

    from .depth import fit_depth, moved  # noqa: PLC0415
    from .gemx import GemX, PersonDetector, square_boxes  # noqa: PLC0415
    from .skeleton import bodies_message  # noqa: PLC0415
    from .tracking import Tracker  # noqa: PLC0415

    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    print("xblick-gem-x: loading GEM-X and YOLOX", flush=True)
    gemx = GemX(device, autocast=None if args.autocast == "none" else args.autocast)
    detector = PersonDetector(device=f"cuda:{device.index}" if args.detector_device == "cuda" else "cpu",
                              score=args.min_score)
    wake = threading.Event()
    cameras = []
    for spec in specs:
        key = args.bodies_key.format(camera=spec.camera, node=args.node)
        # colour limited to --max-hz, its depth the nearest within --max-offset-ms (decoded only for colour taken)
        inputs = RGBDInputs(session, spec, args.history, args.local, wake, max_hz=args.max_hz, device=device,
                            max_offset_ns=None if args.max_offset_ms is None else round(args.max_offset_ms * 1e6))
        report = node.Reporter(session, spec.camera, args.node, "xblick-gem-x", log_prefix=f"{spec.camera}: ",
                               max_hz=args.max_hz)
        report.watch(inputs)
        # the attachment (seq, publish_ns) as every Durchblick sample has it: the bodies' age at send
        cameras.append(Camera(spec, inputs, FramePreparer(device),
                              CountedPublisher(session, key, zenoh.Encoding.APPLICATION_JSON, source="gem-x"),
                              Tracker(), report))
        print(f"xblick-gem-x: {spec} -> {key}", flush=True)

    def process(camera: Camera, frame, stamp: int):
        started = time.monotonic()
        rgb = frame.rgb.cpu().numpy()
        boxes, scores = detector.detect(rgb)
        tall = (boxes[:, 3] - boxes[:, 1]) >= args.min_height
        boxes = boxes[tall]
        if len(boxes) > args.max_persons:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes[np.argsort(-area)[:args.max_persons]]
        ids = camera.tracker.update(boxes)
        out = []
        if len(boxes):
            bodies = gemx.predict(rgb, square_boxes(boxes), frame.K)
            for i, person in enumerate(ids):
                joints, fit = bodies.joints[i], None
                if not args.no_depth_fit:
                    fit = fit_depth(bodies.vertices[i], frame.depth, frame.K, gate=args.gate)
                    if fit is not None:
                        joints = moved(joints, joints[0], fit.shift)
                        camera.report.counts["fitted"] += 1
                out.append({"id": person, "joints": joints.cpu().numpy(), "valid": bodies.valid[i].cpu().numpy(),
                            "depth_fit": None if fit is None else fit.as_dict()})
        latency = (time.time_ns() - stamp) / 1e6
        camera.publisher.put(bodies_message(stamp, frame.frame_id, latency, out))
        camera.report.counts["frames"] += 1
        camera.report.counts["bodies"] += len(out)
        camera.report.counts["seconds"] += time.monotonic() - started

    def report(camera: Camera):
        c, seconds = camera.report.take()
        n = max(c["frames"], 1)
        level, message = node.state(f"{c['frames'] / seconds:.1f} frames/s, {c['bodies'] / n:.1f} bodies a frame "
                                    f"({c['fitted']} of {c['bodies']} fitted to the depth)",
                                    waiting=None if c["frames"] else camera.inputs.waiting())
        camera.report.publish(level, message, {"frames/s": c["frames"] / seconds, "bodies/frame": c["bodies"] / n,
                                               "ms/frame": 1e3 * c["seconds"] / n, "streams": str(camera.spec)},
                              log_extra=f", {1e3 * c['seconds'] / n:.0f} ms a frame")

    try:
        while True:
            wake.wait(timeout=0.1)
            wake.clear()
            for name in LIVE:  # the live parameters' values now, in this thread (read per frame)
                setattr(args, name, params[name])
            detector.score = args.min_score
            for camera in cameras:
                camera.inputs.check()
                stamp = camera.inputs.newest_complete()
                if stamp is None or stamp == camera.last_stamp:
                    continue
                camera.last_stamp = stamp
                frame = camera.prepare(camera.inputs, stamp)
                if frame is not None:
                    process(camera, frame, stamp)
            for camera in cameras:
                if camera.report.due():
                    report(camera)
    except KeyboardInterrupt:
        pass
    finally:
        for camera in cameras:
            camera.inputs.close()
            camera.publisher.close()
            camera.report.close()
        params.close()
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
