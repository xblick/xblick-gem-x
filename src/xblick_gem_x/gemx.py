"""GEM-X (github.com/NVlabs/GEM-X, NVIDIA, 2026: full-body and hand pose from images and video, on the SOMA body
model): its Body Pose model, one image crop per person, with the camera's intrinsics: per person SOMA's 77 joints and
its mesh in the camera [m], and the joints' projections. Its person detector: YOLOX-X (OpenMMLab's HumanArt ONNX
model) on ONNX Runtime, on the GPU with ONNX Runtime's CUDA build.

The code: the package ``gem`` at a fixed commit (pyproject.toml); the weights (Hugging Face nvidia/GEM-X, SOMA 0.3.3's
assets nvidia/SOMA-X at their revisions, YOLOX at its archive's SHA-256) in weights/: weights.py downloads them,
at the node's start when missing. Nothing is fetched while it works; Warp's kernel cache (SOMA's skinning) is
weights/warp-cache.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights"
GEMX_REVISION = "4fce057963671fd95cd4154297944397e7b5a9a0"  # nvidia/GEM-X
SOMA_REVISION = "d281db2d01553f7230c56e764cec00fe27ef23f7"  # nvidia/SOMA-X, tag v0.3.3
BODY_FILES = ("body_s240000.ckpt", "body_config.json")
SOMA_FILES = ("SOMA_neutral.npz", "SOMA_template_rig.usda", "SOMA_procedural_transforms.json",
              "correctives_model.pt")
DETECTOR = "yolox_humanart.onnx"
BOX_ENLARGE = 1.2  # a person's crop around its box (GEM-X's detect_track)


@dataclass
class Bodies:
    """GEM-X's prediction for N persons (torch, on the model's device): joints (N, 77, 3) and vertices (N, V, 3) in
    the camera [m], the joints' projections (N, 77, 2) [px] and their validity (N, 77)."""

    joints: object
    vertices: object
    keypoints: object
    valid: object


def square_boxes(boxes_xyxy: np.ndarray, enlarge: float = BOX_ENLARGE) -> np.ndarray:
    """Person boxes (N, 4) x0, y0, x1, y1 -> GEM-X's square crops (N, 3) centre x, centre y, side."""
    b = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    side = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) * enlarge
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, side], 1)


class GemX:
    """GEM-X's Body Pose on ``device``, loaded once; ``autocast`` bfloat16 runs its backbone so (30 against 43 ms a
    person on an RTX PRO 6000; joints within 2 mm of float32's)."""

    def __init__(self, device="cuda", weights: Path = WEIGHTS, autocast: str | None = "bfloat16"):
        import torch  # noqa: PLC0415

        for path in [weights / f for f in BODY_FILES] + [weights / "soma" / f for f in SOMA_FILES]:
            if not path.is_file():
                raise FileNotFoundError(f"{path}: missing (xblick-gem-x --download-only downloads it)")
        from .skeleton import check_soma77  # noqa: PLC0415

        check_soma77(weights / "soma" / "SOMA_template_rig.usda")
        os.environ.setdefault("WARP_CACHE_PATH", str(weights / "warp-cache"))
        from gem.inference.pose import load_predictor  # noqa: PLC0415

        if autocast not in (None, "bfloat16"):
            raise ValueError(f"autocast {autocast!r}: None or bfloat16")
        self.device = torch.device(device)
        self.predictor, self.release = load_predictor(str(weights / "body_s240000.ckpt"), task="body",
                                                      release="body", soma_assets=str(weights / "soma"),
                                                      device=str(self.device))
        self.dtype = torch.bfloat16 if autocast else None
        self.faces = None

    def predict(self, rgb: np.ndarray, boxes: np.ndarray, K: np.ndarray) -> Bodies:
        """``rgb`` (H, W, 3) uint8, ``boxes`` (N, 3) square crops (square_boxes), ``K`` the camera's (3, 3)."""
        import contextlib  # noqa: PLC0415

        import torch  # noqa: PLC0415

        cast = torch.autocast("cuda", dtype=self.dtype) if self.dtype is not None else contextlib.nullcontext()
        with torch.no_grad(), cast:
            p = self.predictor.predict(np.ascontiguousarray(rgb), torch.as_tensor(boxes, dtype=torch.float32),
                                       torch.as_tensor(K, dtype=torch.float32))
        if self.faces is None:
            self.faces = p.faces[0]
        return Bodies(p.joints_camera.float(), p.vertices_camera.float(), p.keypoints_2d.float(), p.projection_valid)


class PersonDetector:
    """YOLOX-X HumanArt on ONNX Runtime: persons' boxes (N, 4) x0, y0, x1, y1 and scores in an RGB image."""

    def __init__(self, weights: Path = WEIGHTS, device: str = "cuda", score: float = 0.5):
        path = weights / DETECTOR
        if not path.is_file():
            raise FileNotFoundError(f"{path}: missing (xblick-gem-x --download-only downloads it)")
        if device.startswith("cuda"):
            import onnxruntime  # noqa: PLC0415

            onnxruntime.preload_dlls()  # its CUDA 12 libraries from the nvidia-* wheels, beside torch's CUDA 13
        from gem.common.detectors.yolox import YOLOXONNX  # noqa: PLC0415

        self.detector = YOLOXONNX(str(path), score_thr=score, device=device)

    @property
    def score(self) -> float:
        """The least score of a person's box (YOLOX's own threshold, read at every detection)."""
        return self.detector.score_thr

    @score.setter
    def score(self, value: float):
        self.detector.score_thr = value

    def detect(self, rgb: np.ndarray):
        boxes, scores = self.detector.detect(np.ascontiguousarray(rgb[..., ::-1]))  # it takes BGR
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(scores, dtype=np.float32)
