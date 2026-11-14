"""The node's weights: which files it reads, where they come from, and their download.

This module is the one list of them: the node's start and ``tools/setup.sh`` (through ``xblick-gem-x
--download-only``) both use it. ``STORE.ensure`` touches neither the disk nor the network when every file is there. A
missing file is downloaded under a temporary name in its directory (``<name>.part``), checked and only then renamed,
all under a file lock, so that two nodes started at once download once and never read half a file.

* GEM-X's Body Pose (Hugging Face nvidia/GEM-X at ``gemx.GEMX_REVISION``, ``gemx.BODY_FILES``) in ``weights/``
* SOMA 0.3.3's body assets (nvidia/SOMA-X at ``gemx.SOMA_REVISION``, ``gemx.SOMA_FILES``) in ``weights/soma/``
* the YOLOX-X HumanArt person detector (``gemx.DETECTOR``): ``end2end.onnx`` of OpenMMLab's archive, the archive
  checked by its SHA-256 (``ZipMember``)
* ``SHA256``: each Hugging Face file's SHA-256 (Hugging Face's ``lfs.oid`` at the revision for LFS files, the hash of
  the file at the revision for the two small JSON files; all checked against copies downloaded before)
* ``needed()``: all of them (every start reads them)

The download itself (lock, ``.part`` files, checks, progress, ``--download-only``/``--no-download``) is the core's
``xblick.weights``; this module keeps the node's files, its terms notice and its ``STORE``.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from xblick.weights import CHUNK, HuggingFace, Store, Weight, WeightsError, sha256_of

from .gemx import BODY_FILES, DETECTOR, GEMX_REVISION, REPO, SOMA_FILES, SOMA_REVISION

PROG = "xblick-gem-x"
NODE = "gem-x"  # the short name: its cache directory is ~/.cache/xblick/gem-x
FORMER = REPO / "weights"  # the weights' place before 0.2.4 (the checkout): not used, the store names the move
REQUIRED = ()  # (path, what): code from tools/setup.sh that must be there first; none here
YOLOX_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
YOLOX_SHA256 = "1c6fccc5cd450eb563b1fe4eba13c9bdf9ea8263b6ae1cb3ada765aa4ebea2c7"  # of the archive
SIZES = {  # bytes, exact (checked after a download)
    "body_s240000.ckpt": 3_538_753_341, "body_config.json": 1_086, "SOMA_neutral.npz": 26_608_121,
    "SOMA_template_rig.usda": 28_306_038, "SOMA_procedural_transforms.json": 20_045,
    "correctives_model.pt": 70_549_157, DETECTOR: 396_283_264,
}
SHA256 = {  # the Hugging Face files; YOLOX: its archive's YOLOX_SHA256 (the extracted file has none from a source)
    "body_s240000.ckpt": "e244d27f2834eb6382c5f6b9cbf2a38681b685514a1b2e86f8f3d2592d614959",
    "body_config.json": "61c29c72b93f88789ecb7beb21c526c78e92e24ffbce91d91e51984272b26368",
    "SOMA_neutral.npz": "5829af4a5333abdba5ad449360b868faacee99d395971e62f35c0e7169432b63",
    "SOMA_template_rig.usda": "a952cb7bdf4ccd801e355d53122d395b985b69d0d9181439edad5081bb171856",
    "SOMA_procedural_transforms.json": "9853e19ab51625ffaa7d9e7ccf0e443272c2740837604e6bacd66604e8edf509",
    "correctives_model.pt": "411d061a651051a7a4ec941543495d35eac21e5a7a69c8ce042b89cd12772a43",
}
# The terms of what is downloaded, printed before the download (whoever downloads is the one who uses the models).
TERMS = """\
The models are the work of others, under their terms, and you are the one who uses them:
  * GEM-X Body Pose weights (NVIDIA, Hugging Face nvidia/GEM-X): NVIDIA Open Model License.
  * SOMA body assets (NVIDIA, Hugging Face nvidia/SOMA-X): Apache License 2.0.
  * YOLOX-X HumanArt person detector (OpenMMLab's download): OpenMMLab states no licence for the model file.
    It is trained on COCO and Human-Art, and Human-Art is offered for non-commercial purposes only.
    Whether you may use the detector for your purpose, a commercial one above all, is yours to check."""


def needed() -> list[Weight]:
    """Every file a start reads."""
    return ([Weight(Path(f), HuggingFace("nvidia/GEM-X", GEMX_REVISION, f), SIZES[f], SHA256[f])
             for f in BODY_FILES]
            + [Weight(Path("soma") / f, HuggingFace("nvidia/SOMA-X", SOMA_REVISION, f), SIZES[f], SHA256[f])
               for f in SOMA_FILES]
            + [Weight(Path(DETECTOR), ZipMember(YOLOX_URL, YOLOX_SHA256, "*/end2end.onnx"), SIZES[DETECTOR])])


@dataclass(frozen=True)
class ZipMember:
    """The one file of a zip archive at a URL whose name matches ``member`` (a glob); the archive checked against its
    ``sha256``. The archive goes beside the file (``<tmp>.zip``) and is removed after; a wrong SHA-256 keeps it."""

    url: str
    sha256: str
    member: str

    def describe(self) -> str:
        return f"{self.member} of {self.url} (the archive checked by its SHA-256)"

    def fetch(self, tmp: Path, progress, store: Store) -> str:
        archive = tmp.with_name(tmp.name + ".zip")
        try:
            digest = store.http(self.url, archive, progress)
        except BaseException:
            archive.unlink(missing_ok=True)
            raise
        if digest != self.sha256:
            raise WeightsError(f"{store.short(archive)}: wrong SHA-256 ({digest}, expected {self.sha256}): not the "
                               "archive this node was tested with; remove it, then start again")
        try:
            with zipfile.ZipFile(archive) as z:
                names = fnmatch.filter(z.namelist(), self.member)
                if len(names) != 1:
                    raise WeightsError(f"{self.url}: {len(names)} files match {self.member}, expected one")
                with z.open(names[0]) as source, open(tmp, "wb") as f:
                    shutil.copyfileobj(source, f, CHUNK)
                    f.flush()
                    os.fsync(f.fileno())
        finally:
            archive.unlink(missing_ok=True)
        return sha256_of(tmp)


def options(args) -> list[str]:
    """The options that select the weights (and --weights-dir), for the download command the lines name."""
    return [] + (["--weights-dir", args.weights_dir] if args.weights_dir else [])


def store_for(parser, args) -> Store:
    """STORE at --weights-dir (not given: the node's cache directory); a wrong --weights-dir is an argument error."""
    try:
        return STORE.at(args.weights_dir)
    except ValueError as error:
        parser.error(f"--weights-dir: {error}")


# the download (xblick.weights): the root is the node's cache directory, its lock there
STORE = Store(PROG, node=NODE, terms=TERMS, required=REQUIRED, repo=REPO, former=[FORMER])
