#!/bin/bash
# Sets xblick-gem-x up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv; GEM-X's package from GitHub at a fixed commit)
# * the weights in weights/ (only if missing; the node runs offline): GEM-X's Body Pose (Hugging Face nvidia/GEM-X,
#   3.5 GB, fixed revision; NVIDIA Open Model License), SOMA 0.3.3's body assets (nvidia/SOMA-X, its v0.3.3 revision;
#   Apache-2.0) and the YOLOX-X HumanArt person detector (OpenMMLab's ONNX archive, checked by its SHA-256)
#
#     tools/setup.sh
#
# Needs uv (UV names it, default uv on PATH), git, curl and unzip. Run again, it does only what
# changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }
YOLOX_URL=https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip
YOLOX_SHA256=1c6fccc5cd450eb563b1fe4eba13c9bdf9ea8263b6ae1cb3ada765aa4ebea2c7

"$UV" sync --frozen
mkdir -p weights/soma
HF_HUB_OFFLINE=0 .venv/bin/python -c "
from huggingface_hub import hf_hub_download
from xblick_gem_x.gemx import BODY_FILES, GEMX_REVISION, SOMA_FILES, SOMA_REVISION, WEIGHTS
for f in BODY_FILES:
    if not (WEIGHTS / f).is_file():
        print(hf_hub_download('nvidia/GEM-X', f, revision=GEMX_REVISION, local_dir=WEIGHTS))
for f in SOMA_FILES:
    if not (WEIGHTS / 'soma' / f).is_file():
        print(hf_hub_download('nvidia/SOMA-X', f, revision=SOMA_REVISION, local_dir=WEIGHTS / 'soma'))"
if [[ ! -f weights/yolox_humanart.onnx ]]; then
    curl -sSfL -o weights/yolox_humanart.zip.tmp "$YOLOX_URL"
    echo "$YOLOX_SHA256  weights/yolox_humanart.zip.tmp" | sha256sum --check --quiet || {
        echo "$YOLOX_URL: not the archive this node was tested with" >&2; exit 1; }
    unzip -p weights/yolox_humanart.zip.tmp "*/end2end.onnx" > weights/yolox_humanart.onnx.tmp
    mv weights/yolox_humanart.onnx.tmp weights/yolox_humanart.onnx
    rm weights/yolox_humanart.zip.tmp
fi
echo "xblick-gem-x: set up; weights in weights/"
