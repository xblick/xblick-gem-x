#!/bin/bash
# Sets xblick-gem-x up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv; GEM-X's package from GitHub at a fixed commit)
# * the weights in weights/ through xblick-gem-x --download-only (weights.py; only if missing): GEM-X's Body Pose
#   (Hugging Face nvidia/GEM-X, 3.5 GB, fixed revision; NVIDIA Open Model License), SOMA 0.3.3's body assets
#   (nvidia/SOMA-X, its v0.3.3 revision; Apache-2.0) and the YOLOX-X HumanArt person detector (OpenMMLab's ONNX
#   archive, checked by its SHA-256). The node downloads missing ones at its start too.
#
#     tools/setup.sh
#
# Before it downloads a model, xblick-gem-x prints the model's terms (weights.TERMS): whoever runs the script is the
# one who uses the model.
#
# Needs uv (UV names it, default uv on PATH) and git. Run again, it does only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }

"$UV" sync --frozen
.venv/bin/xblick-gem-x --download-only
echo "xblick-gem-x: set up; weights in weights/"
