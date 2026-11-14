#!/bin/bash
# Sets xblick-gem-x up from scratch or updates it:
#
# * the venv exactly as uv.lock fixes it (.venv; GEM-X's package from GitHub at a fixed commit)
# * the weights in weights/ (only if missing; the node runs offline): GEM-X's Body Pose (Hugging Face nvidia/GEM-X,
#   3.5 GB, fixed revision; NVIDIA Open Model License), SOMA 0.3.3's body assets (nvidia/SOMA-X, its v0.3.3 revision;
#   Apache-2.0) and the YOLOX-X HumanArt person detector (OpenMMLab's ONNX archive, checked by its SHA-256)
#
#     tools/setup.sh [--accept-model-terms]
#
# Before it downloads a model it names the models' terms and asks. A script without a terminal passes
# --accept-model-terms (or sets XBLICK_GEM_X_ACCEPT_MODEL_TERMS=yes); without either it ends with an error.
#
# Needs uv (UV names it, default uv on PATH), git, curl and unzip. Run again, it does only what
# changed.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
command -v "$UV" > /dev/null || { echo "no uv: $UV" >&2; exit 1; }
YOLOX_URL=https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip
YOLOX_SHA256=1c6fccc5cd450eb563b1fe4eba13c9bdf9ea8263b6ae1cb3ada765aa4ebea2c7

ACCEPT=${XBLICK_GEM_X_ACCEPT_MODEL_TERMS:-}
[[ -z "$ACCEPT" || "$ACCEPT" == yes ]] || {
    echo "XBLICK_GEM_X_ACCEPT_MODEL_TERMS=$ACCEPT: yes, or not set" >&2; exit 1; }
for argument in "$@"; do
    case "$argument" in
        --accept-model-terms) ACCEPT=yes ;;
        *) echo "$argument: unknown argument (tools/setup.sh [--accept-model-terms])" >&2; exit 1 ;;
    esac
done

# The terms of the models to download, and the user's yes to them: asked on a terminal, given by
# --accept-model-terms or XBLICK_GEM_X_ACCEPT_MODEL_TERMS=yes, else an error. $1: the missing files.
accept_model_terms() {
    cat <<TERMS
xblick-gem-x: to download: $1
The models are the work of others, under their terms, and you are the one who uses them:
  * GEM-X Body Pose weights (NVIDIA, Hugging Face nvidia/GEM-X): NVIDIA Open Model License.
  * SOMA body assets (NVIDIA, Hugging Face nvidia/SOMA-X): Apache License 2.0.
  * YOLOX-X HumanArt person detector (OpenMMLab's download): OpenMMLab states no licence for the model file.
    It is trained on COCO and Human-Art, and Human-Art is offered for non-commercial purposes only.
    Whether you may use the detector for your purpose, a commercial one above all, is yours to check.
TERMS
    if [[ "$ACCEPT" == yes ]]; then
        echo "xblick-gem-x: terms accepted (--accept-model-terms or XBLICK_GEM_X_ACCEPT_MODEL_TERMS=yes)"
    elif [[ -t 0 && -t 1 ]]; then
        read -r -p "Download them under these terms? [y/N] " answer
        [[ "$answer" == y || "$answer" == Y || "$answer" == yes ]] || {
            echo "xblick-gem-x: nothing downloaded" >&2; exit 1; }
    else
        echo "xblick-gem-x: no terminal to ask on: pass --accept-model-terms or set" \
             "XBLICK_GEM_X_ACCEPT_MODEL_TERMS=yes once you accept these terms; nothing downloaded" >&2
        exit 1
    fi
}

"$UV" sync --frozen
mkdir -p weights/soma
missing=$(.venv/bin/python -c "
from xblick_gem_x.gemx import BODY_FILES, SOMA_FILES, WEIGHTS
print(' '.join([f for f in BODY_FILES if not (WEIGHTS / f).is_file()]
               + ['soma/' + f for f in SOMA_FILES if not (WEIGHTS / 'soma' / f).is_file()]))")
[[ -f weights/yolox_humanart.onnx ]] || missing="$missing yolox_humanart.onnx"
[[ -z "${missing// /}" ]] || accept_model_terms "$missing"
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
