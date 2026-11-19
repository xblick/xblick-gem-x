# xblick-gem-x

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)
[![tests](https://github.com/xblick/xblick-gem-x/actions/workflows/tests.yml/badge.svg)](https://github.com/xblick/xblick-gem-x/actions/workflows/tests.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![device](https://img.shields.io/badge/needs-NVIDIA%20GPU-76b900)](README.md)

This [Durchblick](https://github.com/xblick/xblick) node computes the 3D bodies of the people in
front of a colour camera with depth. With `--no-depth` it works on a colour stream alone, with distances estimated
from the image. It takes about 60 ms per frame with two persons, so it keeps up with a camera at
10 frames per second.

The node uses [GEM-X](https://github.com/NVlabs/GEM-X), a model from NVIDIA (2026) that estimates full-body and hand
pose from images and video. GEM-X works on SOMA, NVIDIA's parametric body model with 77 joints and a mesh of 18,000
vertices. The node is an alternative to the body tracking of the Azure Kinect (`xblick-k4abt` in
xblick-azure-kinect). It needs neither that camera nor its SDK, and it publishes its bodies in the same JSON
layout.

## What it publishes

The node publishes one message per processed frame and camera. The message has the stamp of the frame.

| Topic | Message type | Content |
| --- | --- | --- |
| `<camera>/gem_x/bodies` | JSON, the layout of xblick-azure-kinect's bodies | model `K4ABT_BODY_32`: per body an id and 32 joints in mm in the frame of the colour camera. Each body also has `soma77_mm` with all 77 SOMA joints and `depth_fit` with the result of the depth fit. With `--no-depth` the message also has `"depth": "none"`, and `depth_fit` is null. |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | status `gem_x` once per second: frames per second, bodies per frame, ms per frame, the streams in use |

`--bodies-key` changes the topic of the bodies, and `--node` changes the `gem_x` part. `xblick rerun` draws the
bodies as skeletons in 3D and over the colour stream.

The node reads `<camera>/<color>` and `<camera>/<depth>`. The depth must be 16UC1 in mm in the geometry of the colour
camera. With `--no-depth` it reads `<camera>/<color>` alone. The positions are then GEM-X's estimate: the size of a
person from its learned body shape, the distance from that size and the camera's intrinsics, not measured. [How the
node works](docs/design.md) describes the message, the four steps per frame and what changes without depth.

## Requirements

| What | Requirement |
| --- | --- |
| GPU | an NVIDIA GPU with driver 580 or newer (PyTorch for CUDA 13) |
| VRAM | 7.5 GiB with one person in view (the node alone, 1280 x 720 at 30 frames per second, measured 2026-11-02) |
| Durchblick | 0.5 or newer, below 0.6 (Python package `xblick`) |
| Python | 3.12, which uv installs with the node |
| Tools for the install | git and uv (step 1 of the installation installs them) |
| Disk | 3.5 GB for the GEM-X weights, plus the SOMA assets and the detector |

Model licences:

| Part | Licence |
| --- | --- |
| GEM-X code | Apache-2.0. Some hand files are under OpenMDW-1.1, and the node does not use them. |
| GEM-X Body Pose weights | NVIDIA Open Model License |
| SOMA body assets | Apache-2.0 |
| YOLOX-X HumanArt person detector | Published by OpenMMLab, who state no licence for the model file. It comes from OpenMMLab's download and is separate from GEM-X, as in GEM-X's own instructions. It is trained on COCO and Human-Art, and Human-Art is offered for non-commercial purposes only: check whether you may use the detector for your purpose. |

The install fetches the GEM-X code, and the node downloads the three others. This repository contains no copy of them.

## Install

1. Install the prerequisites. `nvidia-smi` prints the GPU and the driver version, which must be 580 or newer. On
   Ubuntu, `sudo apt install nvidia-driver-580` and a restart install that driver. After the installation of uv, open
   a new terminal, so that `uv` is on `PATH`.

   ```bash
   nvidia-smi
   sudo apt install git
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

2. Install the node and download its weights:

   ```bash
   uv tool install git+https://github.com/xblick/xblick-gem-x@v0.2.8
   xblick-gem-x --download-only
   ```

   uv puts `xblick-gem-x` on `PATH`, in an environment of its own, with PyTorch for CUDA 13 (about 3 GB to download,
   7 GB on disk), Python 3.12 and the GEM-X code at the commit that `pyproject.toml` fixes. It resolves the other
   dependencies afresh within the bounds of `pyproject.toml`: `uv.lock` applies only to a clone.
   `xblick-gem-x --download-only` prints the models' terms and downloads the weights into `~/.cache/xblick/gem-x/`:
   GEM-X's Body Pose (3.5 GB), SOMA's body assets and the person detector, each with its progress.

   `uv tool upgrade xblick-gem-x` updates the dependencies within the bounds of the installed version. A newer
   version is the same install command with its tag. `uv tool uninstall xblick-gem-x` removes the node. The weights
   stay in the cache directory.

3. Install the core's command `xblick` with the viewer, video files and webcams. uv puts it on `PATH`. A second
   `uv tool install` of xblick with other extras replaces the first and removes what only the first had, so every
   xblick node names this one line:

   ```bash
   uv tool install "xblick[rerun,video,cam]>=0.5,<0.6"
   ```

4. Start the node on a video, without a camera. Take a short video of a person, whole in the picture, for example
   from a phone. Run each command in its own terminal. If the PC already runs a Zenoh router (`xblick router` or
   `zenohd`), leave out the first one.

   ```bash
   xblick router                                               # the Zenoh router of this PC
   xblick play ~/Videos/clip.mp4 --camera demo/video --loop    # the video as the camera demo/video
   xblick-gem-x --camera demo/video --no-depth
   xblick rerun                                                # the viewer
   ```

   The viewer shows the video with each person's skeleton over it, and the skeletons in its 3D view. `--no-depth`
   makes the node work on the colour stream alone: the distances of the bodies are estimates from the image and
   from the intrinsics, which `xblick play` takes from an assumed field of view of 60 degrees. The node's start
   line and every message say so (`"depth": "none"`).

5. With a camera that has a depth stream in the geometry of its colour camera, the node fits each body to the
   measured depth (the full mode). [xblick-azure-kinect](https://github.com/xblick/xblick-azure-kinect) publishes
   such a camera, `cam/azurekinect` with the depth `depth_to_color`:

   ```bash
   xblick-gem-x --camera cam/azurekinect:depth=depth_to_color
   xblick rerun
   ```

### From a clone

For development, and for the versions that `uv.lock` fixes:

```bash
git clone https://github.com/xblick/xblick-gem-x.git
cd xblick-gem-x
tools/setup.sh
```

When git or uv is missing, the script stops and prints the command that installs it. Without an NVIDIA GPU with
driver 580 or newer it says so and goes on. `tools/setup.sh --help` describes it. In a clone the commands are in
`.venv/bin/`.

The script creates `.venv` exactly as `uv.lock` states. Then it runs `xblick-gem-x --download-only`, which
downloads the weights into `~/.cache/xblick/gem-x/`. It downloads only what is missing, so you can run it again after an update.

### Downloads at the start

The node itself also downloads missing weights at its start, before it loads the models. Before a download it
prints the models' terms from the table above, the files and their size. A preparation script downloads without a
start with `xblick-gem-x --download-only` (`.venv/bin/xblick-gem-x` in a clone).

`--download-only` needs no camera, no Zenoh router and no GPU, and it exits with 0 when the weights are complete. A
machine that must never start a download on its own, such as a robot, starts the node with `--no-download`. The
node then ends at once when a file is missing, and it names the file.

### Where the weights are

The node keeps its weights in its cache directory, `~/.cache/xblick/gem-x/`. `XBLICK_HOME=/dir` moves it to `/dir/cache/gem-x/`,
and `XDG_CACHE_HOME` moves it as well. `--weights-dir DIR` names another directory. The place is the same for an
install with uv and for a clone. [Where files are](https://github.com/xblick/xblick/blob/main/docs/reference/files.md) lists the places, and it shows how to copy the cache to a
PC without internet.

Up to version 0.2.3, the weights were in the checkout's `weights/`. The node does not read them there. At its start it
prints the command that moves them. In the checkout it is:

```bash
mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/gem-x
```

## Run

Start a Zenoh router (`xblick router` or `zenohd`) on the PC, then start the node for one camera. The camera
names below are examples.
In a clone the commands are in `.venv/bin/`.

```bash
xblick-gem-x --camera cam/azurekinect:depth=depth_to_color
```

A camera specification has the form `CAMERA[:color=NAME][:depth=NAME]`. Repeat `--camera` for several cameras. The
second example reads a stereo camera whose depth comes from xblick-fast-foundationstereo, and it publishes the
bodies on the topic of the Azure Kinect tracker.

```bash
xblick-gem-x --camera cam/rc_visard:color=left:depth=ffs/depth --bodies-key "{camera}/bodies"
```

With `--bodies-key "{camera}/bodies"` the node replaces the Azure Kinect tracker for the readers of that topic. Run
only one of the two on a camera.

| Option | Default | Meaning |
| --- | --- | --- |
| `--bodies-key KEY` | `{camera}/{node}/bodies` | the topic of the bodies. The node fills in `{camera}` and `{node}`. |
| `--max-hz HZ` | 20 | the most colour frames per second that the node processes. `0` or `none` means every frame. |
| `--max-offset-ms MS` | 20 | the largest stamp difference between a colour frame and its depth frame. A colour frame without a depth frame that close is skipped and counted. `0` accepts only equal stamps. |
| `--no-depth-fit` | off | publish the distance that GEM-X estimates from the image alone. The node still reads and waits for the depth stream. |
| `--no-depth` | off | colour only: read no depth stream (a camera is `CAMERA[:color=NAME]`). The distances are GEM-X's estimate, never fitted, and the message has `"depth": "none"`. Excludes `--no-depth-fit`. |
| `--min-height PX` | 60 | ignore persons whose box is lower than this. Live: it changes while the node runs. |
| `--max-persons N` | 6 | process only the N largest persons of a frame. Live: it changes while the node runs. |
| `--local` | `auto` | how frames of a camera on this host reach the node: `auto`, `shm` or `cuda` |
| `--no-download` | off | end at once when a weight file is missing. The node downloads nothing. |
| `--download-only` | off | download the missing weights, check them and end. `--camera` may be left out. |
| `--weights-dir DIR` | the cache directory | where the weights are and go, in place of `~/.cache/xblick/gem-x/` |

`xblick-gem-x --help` lists all options, and [How the node works](docs/design.md) explains them.

Every option is also a parameter of the node `gem_x`, which Durchblick's terminal interface (`xblick`, tab Parameters) and `xblick param` show. `min_score`, `min_height`, `max_persons` and `gate` change while the node runs. The other options apply at the next start, where `--param NAME:=VALUE` or `--params-file FILE` set them. `--download-only` is a mode of the command line and has no parameter.

## Why it is fast

The node needs 59 to 60 ms per frame with two persons on an RTX PRO 6000.

| Change | Effect |
| --- | --- |
| GEM-X computes in bfloat16, a 16-bit number format that GPUs compute faster than 32-bit floats. | 30 ms per person, against 43 ms in float32 |
| The person detector YOLOX executes on the GPU through the CUDA build of ONNX Runtime. | 16 ms per frame, against 200 to 450 ms on the CPU |
| GEM-X processes all persons of a frame in one batch. | one model call per frame |
| The node decodes a depth frame only for a colour frame that it processes. | no work for frames above `--max-hz` |

[Accuracy and speed](docs/performance.md) has the measurements and their setup.

## Limits

- GEM-X's own accuracy is the published one: 62 mm MPJPE on 3DPW and 68 mm on EMDB.
- The node publishes no joint orientations. The field `orientation_wxyz` is null.
- The node does not estimate hands. GEM-X's Hand Pose needs separate weights, and their owner gates the download.
- The node does not use Body Motion, GEM-X's temporal model that works in world coordinates.
- An id ends when its person is missing for more than 10 frames. The node does not reuse ids.

## Documentation

- [How the node works](docs/design.md): the four steps per frame, the bodies message, the options, the tests
- [Accuracy and speed](docs/performance.md): the synthetic test and the timings
- Durchblick: [streams and topics](https://github.com/xblick/xblick/blob/main/docs/concepts/streams-and-topics.md),
  [pairing streams](https://github.com/xblick/xblick/blob/main/docs/guides/pairing-streams.md),
  [transports](https://github.com/xblick/xblick/blob/main/docs/concepts/transports.md),
  [timestamps](https://github.com/xblick/xblick/blob/main/docs/concepts/timestamps.md),
  [sessions and routers](https://github.com/xblick/xblick/blob/main/docs/concepts/sessions.md)

## Licence and credits

GEM-X and SOMA are by NVIDIA ([NVlabs/GEM-X](https://github.com/NVlabs/GEM-X)). The person detector YOLOX-X HumanArt
is by OpenMMLab. "Requirements" lists their licences.

The code of this repository is under the Apache License 2.0, see [LICENSE](LICENSE). [NOTICE](NOTICE) names the
copyright holder and the parts of others.

## Citation

This node uses GEM-X (NVlabs/GEM-X, nvidia/GEM-X on Hugging Face). The GEM-X repository and its model card state no
citation. The model card names [https://arxiv.org/abs/2505.01425](https://arxiv.org/abs/2505.01425) as its reference.

The bodies are those of the body model SOMA (NVlabs/SOMA-X). If you use the node in research, cite SOMA. The entries
below are copied from [its README](https://github.com/NVlabs/SOMA-X/blob/v0.3.3/README.md):

```bibtex
@article{soma2026,
  title={SOMA: Unifying Parametric Human Body Models},
  author={Jun Saito and Jiefeng Li and Michael de Ruyter and Miguel Guerrero and Edy Lim and Ehsan Hassani and Roger Blanco Ribera and Hyejin Moon and Magdalena Dadela and Marco Di Lucca and Qiao Wang and Xueting Li and Sam Wu and Chaeyeon Chung and Yeongho Seol and Jan Kautz and Simon Yuen and Umar Iqbal},
  eprint={2603.16858},
  archivePrefix={arXiv},
  year={2026},
  url={https://arxiv.org/abs/2603.16858},
}
```

The person detector is a YOLOX model of OpenMMLab's RTMPose project. Cite YOLOX The entries below are copied from [its
README](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.1.1rc0/README.md):

```bibtex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

and the RTMPose project, whose detector file the node downloads. The entries below are copied from [its
README](https://github.com/open-mmlab/mmpose/blob/v1.3.2/projects/rtmpose/README.md):

```bibtex
@misc{https://doi.org/10.48550/arxiv.2303.07399,
  doi = {10.48550/ARXIV.2303.07399},
  url = {https://arxiv.org/abs/2303.07399},
  author = {Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}

@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
