# xblick-gem-x

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)

This [Durchblick](https://github.com/xblick/xblick) node computes the 3D bodies of the people in
front of a colour camera with depth. It takes about 60 ms per frame with two persons, so it keeps up with a camera at
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
| `<camera>/gem_x/bodies` | JSON, the layout of xblick-azure-kinect's bodies | model `K4ABT_BODY_32`: per body an id and 32 joints in mm in the frame of the colour camera. Each body also has `soma77_mm` with all 77 SOMA joints and `depth_fit` with the result of the depth fit. |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | status `gem_x` once per second: frames per second, bodies per frame, ms per frame, the streams in use |

`--bodies-key` changes the topic of the bodies, and `--node` changes the `gem_x` part. `xblick rerun` draws the
bodies as skeletons in 3D and over the colour stream.

The node reads `<camera>/<color>` and `<camera>/<depth>`. The depth must be 16UC1 in mm in the geometry of the colour
camera. [How the node works](docs/design.md) describes the message and the four steps per frame.

## Requirements

| What | Requirement |
| --- | --- |
| GPU | an NVIDIA GPU with CUDA. We tested on an RTX PRO 6000. |
| VRAM | 7.5 GiB with one person in view (the node alone, 1280 x 720 at 30 frames per second, measured 2026-11-02) |
| Durchblick | 0.5 or newer, below 0.6 (Python package `xblick`) |
| Python | 3.12, which `tools/setup.sh` installs into `.venv` |
| Tools for the setup | uv and git |
| Disk | 3.5 GB for the GEM-X weights, plus the SOMA assets and the detector |

Model licences:

| Part | Licence |
| --- | --- |
| GEM-X code | Apache-2.0. Some hand files are under OpenMDW-1.1, and the node does not use them. |
| GEM-X Body Pose weights | NVIDIA Open Model License |
| SOMA body assets | Apache-2.0 |
| YOLOX-X HumanArt person detector | Published by OpenMMLab, who state no licence for the model file. It comes from OpenMMLab's download and is separate from GEM-X, as in GEM-X's own instructions. It is trained on COCO and Human-Art, and Human-Art is offered for non-commercial purposes only: check whether you may use the detector for your purpose. |

The setup fetches the GEM-X code, and the node downloads the three others. This repository contains no copy of them.

## Install

```bash
tools/setup.sh
```

The script creates `.venv` exactly as `uv.lock` states. Then it runs `xblick-gem-x --download-only`, which
downloads the weights into `~/.cache/xblick/gem-x/`. It downloads only what is missing, so you can run it again after an update.

The node itself also downloads missing weights at its start, before it loads the models. Before a download it
prints the models' terms from the table above, the files and their size. A preparation script downloads without a
start:

```bash
.venv/bin/xblick-gem-x --download-only
```

`--download-only` needs no camera, no Zenoh router and no GPU, and it exits with 0 when the weights are complete. A
machine that must never start a download on its own, such as a robot, starts the node with `--no-download`. The
node then ends at once when a file is missing, and it names the file.

### Where the weights are

The node keeps its weights in its cache directory, `~/.cache/xblick/gem-x/`. `XBLICK_HOME=/dir` moves it to `/dir/cache/gem-x/`,
and `XDG_CACHE_HOME` moves it as well. `--weights-dir DIR` names another directory. The place is the same for a
checkout and for a pip install. [Where files are](https://github.com/xblick/xblick/blob/main/docs/reference/files.md) lists the places, and it shows how to copy the cache to a
PC without internet.

Up to version 0.2.3, the weights were in the checkout's `weights/`. The node does not read them there. At its start it
prints the command that moves them. In the checkout it is:

```bash
mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/gem-x
```

## Run

Start a Zenoh router (`zenohd`) on the PC, then start the node for one camera. The camera names below are examples.

```bash
.venv/bin/xblick-gem-x --camera cam/azurekinect:depth=depth_to_color
```

A camera specification has the form `CAMERA[:color=NAME][:depth=NAME]`. Repeat `--camera` for several cameras. The
second example reads a stereo camera whose depth comes from xblick-fast-foundationstereo, and it publishes the
bodies on the topic of the Azure Kinect tracker.

```bash
.venv/bin/xblick-gem-x --camera cam/rc_visard:color=left:depth=ffs/depth --bodies-key "{camera}/bodies"
```

With `--bodies-key "{camera}/bodies"` the node replaces the Azure Kinect tracker for the readers of that topic. Run
only one of the two on a camera.

| Option | Default | Meaning |
| --- | --- | --- |
| `--bodies-key KEY` | `{camera}/{node}/bodies` | the topic of the bodies. The node fills in `{camera}` and `{node}`. |
| `--max-hz HZ` | 20 | the most colour frames per second that the node processes. `0` or `none` means every frame. |
| `--max-offset-ms MS` | half the frame period of the depth stream | the largest stamp difference between a colour frame and its depth frame. `0` accepts only equal stamps. |
| `--no-depth-fit` | off | publish the distance that GEM-X estimates from the image alone |
| `--min-height PX` | 60 | ignore persons whose box is lower than this. Live: it changes while the node runs. |
| `--max-persons N` | 6 | process only the N largest persons of a frame. Live: it changes while the node runs. |
| `--local` | `auto` | how frames of a camera on this host reach the node: `auto`, `shm` or `cuda` |
| `--no-download` | off | end at once when a weight file is missing. The node downloads nothing. |
| `--download-only` | off | download the missing weights, check them and end. `--camera` may be left out. |
| `--weights-dir DIR` | the cache directory | where the weights are and go, in place of `~/.cache/xblick/gem-x/` |

`.venv/bin/xblick-gem-x --help` lists all options, and [How the node works](docs/design.md) explains them.

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

- We tested the node only on a synthetic RGB-D recording. We have not tested it on a real camera or on a robot.
- We have not compared it with the Azure Kinect tracker on the same people.
- GEM-X's own accuracy is the published one: 62 mm MPJPE on 3DPW and 68 mm on EMDB. We have not measured how much the
  depth fit improves it on real people.
- The node publishes no joint orientations. The field `orientation_wxyz` is null.
- The node does not estimate hands. GEM-X's Hand Pose needs separate weights, and their owner gates the download.
- The node does not use Body Motion, GEM-X's temporal model that works in world coordinates.
- An id ends when its person is missing for more than 10 frames. The node does not reuse ids.
- We tested the download against a local server, and we fetched `body_config.json` and
  `SOMA_procedural_transforms.json` from Hugging Face with it. We have not downloaded the large files or the YOLOX
  archive through the node.

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
