# Changelog

## 0.2.7 (2026-11-19)

An installation that a beginner can follow from the README alone.

### Changed

- The README's installation is a numbered path: the prerequisites with their install commands, the node with
  `uv tool install`, the core's command with one line for every node
  (`uv tool install "xblick[rerun,video,cam]>=0.5,<0.6"`), and a first start. The clone with `tools/setup.sh` stays
  the second way.
- `tools/setup.sh --help` describes the script and its options and ends. Before, it ran the setup.
- `tools/setup.sh` checks its prerequisites first. Without git or uv it stops and prints the command that installs
  them. On a PC other than Linux on x86-64, for which the core has no wheels, it stops at once.
- Before its downloads, `tools/setup.sh` lists what it downloads, with the sizes and the licences.
- `tools/setup.sh` says when it finds no NVIDIA GPU with driver 580 or newer, which the node needs to start, and
  goes on (`--download-only` needs no GPU). It refuses an unknown option; before, it ignored every option.

## 0.2.6 (2026-11-18)

The Durchblick core 0.5.14.

### Changed

- The node needs `xblick>=0.5.14,<0.6`; uv.lock fixes xblick 0.5.14 (was 0.5.5). No other package changed.
- The node declares itself on the network (core 0.5.9): its name, version, host, process and whole command line on
  `xblick/sessions/<zid>/program`. The terminal interface lists it in its Nodes and Graph tabs.
- The start line says where the node reads its frames, for example `reading the host ring: shm` (core 0.5.8).
- A `--camera` that names a stream (`cam/video2/color`) ends the node with one line that names the camera, instead of
  waiting (core 0.5.7). A router that cannot be reached ends it with one line instead of a traceback.
- A `--connect` to another PC's router in the default mode prints a notice with the three remedies (core 0.5.7).
- A parameter set whose deadline has passed when the node comes to act on it is refused, and the caller raises
  `Expired`. A set meant for another node on the same key is refused (core 0.5.13).
- The pairing default of the core (20 ms since 0.5.11) changes nothing here: `--max-offset-ms` was 20 ms already.
- `tests/test_weights.py`: its check for network use lets an address given as numbers resolve. The core looks up the
  hosts of `--connect` since 0.5.7, to tell whether they are this PC.

### Fixed

- A node that closed its session before its parameters no longer prints a traceback (core 0.5.7).
- Closing no longer raises "Already mutably borrowed" in the thread of the program's declaration (core 0.5.13).

Compatible with earlier clients: nothing on the wire changed.

## 0.2.5 (2026-11-15)

A fixed pairing offset, a fixed GPU, and the MPS and transport checks of the other nodes.

### Changed

- Changed default: `--max-offset-ms` is a fixed 20 ms, no longer half the depth stream's measured frame period. It pairs
  the colour and depth of one capture of the Azure Kinect (up to 17 ms apart) and of the rc_visard (equal stamps). A
  colour frame without a depth frame that close is skipped and counted, as before.
- `--device cuda` is GPU 0, `cuda:N` GPU N, no longer the current CUDA device. Another value, or a GPU that cannot be
  used, ends the node with one line.
- The start line says where GEM-X and YOLOX compute.
- README: how to cite GEM-X, SOMA and YOLOX.

### Fixed

- With `CUDA_MPS_PIPE_DIRECTORY` set the node stops at the start when no MPS daemon runs.
- `--local cuda` stops the node when a camera on this host offers no CUDA ring.

Compatible with earlier clients: nothing on the wire changed.

## 0.2.4 (2026-11-14)

- The weights moved out of the checkout into the node's cache directory, `~/.cache/xblick/gem-x/`. `XBLICK_HOME` and
  `XDG_CACHE_HOME` move it, and `--weights-dir DIR` names another directory. Weights at the former place, the
  checkout's `weights/`, are not used. The node prints the command that moves them. In the checkout it is
  `mkdir -p ~/.cache/xblick && mv weights ~/.cache/xblick/gem-x`.
- `--weights-dir DIR` is a new start option and a read-only parameter.
- The message about a missing file names the full download command.
- The node needs `xblick>=0.5.5,<0.6`.
- Warp's kernel cache (SOMA's skinning) is `warp-cache` in the same cache directory. The `mv` moves the old
  `weights/warp-cache` with the weights.
- `GemX` and `PersonDetector` take the weights' directory (`weights.STORE.root`, or `--weights-dir`'s).

## 0.2.3 (2026-11-14)

- The download of the weights is the core's `xblick.weights` (xblick 0.5.3). The node's own copy of that code is
  gone. `weights.py` keeps the node's files, its terms notice and its `STORE`. The options, the messages, and the
  files and their places stay as they were.
- The node needs `xblick>=0.5.3,<0.6`.
- `tests/test_weights.py` keeps the node's own checks. The core's tests cover the download itself.

## 0.2.2 (2026-11-14)

- The node downloads missing weights at its start, before the models load. It prints the models' terms, the files
  and their size first. With all weights there it uses no network and writes no file.
- `--download-only` downloads and checks the weights, then exits 0. It needs no camera, Zenoh router or GPU, and
  `--camera` may be left out. `tools/setup.sh` calls it after `uv sync` and no longer needs curl or unzip.
- `--no-download` ends a start at once when a weight file is missing and names it. It is a read-only parameter.
- `weights.py` is the one list of the weight files: the names and revisions of `gemx.py`, YOLOX's archive with its
  SHA-256 (moved from `tools/setup.sh`) and the models' terms (moved from `tools/setup.sh`, unchanged).
- A download goes to `<name>.part` under the lock `weights/.download.lock` and gets its final name after the check.
- Each Hugging Face file has its SHA-256 in `weights.py`: Hugging Face's `lfs.oid` at the fixed revision, or for the
  two small JSON files the hash of the file at that revision. A download is checked against it and against the exact
  size; `--download-only` also checks the files that exist. `yolox_humanart.onnx` stays covered by its archive's
  SHA-256.

## 0.2.1 (2026-11-14)

`tools/setup.sh` prints the models' terms before it downloads one. The README no longer names a licence for the YOLOX-X HumanArt person detector: OpenMMLab states none
for the model file, and one of its training datasets, Human-Art, is offered for non-commercial purposes only.

## 0.2.0 (2026-11-13)

Every name in code, on the wire and on disk is `xblick` now. Only the prose name of the project stays Durchblick.
The node needs the core `xblick` 0.5 and does not work with programs of Durchblick 0.4 (`blick`) or older: run every
program of a setup at 0.5.

| What | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Repository | github.com/xblick/durchblick-gem-x | github.com/xblick/xblick-gem-x |
| Distribution | `durchblick-gem-x` | `xblick-gem-x` |
| Python package | `durchblick_gem_x` | `xblick_gem_x` |
| Commands | `durchblick-gem-x` | `xblick-gem-x` |
| Program name in log lines, the logger and reports | `durchblick-gem-x` | `xblick-gem-x` |
| Core dependency | `blick>=0.4,<0.5`, git main of github.com/xblick/durchblick | `xblick>=0.5.0,<0.6`, github.com/xblick/xblick at 45908e6 |
| Core imports and command | `import blick`, `blick rerun`, `blick play`, `blick param` | `import xblick`, `xblick rerun`, `xblick play`, `xblick param` |
| Links to the core's docs | github.com/xblick/durchblick/blob/main/docs/... | github.com/xblick/xblick/blob/main/docs/... |
| Other programs named in the docs | `durchblick-azure-kinect`, its `durchblick-k4abt`, `durchblick-fast-foundationstereo` | `xblick-azure-kinect`, `xblick-k4abt`, `xblick-fast-foundationstereo` |

## 0.1.0

The node as `durchblick-gem-x` (Python package `durchblick_gem_x`) on the core `blick` 0.4.
