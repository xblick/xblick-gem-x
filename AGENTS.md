# AGENTS.md

xblick-gem-x: NVIDIA's GEM-X Body Pose as a Durchblick node. 3D bodies of people from any colour camera with
depth, published in the Azure Kinect tracker's JSON. Version 0.2.4, tested only on a synthetic RGB-D recording.
Human documentation: `README.md` (topics, options, limits), `docs/design.md` (steps, message, setup),
`docs/performance.md`. Transport-wide traps (Zenoh sessions, CUDA rings, looped playback, hanging
tests) are in the core's `AGENTS.md` (github.com/xblick/xblick). Style of all documentation: the core's
`docs/project/writing-docs.md`.

## Commands

```bash
tools/setup.sh                                   # uv sync --frozen, then xblick-gem-x --download-only
.venv/bin/python -m pytest tests                 # test_gemx.py needs a CUDA device and the weights in ~/.cache/xblick/gem-x
.venv/bin/python -m pytest tests/test_weights.py # CPU, 4 s: the node's files and its start up to the weights (the download: the core's tests)
.venv/bin/xblick-gem-x --download-only           # GEM-X/SOMA (fixed revisions), YOLOX (archive SHA-256) if missing
.venv/bin/python -m pytest tests/test_helpers.py # CPU: tracker, skeleton mapping, SOMA rig check, depth fit
.venv/bin/ruff check src tests                   # lint; 120 columns, rules in pyproject.toml
.venv/bin/xblick-gem-x --camera cam/azurekinect:depth=depth_to_color   # needs zenohd on the PC
```

- Console script: `xblick-gem-x`.
- `tools/setup.sh` needs uv (`UV` names it) and git. `--download-only` downloads only what is missing (3.5 GB of
  GEM-X weights on the first call). Before a download it prints the models' terms (`weights.TERMS`).
- `--no-download`: a start with a missing file ends with one line (exit 1). Excludes `--download-only`.
- Durations: not recorded.
- Without a camera: `xblick play <recording.mcap> --loop` of an RGB-D recording with people. None existed when the
  node was written; the test in `docs/performance.md` is synthetic.
- rc_visard example: `--camera cam/rc_visard:color=left:depth=ffs/depth` (needs xblick-fast-foundationstereo).
- `xblick rerun` draws the bodies in 3D and over the colour stream.

## Rules

- Files: the weights are in the node's cache directory (`xblick.paths.cache_dir("gem-x")`, `~/.cache/xblick/gem-x`), never in
  the checkout. `weights.STORE = Store(PROG, node=NODE, ..., former=[FORMER])`, `Weight` paths relative to its root;
  `store = weights.store_for(parser, args)` (`--weights-dir`) for `download_only`, `ensure` and `fail`, with
  `options=weights.options(args)`; model code takes its paths from `store.root`. `FORMER` (the checkout's `weights/`) is never
  read: the store prints the command that moves it. The core's `docs/reference/files.md` has the rule. Warp's kernel cache goes there too (`warp-cache`).
- Fail fast. A SOMA rig that is not the expected one (`skeleton.check_soma77`), a camera spec with a
  segmenter: raise with a message that says what to do.
- Work offline once the weights are there. `gem` is pinned to a git commit in `pyproject.toml`. Weights come at fixed
  revisions (`gemx.GEMX_REVISION`, `SOMA_REVISION`, the file lists in `gemx.py`) or by SHA-256 (YOLOX's archive,
  `weights.py`). `weights.STORE.ensure` runs at the start before the models load; `--download-only` runs it alone. All
  files there: no network, no file written (`test_weights.py` checks it). Fetch nothing while working.
- Downloads: `<name>.part` in the target directory, renamed after the check, under the lock
  `~/.cache/xblick/gem-x/.download.lock`. Never let a partial or unchecked file get its final name. No fallbacks.
- Treat the bodies JSON as a contract. Its layout is xblick-azure-kinect's (`skeleton.py`). Readers (the core's
  Rerun bridge, other readers of the Azure Kinect bodies) take either. Change it only together with them.
- Durchblick pin: `xblick>=0.5.5,<0.6`, the core's Python package (`xblick.rgbd` with `max_hz` and `max_offset_ns`,
  `xblick.node`, `CountedPublisher` for the attachment).
- 120 columns. Docstrings say what, comments say why.
- Update `README.md` and `docs/` with every change of behaviour.
- Run GPU tests and live checks in the background with `timeout`. Stop them by PID.

## Traps

| Trap | What happens | Fix |
| --- | --- | --- |
| Two CUDA stacks in one process | torch is cu130. ONNX Runtime brings CUDA 12 and cuDNN as `nvidia-*` wheels. | `PersonDetector` calls `onnxruntime.preload_dlls()` before it creates its session. Keep that order. |
| ONNX Runtime notice "no registered plugin EP device" | appears for CUDA | Harmless. The logger is set to errors only. |
| YOLOX takes BGR | GEM-X's predictor takes RGB, the detector does not | `PersonDetector.detect` flips the RGB frame. Keep the flip. |
| Warp kernel cache | SOMA's skinning compiles Warp kernels. Warp's default cache is in the home directory. | `WARP_CACHE_PATH` defaults to `warp-cache` in the node's cache directory (`GemX(weights=...)`). Set it before GEM-X is imported. |
| Skeleton mapping | SOMA has 77 joints and no nose or ears, readers expect the Kinect's 32 | Mapped from SOMA by anatomy. Nose and ears are derived from head and eyes, confidence 1. Orientations are null. |
| Depth fit keeps the size | GEM-X's distance comes from its learned body size | By design. The fit only moves the body along the hips ray (median difference, then again within `--gate` 8 cm). `--no-depth-fit` keeps the monocular distance. |
| Depth geometry | the fit compares depth and mesh per pixel of the colour camera | Depth must be 16UC1 mm in the colour camera's geometry (Azure Kinect: `depth_to_color`). |
| `--bodies-key {camera}/bodies` | the node publishes where the Azure Kinect tracker publishes | Run one of the two per camera, never both. |
| Ids | a person returns with a new id | Ids end after 10 frames without their person and are not reused. |
| `--download-only` and `--camera` | `--camera` is required for a start only. `xblick.weights.parse_args` lifts `required` when the token `--download-only` is spelled out (an abbreviation is not seen). | Spell it out. |
| `download_only` as a parameter | `declare_options` would declare it. | It is in `leave_out`. `no_download` is a read-only parameter. |
| Wrong SHA-256 of the YOLOX archive | `yolox_humanart.onnx.part.zip` stays, the error names it. | Remove it, start again. |
| Interrupt during a download | Ctrl-C: exit 130, SIGTERM: exit 143 (`xblick.weights` overrides `node.start`'s handler while downloading). The `.part` file is removed. | |
| Zenoh sessions (core) | a session with only `--connect` does not listen, and the router does not route between such sessions | Start nodes without endpoints. They find the PC's router. |

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_gem_x/node.py` | the node: cameras, per-frame pipeline, publishing, diagnostics |
| `src/xblick_gem_x/weights.py` | `needed`, YOLOX's archive (`ZipMember`), `TERMS`, `STORE` (the core's `xblick.weights.Store`: the download, `--download-only`/`--no-download`) |
| `src/xblick_gem_x/gemx.py` | `GemX` (Body Pose, weights and revisions), `PersonDetector` (YOLOX on ONNX Runtime) |
| `src/xblick_gem_x/tracking.py` | person ids by box overlap |
| `src/xblick_gem_x/depth.py` | the depth fit along the hips ray |
| `src/xblick_gem_x/skeleton.py` | SOMA 77 -> K4ABT_BODY_32, the bodies JSON |
| `~/.cache/xblick/gem-x/` | GEM-X, `soma/`, `yolox_humanart.onnx` (downloaded by the node), `.download.lock`, `warp-cache/` (Warp): the node's cache directory. The checkout's `weights/` is the former place. |
| `tests/` | `test_helpers.py`, `test_weights.py` (CPU), `test_gemx.py` (GPU) |
| `tools/setup.sh` | venv, then `xblick-gem-x --download-only` |

## Defaults

| What | Value |
| --- | --- |
| Inputs | `<camera>/<color>`, `<camera>/<depth>` (16UC1 mm) through `xblick.rgbd` |
| Bodies topic | `<camera>/gem_x/bodies` (`--bodies-key {camera}/{node}/bodies`), JSON, `source` `gem-x` |
| Diagnostics | `<camera>/diagnostics`, status `gem_x`, once per second |
| `--max-hz` | 20 |
| `--max-offset-ms` | half the depth's frame period |
| `--local` | `auto` (`shm`, `cuda`) |
| `--autocast` | `bfloat16` (`none`) |
| `--detector-device` | `cuda` (`cpu`) |
| `--min-score`, `--min-height`, `--max-persons` | 0.5, 60 px, 6 |
| `--gate`, `--history` | 0.08 m, 10 |
| Tracker | `min_iou` 0.3, `max_missing` 10 |
| Depth fit | `min_pixels` 100 |
| Environment | `UV` (setup), `WARP_CACHE_PATH` (default `~/.cache/xblick/gem-x/warp-cache`), `XBLICK_HOME`, `XDG_CACHE_HOME` (cache directory) |
