# AGENTS.md

xblick-gem-x: NVIDIA's GEM-X Body Pose as a Durchblick node. 3D bodies of people from any colour camera with
depth, published in the Azure Kinect tracker's JSON. Version 0.2.0, tested only on a synthetic RGB-D recording.
Human documentation: `README.md` (topics, options, limits), `docs/design.md` (steps, message, setup),
`docs/performance.md`. Transport-wide traps (Zenoh sessions, CUDA rings, looped playback, hanging
tests) are in the core's `AGENTS.md` (github.com/xblick/xblick). Style of all documentation: the core's
`docs/project/writing-docs.md`.

## Commands

```bash
tools/setup.sh                                   # uv sync --frozen, GEM-X/SOMA weights (fixed revisions), YOLOX
.venv/bin/python -m pytest tests                 # 8 tests; test_gemx.py needs a CUDA device and weights/
.venv/bin/python -m pytest tests/test_helpers.py # CPU: tracker, skeleton mapping, SOMA rig check, depth fit
.venv/bin/ruff check src tests                   # lint; 120 columns, rules in pyproject.toml
.venv/bin/xblick-gem-x --camera robot/camera:depth=depth_to_color   # needs zenohd on the PC
```

- Console script: `xblick-gem-x`.
- `tools/setup.sh` needs uv (`UV` names it), git, curl, unzip. It downloads only what is
  missing (3.5 GB of GEM-X weights on the first call). Before a download it names the models' terms and asks; without
  a terminal it needs `--accept-model-terms` or `XBLICK_GEM_X_ACCEPT_MODEL_TERMS=yes`, which only the user may
  decide to give: never pass them on your own.
- Durations: not recorded.
- Without a camera: `xblick play <recording.mcap> --loop` of an RGB-D recording with people. None existed when the
  node was written; the test in `docs/performance.md` is synthetic.
- rc_visard example: `--camera robot/rc_visard:color=left:depth=ffs/depth` (needs xblick-fast-foundationstereo).
- `xblick rerun` draws the bodies in 3D and over the colour stream.

## Rules

- Fail fast. Missing weights, a SOMA rig that is not the expected one (`skeleton.check_soma77`), a camera spec with a
  segmenter: raise with a message that says what to do.
- Work offline. `gem` is pinned to a git commit in `pyproject.toml`. Weights come at fixed revisions
  (`gemx.GEMX_REVISION`, `SOMA_REVISION`) or by SHA-256 (YOLOX) through `tools/setup.sh`. Fetch nothing at run time.
- Treat the bodies JSON as a contract. Its layout is xblick-azure-kinect's (`skeleton.py`). Readers (the core's
  Rerun bridge, other readers of the Azure Kinect bodies) take either. Change it only together with them.
- Durchblick pin: `xblick>=0.5.0,<0.6`, the core's Python package (`xblick.rgbd` with `max_hz` and `max_offset_ns`,
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
| Warp kernel cache | SOMA's skinning compiles Warp kernels. Warp's default cache is in the home directory. | `WARP_CACHE_PATH` defaults to `weights/warp-cache`. Set it before GEM-X is imported. |
| Skeleton mapping | SOMA has 77 joints and no nose or ears, readers expect the Kinect's 32 | Mapped from SOMA by anatomy. Nose and ears are derived from head and eyes, confidence 1. Orientations are null. |
| Depth fit keeps the size | GEM-X's distance comes from its learned body size | By design. The fit only moves the body along the hips ray (median difference, then again within `--gate` 8 cm). `--no-depth-fit` keeps the monocular distance. |
| Depth geometry | the fit compares depth and mesh per pixel of the colour camera | Depth must be 16UC1 mm in the colour camera's geometry (Azure Kinect: `depth_to_color`). |
| `--bodies-key {camera}/bodies` | the node publishes where the Azure Kinect tracker publishes | Run one of the two per camera, never both. |
| Ids | a person returns with a new id | Ids end after 10 frames without their person and are not reused. |
| Zenoh sessions (core) | a session with only `--connect` does not listen, and the router does not route between such sessions | Start nodes without endpoints. They find the PC's router. |

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_gem_x/node.py` | the node: cameras, per-frame pipeline, publishing, diagnostics |
| `src/xblick_gem_x/gemx.py` | `GemX` (Body Pose, weights and revisions), `PersonDetector` (YOLOX on ONNX Runtime) |
| `src/xblick_gem_x/tracking.py` | person ids by box overlap |
| `src/xblick_gem_x/depth.py` | the depth fit along the hips ray |
| `src/xblick_gem_x/skeleton.py` | SOMA 77 -> K4ABT_BODY_32, the bodies JSON |
| `weights/` | GEM-X, `soma/`, `yolox_humanart.onnx` (`tools/setup.sh`), `warp-cache/` (Warp). Not in git. |
| `tests/` | `test_helpers.py` (CPU), `test_gemx.py` (GPU) |
| `tools/setup.sh` | venv and weights |

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
| Environment | `UV` (setup), `WARP_CACHE_PATH` (default `weights/warp-cache`) |
