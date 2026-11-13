# Changelog

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
