# supreme-screw-sortering — screw-synth

Synthetic data and models for detecting, classifying and **measuring** metric fasteners (screws, nuts,
washers, T-nuts) lying on the bed of a converted Ender 3, photographed by a fixed overhead Raspberry Pi
Camera Module 3.

Photorealistic training images are rendered with Blender through [BlenderProc](https://github.com/DLR-RM/BlenderProc),
labelled automatically from the render ground truth, and used to train a two-stage YOLO pipeline:

1. **Stage 1:** a segmentation model on the 1024 px downscaled region of interest (ROI) detects each part and
   classifies its family and size (e.g. `DIN912_M4`).
2. **Stage 2:** measurement on full-resolution crops. Keypoints and sub-pixel edges are parallax-corrected and
   snapped to the nearest standard length.

Project conventions are in [CLAUDE.md](CLAUDE.md). The build guide, with one prompt per tool, is in [docs/prompts/](docs/prompts/).

## Pipeline

| #  | Tool        | Purpose | Status |
|----|-------------|---------|--------|
| 01 | catalogue   | `catalogue.yaml` of every part + validator + CLI to add parts | done |
| 02 | partgen     | parametric mesh generator: render mesh, collision proxy, keypoints, `part.json` | done |
| 03 | calibration | Pi capture scripts, ChArUco board, intrinsics, extrinsics, sensor profile | TODO |
| 04 | restposes   | stable resting poses of each part (precomputed once) | TODO |
| 05 | scenegen    | BlenderProc scene generation and rendering (RGB + instance mask + `meta.json`) | TODO |
| 06 | postprocess | lens distortion (RGB and mask), noise, vignetting, compression | TODO |
| 07 | labels      | export to YOLO seg/detect/pose/OBB and COCO, splits, preview overlays | TODO |
| 08 | train       | train stage 1 and stage 2, measurement, inference, evaluation | TODO |
| 09 | realdata    | capture and label a real validation set, compare real vs synthetic | TODO |

Each tool has code in `src/screwsynth/<tool>/` and docs in `tools/<tool>/README.md`.

## Install

You need Python 3.10 or newer. An NVIDIA GPU is recommended for rendering and training.

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
# add the groups you need, e.g.
pip install -e ".[catalogue,partgen,calib,train,dev]"
```

**Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[catalogue,partgen,calib,train,dev]"
```

**BlenderProc** is installed separately and is *not* a dependency of the package:

```bash
pip install blenderproc
blenderproc quickstart   # first run downloads its own Blender
```

The optional dependency groups are:

| Group | For tools |
|-------|-----------|
| `catalogue` | 01 |
| `partgen` | 02, 04 |
| `calib` | 03, 06, 07, 09 |
| `train` | 08 |
| `dev` | tests, linting |

The Raspberry Pi capture scripts have their own requirements (`tools/calibration/pi/requirements.txt`, added by tool 03).

## Development

```bash
pytest                 # all tests
pytest -m "not slow"   # skip Blender/GPU tests
ruff check . && ruff format --check .
pre-commit install
```

## Order of operations

The tools run in this order. Calibration (03) can be done in parallel with 01, 02 and 04. Until it exists,
`scenegen --nominal-camera` uses nominal Pi Camera 3 values.

1. **catalogue**: define and validate the parts, then generate `configs/classes.yaml`.
2. **partgen**: build meshes, collision proxies and keypoints into `data/assets/<part_id>/`.
3. **calibration**: on the rig, lock the camera settings, calibrate intrinsics and extrinsics, and build the sensor profile in `data/calibration/`.
4. **restposes**: compute stable poses into `data/rest_poses/`.
5. **scenegen**: render scenes with `blenderproc run tools/scenegen/generate.py ...` into `output/<run>/scenes/`.
6. **postprocess**: apply distortion, sensor noise and compression into `output/<run>/processed/`.
7. **labels**: export YOLO and COCO datasets and measurement crops into `datasets/<name>/`.
8. **train**: train stage 1 and stage 2, then evaluate and run inference.
9. **realdata**: capture and label a real validation set, then produce the sim-to-real report.

End-to-end dataset production:
`catalogue -> partgen -> restposes -> scenegen -> postprocess -> labels -> train`.
