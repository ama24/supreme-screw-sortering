# CLAUDE.md — screw-synth

screw-synth is a toolkit that generates photorealistic synthetic training data with Blender (through
BlenderProc) for a YOLO model that detects, segments and measures metric fasteners lying on the bed
of a converted Ender 3, seen by a fixed overhead Raspberry Pi Camera Module 3.

The full build guide (one prompt per tool) lives in [docs/prompts/](docs/prompts/). Read the prompt for
the tool you are working on before changing it.

## Workflow
- Each tool is built from its own prompt (01 to 09), on its own branch `tool/<name>`, and merged
  through a pull request into `main`. Never push to `main` directly.
- Run `pytest` and `ruff check .` before every commit.

## Hardware
- Recycled Ender 3, bed area 220 x 220 mm. A 1.5-2 mm matte TPU plate (mid-tone saturated colour,
  e.g. blue or green) sits on the bed. The bed moves in Y, so it is homed/parked before every capture.
- Raspberry Pi Camera Module 3 (standard lens, IMX708, 4608 x 2592), fixed overhead at about 300 mm,
  autofocus LOCKED to a fixed lens position, exposure/gain/white balance LOCKED.
- About 11.7 px/mm at full resolution. The working area is a region of interest (ROI) of roughly
  2560 x 2560 px, defined in calibration.json.
- Diffuse LED lighting with a light shroud.
- Rendering/training PC: NVIDIA RTX 2070 Super, 8 GB VRAM.

## Objects (all are detection targets, each with its own class)
- Screws: DIN 912 (socket head cap), DIN 933 (hex head, fully threaded), ISO 7380 (button head),
  DIN 7991 (countersunk socket). Sizes M2, M2.5, M3, M4, M5, M6, M8, M10.
- Nuts, washers, slide-in T-nuts and drop-in T-nuts (for aluminium extrusion slots).
- Finishes: black oxide, zinc plated (clear/yellow), stainless A2/A4, plain steel, brass, nylon.
  NO aluminium and no anodised colours. Finish is a visual variation only, never a class.
  Magnetic vs non-magnetic is NOT distinguished.
- Arrangement: single layer, parts lying on the plate, may touch, no piles.

## Model design
- Class = family + size, e.g. DIN912_M4. Length is NOT a class; it is measured.
- Stage 1: YOLO segmentation model on the ROI downscaled to 1024 px (detect + segment + classify).
- Stage 2: measurement on full-resolution crops around each detection (keypoints and/or sub-pixel
  edges), parallax-corrected using known part geometry, snapped to the nearest standard length.
- Required accuracy: exact standard length (adjacent lengths may differ by only 1 mm for M2).

## Pipeline (one tool per step, each in tools/<name> and src/screwsynth/<name>)
| #  | Tool        | Purpose |
|----|-------------|---------|
| 01 | catalogue   | catalogue.yaml of every part + validator + CLI to add parts |
| 02 | partgen     | parametric mesh generator: render mesh, collision proxy, keypoints, part.json |
| 03 | calibration | Pi capture scripts, ChArUco board, intrinsics, extrinsics, sensor profile |
| 04 | restposes   | stable resting poses of each part (precomputed once) |
| 05 | scenegen    | BlenderProc scene generation and rendering (RGB + instance mask + meta.json) |
| 06 | postprocess | lens distortion (RGB and mask), noise, vignetting, compression |
| 07 | labels      | export to YOLO seg/detect/pose/OBB and COCO, splits, preview overlays |
| 08 | train       | train stage 1 and stage 2, measurement, inference, evaluation |
| 09 | realdata    | capture and label a real validation set, compare real vs synthetic |

## Conventions (binding for all tools)
- Units: every length in configs and metadata is in millimetres. Angles in degrees in configs,
  radians in code.
- Plate frame P: z = 0 is the top surface of the TPU plate, z points up towards the camera, x/y lie
  in the plate. The origin and axes are defined by the extrinsic calibration (tool 03).
- Blender scale: Bullet physics is unstable with millimetre-sized objects, so the Blender scene is
  uniformly scaled by WORLD_SCALE (config, default 10): blender_units = mm * 0.001 * WORLD_SCALE.
  Camera intrinsics are unaffected by a uniform scale; only translations are scaled. All metadata
  written to disk is converted back to real millimetres. Use `screwsynth.common.units`.
- Images: renders cover the ROI only, at full ROI resolution; the principal point is shifted
  accordingly. RGB as 8-bit PNG. Instance masks as 16-bit PNG, 0 = background, k = instance id
  listed in meta.json.
- Part id: `<FAMILY>_<SIZE>x<LENGTH>` for screws (DIN912_M4x16), `<FAMILY>_<SIZE>` for nuts/washers
  (DIN934_M4, DIN125A_M4), `TNUT_<SLIDE|DROP>_S<slot>_<SIZE>` for T-nuts (TNUT_DROP_S6_M4).
  Use `screwsynth.common.naming`.
- Class name: part id without the length (DIN912_M4). Class ids are assigned in a deterministic,
  sorted order and stored in a generated classes.yaml.
- Keypoints: every object has 3 keypoints in this order: head_top, bearing (underside of the head),
  tip. Non-screws mark all 3 as not visible. Nominal length is bearing-to-tip for DIN 912, DIN 933
  and ISO 7380, and head_top-to-tip for DIN 7991 (countersunk length includes the head).
- BlenderProc scripts run inside Blender's own Python. They may only import the standard library,
  numpy, blenderproc/bpy and pure-Python modules from this repo (add src/ to sys.path at the top of
  the script). Anything that needs OpenCV, trimesh, torch etc. runs in normal Python, outside
  Blender. `screwsynth.common` is deliberately pure Python (PyYAML is imported lazily) so it is
  safe to import inside Blender.
- Every tool: a CLI entry point, a README.md, unit tests (pytest), type hints, logging instead of
  print, no hard-coded paths (everything through configs/config.yaml or CLI flags).
- Randomness: every random choice takes a seed; a given seed reproduces the same output.

## Layout
- `configs/config.yaml` — one section per tool; load with `screwsynth.common.config.load_config()`.
- `configs/catalogue.yaml` — parts (tool 01); `configs/classes.yaml` is generated from it.
- `data/` — calibration, plate texture, generated assets, rest poses, real captures.
- `output/` — render runs (gitignored). `datasets/` — exported datasets (gitignored).
- `src/screwsynth/common/` — shared helpers: paths, config, units, io, naming, log, seed.
- Tests marked `@pytest.mark.slow` need Blender, a GPU or large downloads; skip with `-m "not slow"`.
