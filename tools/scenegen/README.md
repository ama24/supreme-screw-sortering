# 05 scenegen

The core renderer. It builds random single-layer scenes of fasteners lying on the TPU plate, seen by a virtual
camera identical to the calibrated one, and writes three files per image: the RGB render, a 16-bit instance mask,
and a `meta.json` with the full ground truth. It runs **inside Blender** through BlenderProc. Lens distortion and
sensor effects are added later by tool 06.

- **Inputs:**
  - `configs/config.yaml` (`scenegen` section), `configs/catalogue.yaml`, `configs/classes.yaml`
  - `data/assets/*` (tool 02) and `data/rest_poses/*` (tool 04)
  - `data/calibration/calibration.json` and `plate_empty.png` (tool 03), or `--nominal-camera` instead
- **Outputs:** `output/<run>/`
- **Guide:** [docs/prompts/05_scenegen.txt](../../docs/prompts/05_scenegen.txt)

**Status: implemented. The Blender part has not been executed yet.**
- The pure-Python modules are unit tested: camera, composition sampler, placement, materials, ground truth and
  the PNG writer.
- The BlenderProc code was written against the installed BlenderProc 2.8 sources.
- It could not be run on the development PC: Windows commit memory was exhausted by other applications (Blender
  needs about 1–2 GB).
- **First thing to do:** run `--preview` (below) and the slow smoke test, and report any error.

## What do I actually run?

```bash
pip install blenderproc                      # once; the first `blenderproc run` downloads Blender 4.2
blenderproc pip install pyyaml               # optional: generate.py installs it on first use

# prerequisites (normal Python)
screwsynth-catalogue classes
screwsynth-partgen build --all
screwsynth-restposes compute --all

# 10 quick scenes (1/4 resolution, 16 samples), no calibration needed
blenderproc run tools/scenegen/generate.py --run preview --preview --nominal-camera

# a real run with the calibrated camera, restarting Blender every 200 scenes
python tools/scenegen/batch.py --run run01 --total 3000 --chunk 200 --seed 1

# inspect one scene in the Blender UI
blenderproc debug tools/scenegen/generate.py --run debug --scenes 1 --nominal-camera

# speed / quality trade-off on your GPU (writes output/bench/benchmark.json)
python tools/scenegen/batch.py --run bench --benchmark --nominal-camera
```

`generate.py` options:
- `--run NAME`, `--scenes N`
- `--start K`: index of the first scene
- `--seed S`: defaults to the config's `seed`
- `--resume`: skip finished scenes
- `--nominal-camera`, `--preview`
- `--config PATH`

Runs are **resumable and deterministic**:
- Scene k always uses the seed `derive_seed(seed, "scene", k)`.
- The class-balance state is saved after every scene, in `balance_state.json`.
- `batch.py` runs chunks of `--chunk` scenes in fresh Blender processes, each with `--resume`. Rerunning an
  interrupted batch continues where it stopped.
- A crash in one scene is logged to `log.txt` and `failed.jsonl` and skipped; it is not fatal.

## How a scene is made

**Built once per Blender session:**
- **Camera:** K and `T_cam_plate` from `calibration.json`. The render covers only the ROI: the principal point
  is shifted by the ROI offset, and the resolution is the ROI size × `resolution_scale`.
  - `bproc.camera.set_intrinsics_from_K_matrix` sets the focal length, shift and pixel aspect.
  - The pose is converted from OpenCV (x right, y down, looking along +z) to Blender (looking along −z).
  - On start, **plate points are projected with Blender's camera and with K/T in numpy. If they differ by more
    than 0.1 px, the run stops.**
- **World scale:** all geometry and translations are scaled by `world_scale` (Blender units = mm · 0.001 · 10).
  Metadata is written back in real mm.
- **Plate:** a grid mesh at z = 0. Its UVs are the projection of each vertex into the ROI, so
  `plate_empty.png` maps onto the plate perspective-correctly.
  - Roughness is randomised, with fine print-line bands as a bump.
  - With `solid_colour_probability`, the plate gets a solid colour instead (domain randomisation).
  - Under it: a passive box collider, and the bed.
- **Surroundings:** a dark matte shroud box encloses everything, so metal reflects the enclosure and the light
  rather than an open room. One rectangular area light panel acts as the diffuse LED panel.

**Per scene:**
1. **Composition.** The scene type is *normal*, *few* (1–5 objects) or *crowded* (parts touch).
   - Classes are drawn by **deficit sampling**: uniformly among classes whose instance count so far is within
     `balance_slack` of the minimum. Counts persist across chunks.
   - Then a length is chosen uniformly among the class's enabled parts, and a finish from the part's allowed
     list (optionally weighted).
2. **Placement.**
   - Each part gets a rest pose drawn by probability (tool 04) and a random yaw.
   - Positions are random inside the working area ∩ the visible ROI, minus `area_margin_mm`.
   - Overlaps are rejected with a separating-axis test on 2D convex footprints (the collision proxy in that rest
     pose).
   - Crowded scenes allow `crowded_gap_mm` of overlap and slide parts against neighbours (`touch_probability`).
3. **Settle.** A short Bullet simulation using the compound convex collision pieces (not the render mesh), with
   mass from volume × density.
   - Parts that fell off, ended up stacked (lowest point > `max_lift_mm`), or moved more than `max_move_mm` are
     discarded (hidden) and listed under `dropped`.
4. **Materials per instance.** PBR presets with jittered colour, roughness, anisotropy (machined look), coat and
   subsurface (nylon), plus optional scratches, grime and fingerprint noise. Nyloc inserts get their own nylon
   material slot. The presets can be changed in the config (`scenegen.materials`).
5. **Light and plate jitter.** Panel size, height, offset, power and colour temperature, a weak world fill, and
   plate roughness and bump. `variants_per_layout` re-renders the same arrangement under new lighting.
6. **Render.** Cycles + OptiX with adaptive sampling, the OptiX denoiser and persistent data.
   - The view transform is *Standard* (no filmic curve).
   - Outputs: RGB, plus instance segmentation via the `category_id` custom property (0 = background, k = object
     k). Depth is optional.
7. **Ground truth.** See the schema below.

## Output layout

```
output/<run>/
  run_config.yaml        exact config + git commit + seed + camera
  balance_state.json     instances per class so far
  log.txt, failed.jsonl
  scenes/000000/
    rgb.png              8-bit RGB, ROI size (undistorted)
    instance.png         16-bit, 0 = background, k = instance id
    depth.npy            only with render.depth: true
    meta.json
```

With `variants_per_layout > 1`, the scene folders are `000000_0`, `000000_1`, …, and they share a `layout_id`.
Tool 07 keeps layouts together when splitting.

### `meta.json` schema (lengths in mm in the plate frame; pixels in the undistorted render)

| Key | Meaning |
|---|---|
| `scene_id`, `layout_id`, `variant`, `seed`, `scene_kind` | Identification (`scene_kind`: normal / few / crowded) |
| `camera` | `K` (render), `K_full`, `dist_coeffs`, `image_size_full`, `roi`, `resolution`, `resolution_scale`, `T_cam_plate`, `distorted: false` |
| `world_scale`, `area_mm` | Blender scale and the placement rectangle |
| `lighting` | Sampled light/plate parameters |
| `objects[]` | One record per kept object (below) |
| `dropped[]` | `{part_id, reason}` for parts discarded after the settle |
| `fully_hidden_ids` | Kept objects with no visible pixel |

Per object:

| Key | Meaning |
|---|---|
| `instance_id` | Value in `instance.png` |
| `part_id`, `class_name`, `class_id`, `kind`, `family`, `size`, `length` | Identity (`class_id` from `classes.yaml`) |
| `finish`, `material` | Finish and the exact shader parameters |
| `pose` | 4×4 part local frame → plate frame (mm) after the settle |
| `rest_pose`, `axis_tilt_deg` | Rest pose index, and the actual axis elevation after the settle |
| `keypoints.{head_top,bearing,tip}` | `xyz` (plate mm), `uv` (render px, undistorted), `visibility` (COCO: 2 visible, 1 occluded, 0 not labelled/outside) |
| `keypoints_labelled` | False for non-screws; all their visibilities are then 0 |
| `axis_height_head_mm`, `axis_height_tip_mm`, `axis_height_head_top_mm` | Heights of the keypoints (on the screw axis) above the plate, for parallax correction |
| `measurable` | Screw with \|axis tilt\| ≤ `restposes.measurable_tilt_deg` |
| `mask_area_px`, `alone_area_px`, `alone_area_in_image_px` | Visible pixels, the silhouette when alone, and the part of it inside the image |
| `visible_pixel_fraction` | `mask_area_px / alone_area_px` |
| `truncated` | Cut by the image border |

- **Visibility.** A keypoint is visible when its pixel, or a pixel within `keypoint_window_px` of it, belongs to
  the object in the instance mask. It counts as occluded when another instance covers it.
- **Silhouette when alone.** Computed cheaply, without an extra render, by rasterising the union of the projected
  convex collision pieces. Their vertices lie on the true radii, so the area matches the render within a few %.

## Config keys (`scenegen` in `configs/config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `objects`, `few_objects`, `crowded_objects` | [15,40], [1,5], [30,50] | Objects per scene type |
| `few_fraction`, `crowded_fraction` | 0.1, 0.2 | Fractions of scene types |
| `min_gap_mm`, `crowded_gap_mm`, `touch_probability` | 1.0, −0.3, 0.5 | Spacing and touching |
| `area_margin_mm`, `max_place_tries` | 3, 60 | Placement |
| `balance_slack`, `finish_weights` | 1, {} | Class balancing and finish weights |
| `variants_per_layout`, `keypoint_window_px` | 1, 2 | Lighting variants and keypoint visibility window |
| `nominal_camera.*` | 4.74 mm, 1.4 µm, 4608×2592, 300 mm | `--nominal-camera` |
| `physics.*` | drop 0.05 mm, 0.4–1.5 s, 20 substeps/iters, friction 0.6, margin 0.05 mm | Settle |
| `discard.*` | lift 0.3 mm, move 15 mm, below 1 mm | Post-settle rejection |
| `render.*` | OPTIX, 64 samples, threshold 0.02, OPTIX denoiser, scale 1.0, bounces 6/3/4/2, depth off | Cycles |
| `preview.*` | 10 scenes, scale 0.25, 16 samples | `--preview` |
| `plate.*` | thickness 1.75, grid 24, roughness, bump, line pitch, solid colours | Plate |
| `shroud.*`, `lights.*` | | Enclosure and light panel ranges (power in Blender units at `world_scale` 10) |
| `wear`, `materials` | | Wear probabilities and preset overrides (defaults in `scenegen/materials.py`) |

## Performance (RTX 2070 Super, 8 GB)

`log.txt` reports seconds per scene. The target is 20–40 s per full-resolution ROI scene (about 2560 × 2560).
Run `python tools/scenegen/batch.py --run bench --benchmark` to time these settings on your machine:
- 16, 32, 64 and 128 samples with OptiX;
- 64 samples without a denoiser;
- 64 samples at half resolution.

Record the results here; they could not be measured on the development PC:

| samples | denoiser | scale | s/scene | notes |
|---|---|---|---|---|
| – | – | – | – | run the benchmark |

Rough guide:
- Samples and resolution drive the time almost linearly.
- With OptiX, 32–64 samples are usually enough for diffuse light.
- Glossy bounces matter for polished stainless and zinc.
- Keep `use_persistent_data` on: the BVH and textures are reused between scenes.

## Tests

`pytest tests/scenegen`. Without Blender, these check:
- **Camera:** ROI and scale intrinsics, agreement with the calibration geometry, and the Blender pose convention.
- **Sampler:** balances classes over 1000 simulated draws (spread ≤ 2), recovers from imbalance, persists state,
  and plans deterministically.
- **Placement and discard:** overlap rejection, area boundaries, touching mode, and the discard rules.
- **Ground truth and helpers:** keypoint visibility, silhouette area, material sampling, and the PNG writer/reader.

`test_blender_smoke` (marked slow) runs `--preview --nominal-camera` in BlenderProc. It checks that the instance
ids in `meta.json` match `instance.png`. Run it with `pytest -m slow` when `blenderproc` is on PATH.
