# 06 postprocess

Tool 05 renders ideal, undistorted images. This tool makes them look like real Pi Camera 3 captures and keeps
the labels consistent. It runs in normal Python (OpenCV, numpy) and never modifies the raw renders.

- **Inputs:**
  - `output/<run>/scenes/*/{rgb.png, instance.png, meta.json}` (tool 05)
  - `data/calibration/{calibration.json, sensor_profile.json, flatfield.png}` (tool 03), all optional
- **Outputs:** `output/<run>/processed/<scene_id>/{image.png|image.jpg, instance.png, meta.json}`
- **Guide:** [docs/prompts/06_postprocess.txt](../../docs/prompts/06_postprocess.txt)

**Status: done.** Not yet tuned against real captures; use `compare` once tool 09 has real images.

## Usage

```bash
screwsynth-post run --run run01 [--workers 4] [--resume] [--seed S]
screwsynth-post show --run run01 --scene 000000      # before/after, mask outline + keypoints -> output/run01/show_000000.png
screwsynth-post compare --run run01 --real data/real/session01 [--crop-roi]   # -> output/run01/compare.html
```

Processing is deterministic: the seed comes from the scene id and the run seed. `--resume` skips scenes that are
already processed. Scenes are processed in parallel; if worker processes cannot start, it falls back to a single
process.

## Effects (in this order) and their config keys (`postprocess` in `configs/config.yaml`)

Each step has `enabled` and `probability` keys, plus its own ranges. A `[lo, hi]` range is sampled uniformly per
image. The exact parameters applied are stored in `meta.json → postprocess`.

1. **Lens distortion** (`distortion`).
   - The render is an ideal pinhole image with the render's K (ROI crop, scaled). The **forward** distortion is
     applied as follows: for every output (distorted) pixel, `cv2.undistortPoints` finds where it comes from in the
     render, and the render is sampled there. Distortion acts on normalised coordinates, so the calibrated
     coefficients apply unchanged to the ROI crop.
   - RGB uses `rgb_interpolation` (`linear` / `cubic` / `lanczos`).
   - The instance mask uses **nearest** only, so ids never blend. Areas outside the render become background.
   - Keypoints are distorted with the same model (`cv2.projectPoints`). The old position is kept as
     `uv_undistorted`. Keypoints that leave the image get visibility 0.
   - `mask_area_px` and `visible_pixel_fraction` are recomputed.
   - With the nominal camera (no distortion), this step is the identity.
2. **Vignetting / flat-field** (`vignetting.strength`).
   - The calibrated full-sensor `flatfield.png` is cropped to the ROI and resampled. The image is multiplied by
     `1 − s·(1 − flatfield)`.
   - Without calibration, a radial falloff `1 − fallback_k·r²` is used.
3. **Optics blur** (`blur.sigma_px`): a Gaussian, with sigma given at full ROI resolution and scaled for previews.
   It matches the lens MTF. Tune it with the edge width in `compare`.
4. **Chromatic aberration** (`chromatic_aberration`, off by default): a tiny R/B magnification about the
   principal point.
5. **Exposure / colour** (`colour`): gain, per-channel white balance and gamma.
   - With `match_plate: true` and a sensor profile, the gains are first **centred on the real empty-plate colour**:
     the synthetic plate median is mapped to `sensor_profile.plate_empty.mean_rgb`, and the jitter is applied
     around that.
6. **Sensor noise** (`noise`): signal-dependent, variance = a·I + b per channel, from `sensor_profile.json` (or
   `fallback`), times `scale`. `fixed_pattern` adds a per-pixel gain map (PRNU) that is fixed per run.
   - **Space:** tool 03 measures the noise on captured 8-bit images (after the Pi's tone curve), so the noise is
     applied in those same units. To first order this equals adding the corresponding noise in linear space
     before the tone curve, without needing the unknown ISP curve.
7. **Compression** (`compression`).
   - The image is saved as `production_format` (`png`, matching what the Pi pipeline saves). With probability
     `jpeg_probability` it is saved as JPEG instead, at a quality in `quality`, for robustness.
   - If production switches to JPEG, set `production_format: jpg`.

## Tuning with the realism report

`screwsynth-post compare --run R --real data/real/<set>` writes `compare.html` and `compare.json` for N synthetic
and N real images. It contains:
- side-by-side tiles and 2× centre crops;
- per-channel intensity histograms;
- noise std vs intensity (measured on flat plate regions after a 5×5 median);
- edge width (the 10–90% rise across object borders, from the mask for synthetic images and from segmentation for
  real ones);
- plate colour and mean brightness;
- suggestions, such as "synthetic edges too sharp: raise blur.sigma_px" or "plate colour differs: check
  colour.match_plate / lights".

Use `--crop-roi` when the real images are full-frame. Iterate in this order:
1. Lights / exposure until brightness and plate colour match.
2. `blur.sigma_px` until the edge widths match.
3. Noise `scale` until the noise curves overlap.
4. Material presets in `scenegen.materials`, for example if zinc looks too bright.

## Output `meta.json` changes

Compared with the raw render:
- `image` is set to the processed image.
- `camera.distorted` is true.
- `postprocess` holds every parameter used.
- `source_scene` points to the raw scene.
- Per object, `keypoints.*.uv` is the **distorted** position, and the old value is kept in `uv_undistorted`.
  Visibility, `mask_area_px` and `visible_pixel_fraction` are updated.

## Tests

`pytest tests/postprocess`:
- A straight plate line, rendered undistorted and then distorted, lies on the calibrated (OpenCV) curve: median
  < 0.2 px. `distort_points` equals `projectPoints`.
- Mask and RGB edges coincide within 1 px after distortion, and ids never blend.
- Keypoints stay on their features (blob centroid < 0.3 px).
- Noise statistics, flat-field crop and vignetting.
- Full scene processing: meta updates, determinism, raw scene untouched.
- The realism report detects too-sharp edges.
- CLI run / resume / show / compare.
