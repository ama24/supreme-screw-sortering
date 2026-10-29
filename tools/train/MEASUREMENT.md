# Measurement methods: A (keypoints) vs B (classical) on synthetic data

`screwsynth-train compare-measure` was run on the **true masks** of rendered scenes, so detection errors do not mix in.

**Setup:**
- 8 scenes, nominal camera, rendered at full resolution (2560 × 2560 ROI, about 11.7 px/mm), 32 samples;
- post-processed by tool 06 (blur, noise, compression);
- 86 measurable screws (lying on the plate, free, inside the image), all four screw families, M2–M10.

| Method | Exact after snapping | Mean abs error | Max abs error | Bias |
|---|---|---|---|---|
| A with the **true** keypoints (geometry only) | 100 % | 0.033 mm | 0.98 mm | −0.005 mm |
| B classical, true mask | 100 % | 0.41 mm | 2.02 mm | −0.27 mm |

B by size:

| Size | n | Mean abs | Max abs | Bias |
|---|---|---|---|---|
| M2 | 7 | 0.17 mm | 0.62 mm | −0.13 mm |
| M2.5 | 3 | 0.12 mm | 0.25 mm | −0.12 mm |
| M3 | 11 | 0.22 mm | 0.54 mm | −0.20 mm |
| M4 | 12 | 0.39 mm | 0.81 mm | −0.21 mm |
| M5 | 11 | 0.40 mm | 0.79 mm | −0.22 mm |
| M6 | 15 | 0.33 mm | 0.78 mm | −0.23 mm |
| M8 | 14 | 0.52 mm | 1.79 mm | −0.51 mm |
| M10 | 13 | 0.77 mm | 2.02 mm | −0.34 mm |

## Reading the numbers

- **The geometry is exact.** With perfect keypoints, the measurement is within 0.03 mm on average. That includes the
  parallax correction from the rest-pose heights, the tilt correction, and the iteration over length-dependent
  heights. The keypoint model therefore only has to locate three points well: 1 px is about 0.085 mm.
- **B is a robust fallback, not the main method for small screws.** Its error grows with the head size. The step
  under large chamfered heads (hex DIN 933, button ISO 7380) is blurred, and B underestimates slightly (−0.27 mm on
  average). M2 lengths can be only 1 mm apart, so B's M2 margin is small (max 0.62 mm against a half-gap of 0.5 mm in
  the worst case).
- **The bias could be calibrated away.** B's bias is systematic per family. Once real data exists (tool 09), a
  per-family offset could remove it.
- **Method A with a trained model has not been measured yet,** because the GPU was busy rendering. Train
  `stage2_pose` on a dataset exported with `crops`, then run:

  ```bash
  screwsynth-train compare-measure --dataset datasets/<name> --stage2 runs/pose/stage2_pose/weights/best.pt
  ```

  If A reaches roughly 1 px, it will be several times more accurate than B. `--method auto` uses A whenever its
  keypoint confidence is high and falls back to B.

## Fixes the comparison found

Two bugs, found on full-resolution renders, were fixed before these numbers:
1. **The bearing search** started a fixed share of the length from the head end, which on long screws with short
   heads was already past the head (errors of 3–10 mm). It now waits until the full head width is reached.
2. **The rim correction** moved the extremity by the full radial parallax shift. Only its component along the screw
   axis moves the visible tip, so screws lying across the radial direction were over-corrected. It is now projected
   onto the axis.
