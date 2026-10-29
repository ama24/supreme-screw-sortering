# 08 train

This tool covers training and the use of the trained models.

- **Training:**
  - stage 1: detect, segment and classify on the 1024 px ROI;
  - stage 2: keypoints on the full-resolution crops.
- **Length measurement**, by two methods:
  - **A:** the keypoint model;
  - **B:** classical sub-pixel edges.

  Both get parallax and tilt correction and snap to the standard lengths.
- **Inference, evaluation, learning curve and fine-tuning on real data.**

Inputs and outputs:
- **Inputs:** `datasets/<name>/` (tool 07; for stage 2 export it with `crops`), `configs/train/*.yaml`, and
  `data/calibration/calibration.json` for real photos.
- **Outputs:**
  - `runs/<stage>/<experiment>/` (weights, plots, `screwsynth_run.json`, `environment.txt`);
  - inference JSON and images;
  - evaluation reports.
- **Guide:** [docs/prompts/08_train_measure.txt](../../docs/prompts/08_train_measure.txt)

**Status: done.** Training itself has not been run on the rendering PC yet (its GPU was busy rendering). The
commands are tested with tiny models on the CPU; see *Expected runtimes*.

## Install

Install the CUDA build of PyTorch first (README step 5), then:

```bash
pip install -e ".[train]"      # ultralytics, onnx, onnxruntime
```

The current Ultralytics release (8.4) uses **YOLO26** models: `yolo26n-seg` / `yolo26s-seg` for stage 1 and
`yolo26n-pose` for stage 2. The pretrained weights download on first use.

## Train

```bash
screwsynth-train run stage1_nano --dataset datasets/dataset_v1      # stage 1 (nano)
screwsynth-train run stage1_small --dataset datasets/dataset_v1     # stage 1 (small, more accurate)
screwsynth-train run stage2_pose --dataset datasets/dataset_v1      # stage 2 keypoints (needs crops)
screwsynth-train run stage1_nano --dataset ... --dry-run            # print the Ultralytics arguments
```

**Experiments.** `configs/train/<name>.yaml` sets:
- the model and image size;
- epochs and patience;
- the batch size (`-1` = auto-fit to the 8 GB GPU, with mixed precision);
- image caching;
- the augmentation.

Any other Ultralytics argument goes under `train:`.

**Augmentation.** The defaults suit a fixed overhead camera:
- any rotation (`degrees: 180`) and both flips;
- small scale jitter only (±10 %; the real scale is fixed);
- no perspective or shear;
- HSV jitter;
- mosaic, switched off for the last 10 epochs.

**Run records.** Every run writes `screwsynth_run.json` with:
- the experiment and the Ultralytics arguments;
- the dataset statistics;
- the training time;
- the metrics.

It also writes `environment.txt` (git commit and package versions).

**Validation.** After training, the run is validated on the **real** set first, when the dataset has one
(`seg_real.yaml`, written by tool 07 with `--val-real`). It is then validated on synthetic val. The real mAP is the
one that matters.

## Length measurement

Both methods give the three keypoints (head_top, bearing, tip) as points on the screw axis in the image, then:

1. **Parallax correction.** Each point is back-projected as a camera ray (with the lens distortion) and intersected
   with the horizontal plane at its height above the plate.
   - The heights come from the class's rest poses (tool 04): the probability-weighted lying poses, interpolated
     between the standard lengths.
   - Since the heights depend on the length, the measurement is iterated: estimate the length, update the heights,
     recompute.
2. **Tilt correction.** The length is the **3D** distance between the two points. A lying screw is tilted, because
   its head lifts the axis.
3. **Nominal length.** Measured bearing → tip for DIN 912, DIN 933 and ISO 7380, and head_top → tip for DIN 7991.
4. **Snapping.** The raw length snaps to the nearest enabled standard length. Exactly between two lengths, the
   shorter one wins.
5. **Output.** The raw mm value, the snapped length and the residual, plus a **confidence**. The confidence is:
   - 1 on a standard length, falling to 0 at half the gap to the next one;
   - reduced when the screw is not measurable (tilted beyond 15°, or cut by the image border);
   - reduced when methods A and B disagree by more than half the gap.
6. **Size check.** The thread diameter is estimated from the shank width at full resolution and compared with the
   stage-1 size class. A mismatch is reported in `size_check`.

**Method A** (`train/infer.py`): YOLO pose on a 384 px full-resolution crop around the detection (the same crop as
tool 07's training crops).

**Method B** (`train/classical.py`):
1. The screw axis is the principal axis of the stage-1 mask; the wider end is the head.
2. The coarse tip, head top and bearing come from the mask's width profile.
3. Each is refined on the image as the strongest intensity edge along lines parallel to the axis (parabolic
   sub-pixel fit).
4. **Rim correction:** the outermost visible point of an end face is its top rim (when the end points away from the
   camera's nadir) or its bottom rim (when it points towards it), not the axis point. From the part geometry, it is
   moved to the axis point behind it. 100 mm off-centre this is worth about 0.5 mm.

`--method auto` (the default) uses A when its keypoint confidence is at least `train.infer.pose_conf`, and B
otherwise.

### Method comparison on synthetic data

```bash
screwsynth-train compare-measure --dataset datasets/<name> [--stage2 runs/pose/.../best.pt] [--split val]
```

This measures every measurable screw in `crops.csv` on the true masks, so detection errors don't mix in. It
compares:
- **A with the true keypoints**, which shows what the geometry alone can reach;
- **A with the keypoint model**, when one is given;
- **B, the classical method.**

The results go to `<dataset>/measurement_comparison/comparison.md`.

Results on the development PC are in [MEASUREMENT.md](MEASUREMENT.md).

## Inference

```bash
screwsynth-infer --image capture.png --stage1 runs/seg/stage1_nano/weights/best.pt \
    --stage2 runs/pose/stage2_pose/weights/best.pt --save-vis
```

1. **Calibration** comes from `paths.calibration_dir` (or `--calibration`; `--nominal-camera` without it).
2. **Input:** a full sensor frame is cropped to the ROI; an image that is already the ROI is used as it is.
3. **Stage 1** runs on the ROI downscaled to 1024 px, and its polygons and boxes are mapped back to full resolution.
4. **Each screw** is measured with A and B.
5. **Output** is `<image>_result.json`. Per object it has:
   - class and size;
   - score, polygon and box;
   - keypoints (A and B);
   - `length` (raw, snapped, residual, confidence, method) and the two raw values;
   - `measurable` and `cut_by_border`;
   - the thread diameter and `size_check`;
   - **`position_mm`** (plate frame) and **`orientation_deg`**, for a pick-and-place head.

   `--save-vis` also writes an annotated image. Unknown parts (class `UNKNOWN`) are listed but not measured.

`train.stage1_weights` / `train.stage2_weights` in the config set the default weights.

### ONNX and CPU latency

```bash
screwsynth-train export --weights runs/seg/stage1_nano/weights/best.pt --imgsz 1024
```

This writes the ONNX model and `<model>.latency.json`. Measured with onnxruntime on the development PC's
i7-9700K (same architecture, untrained weights; the speed does not depend on the weights):

| Model | Input | PC CPU (mean) | Raspberry Pi 5 (estimate: 5–10× slower) |
|---|---|---|---|
| yolo26n-seg (stage 1) | 1024 px | 236 ms | 1.2–2.4 s |
| yolo26s-seg (stage 1) | 1024 px | 692 ms | 3.5–7 s |
| yolo26n-pose (stage 2) | 384 px crop | 15 ms per screw | 0.07–0.15 s per screw |

The nano stage-1 model can run on a Pi 5 at a few seconds per photo, which may be acceptable when the machine takes
one photo per picked part. On the PC's GPU it is about 10 ms.

## Evaluation

```bash
screwsynth-eval --dataset datasets/val_real --stage1 .../best.pt --stage2 .../best.pt \
    --compare-with datasets/dataset_v1
```

It reports:
1. **Detection.** Mask/box mAP per class (Ultralytics val), and the **size confusion** within each family (true size
   → predicted size: is M2 taken for M2.5?).
2. **Length.** The full pipeline runs on every full-resolution scene, and detections are matched to the true screws
   by mask IoU. Per family and size, it gives:
   - the exact-match rate after snapping;
   - the mean and max absolute error before snapping;
   - the bias.
3. **Worst cases.** The 12 worst length errors, as crops in `worst/`.
4. **Sim-to-real gap.** `--compare-with` runs the same on a second dataset (synthetic val) and puts both side by
   side.

The output goes to `<dataset>/eval_<weights>/report.md` and `evaluation.json`.

## Learning curve and fine-tuning

```bash
screwsynth-train curve stage1_nano --dataset datasets/dataset_v1 --fractions 10,25,50,100 --epochs 30
screwsynth-train finetune stage1_nano --weights .../best.pt --data datasets/real_ft/seg.yaml \
    --holdout datasets/val_real/seg.yaml --freeze 10 --lr0 0.0005 --epochs 30
```

- **curve** trains on 10/25/50/100 % of the training layouts. It draws them by layout, so the photos of one layout
  stay together. It plots real-val (and synthetic) mAP against the number of images in `curve.png`. If the curve is
  still rising at 100 %, rendering more data pays off.
- **finetune** continues from the best synthetic model on a small real set:
  - the early layers are frozen and the learning rate is low;
  - it reports mAP on held-out real images before and after (`finetune_comparison.json`);
  - the held-out set is never used for training.

## Expected runtimes (RTX 2070 Super, 8 GB)

Not measured yet: the GPU was busy rendering. These estimates scale Ultralytics' published throughput. Replace
them with the times from `screwsynth_run.json` (`train_minutes`) after the first runs.

| Run | Images | Estimate |
|---|---|---|
| stage1_nano, 100 epochs | 3000 | ~1.5–2.5 h |
| stage1_small, 120 epochs | 3000 | ~4–6 h |
| stage2_pose, 150 epochs | 3000 crops | ~0.5–1 h |
| curve (4 × 30 epochs, nano) | 300–3000 | ~1.5 h |
| inference per photo (GPU) | 1 | ~0.1 s plus ~10 ms per screw |

## Tests

`pytest tests/train` covers:
- parallax and tilt correction within 0.05 mm, with and without lens distortion;
- iterating with length-dependent heights;
- countersunk anchors;
- snapping edge cases (exactly between two lengths, outside the list, a single length);
- confidence rules;
- heights from rest poses;
- sub-pixel edge keypoints on anti-aliased images at any angle, with the head at either end;
- the rim correction;
- experiments → Ultralytics arguments;
- learning-curve subsets kept by layout;
- the method comparison and the per-screw measurement on a fake dataset;
- the full inference pipeline with an untrained model (slow).
