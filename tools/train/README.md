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

## Measurement, inference and evaluation

These moved to [tools/measure/README.md](../measure/README.md) (package `screwsynth.measure`).
They are used on the PC and on the Pi.

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
