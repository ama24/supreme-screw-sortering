# models

Trained model packages from [model-trainer](https://github.com/ama24/model-trainer) go here (not tracked by git).
Copy the run's whole **package** folder, so `model_card.json` comes along:

```
models/
  stage1/                      <- the package folder of the segmentation run
    best.pt
    model_card.json
    exports/ncnn_fp32/best_ncnn_model/   <- after Export > NCNN > fp32
  stage2/                      <- optional: the keypoint run's package
```

Then: `screwsynth-infer --stage1 models/stage1/exports/ncnn_fp32/best_ncnn_model ...` (see `rig/README.md`).
Paths are relative to the repository root, from any folder.
