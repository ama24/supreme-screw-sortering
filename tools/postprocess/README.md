# 06 postprocess

Makes ideal renders look like real Pi Camera 3 captures while keeping labels consistent: forward lens distortion (RGB, mask with nearest sampling, keypoints), flat-field vignetting, blur, colour/exposure jitter, signal-dependent sensor noise and compression; plus a real-vs-synthetic comparison report.

- **Inputs:** output/<run>/scenes/* (tool 05), data/calibration/* (tool 03)
- **Outputs:** output/<run>/processed/<id>/{image, instance.png, meta.json}, compare report (HTML)
- **Guide:** [docs/prompts/06_postprocess.txt](../../docs/prompts/06_postprocess.txt)

**Status: TODO**
