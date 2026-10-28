# 05 scenegen

The core BlenderProc renderer: builds random single-layer scenes of fasteners on the TPU plate seen by a virtual camera identical to the calibrated one, with class-balanced sampling, PBR finishes and lighting randomisation, and writes RGB, a 16-bit instance mask and full ground truth per scene.

- **Inputs:** configs/config.yaml, catalogue.yaml, classes.yaml, data/assets, data/rest_poses, data/calibration (or --nominal-camera)
- **Outputs:** output/<run>/scenes/<id>/{rgb.png, instance.png, meta.json}, run_config.yaml, balance_state.json
- **Guide:** [docs/prompts/05_scenegen.txt](../../docs/prompts/05_scenegen.txt)

**Status: TODO**
