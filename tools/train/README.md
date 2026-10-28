# 08 train

Trains stage 1 (YOLO segmentation at 1024 px) and stage 2 (keypoint model and a classical sub-pixel edge method), implements parallax- and tilt-corrected length measurement with snapping to standard lengths, and provides inference, ONNX export, evaluation, learning curves and real-data fine-tuning.

- **Inputs:** datasets/<name>/ (tool 07), configs/train/*.yaml, data/calibration/calibration.json
- **Outputs:** Trained weights, inference JSON and visualisations, evaluation reports
- **Guide:** [docs/prompts/08_train_measure.txt](../../docs/prompts/08_train_measure.txt)

**Status: TODO**
