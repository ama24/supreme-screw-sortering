# 03 calibration

Calibrates the real Raspberry Pi Camera Module 3 so the Blender camera matches it: ChArUco board generation, locked-settings capture scripts on the Pi (Picamera2), and PC-side intrinsic, extrinsic (plate frame, ROI, px/mm) and sensor-profile solvers with validation and drift recheck.

- **Inputs:** Captures from the Pi (tools/calibration/pi/), measured board square size
- **Outputs:** data/calibration/{camera_settings.json, calibration.json, sensor_profile.json, flatfield.png, plate_empty.png, report.html}
- **Guide:** [docs/prompts/03_calibration.txt](../../docs/prompts/03_calibration.txt)

**Status: TODO**
