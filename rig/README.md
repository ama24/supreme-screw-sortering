# screwrig: the Raspberry Pi side

`screwrig` runs on the Raspberry Pi that sits on the rig. It drives:
- the **Pi Camera 3**, with locked exposure, gain, white balance and focus;
- the **printer bed**: homing and parking Y over USB serial.

It records the captures for calibration (tool 03) and for real-data sessions (tool 09). Later it will also run the
sorter loop: photo → detect → measure → pick.

It uses the `screwsynth` package from the repository root for the shared parts: config, catalogue, session files,
calibration recheck and measurement.

## Install (Raspberry Pi OS Bookworm)

```bash
sudo apt update && sudo apt install -y python3-picamera2 python3-numpy git
git clone https://github.com/ama24/supreme-screw-sortering.git && cd supreme-screw-sortering
python3 -m venv --system-site-packages ~/screwrig     # Picamera2 comes from apt, not pip
source ~/screwrig/bin/activate
pip install -r rig/requirements-pi.txt                # screwsynth + screwrig + OpenCV (headless)
screwrig --help
```

On the PC (for tests or development): `pip install -e . -e rig/`. OpenCV comes from `screwsynth[calib]` there.

## Commands

| Command | What it does |
|---|---|
| `screwrig lock --out camera_settings.json` | Lets exposure and white balance settle, freezes them, runs a focus sweep, and saves the locked settings. Run it once per rig change. |
| `screwrig calib-capture --mode intrinsics\|extrinsic\|sensor\|single --out <dir>` | Calibration captures with the locked settings. It refuses to run if the camera does not apply them exactly. |
| `screwrig capture --session s01 --settings camera_settings.json [--recheck]` | Real-data session. Each Enter homes/parks the bed and saves a photo plus sidecar into `data/real/s01/images`. Create the session first with `screwsynth-real session new`. |
| `screwrig printer park\|status` | Homes Y (and parks it, if configured), then reports the position |

## Printer safety

Only these G-code commands can ever be sent:
- `G28 Y` (home Y);
- `G0 Y<park_y_mm>` (park Y at the configured position);
- `M400` (wait for moves to finish);
- `M114` (report the position).

Anything else is refused before it reaches the serial port. The settings live under `rig.printer` in
`configs/config.yaml`: `port`, `baud`, `park_y_mm`, `feed_mm_min` and `timeout_s`.

## Tests

`pytest rig/tests` runs off the Pi: the camera helpers are tested on synthetic images, and the printer on a fake
serial port.
