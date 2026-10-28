# Dataset builder (pipeline + GUI)

This tool builds a whole dataset from one **recipe**: catalogue → parts → rest poses → render → post-process →
labels. You can run it from the browser GUI (`screwsynth-gui`) or the terminal (`screwsynth-dataset`). Both run
the same commands.

- **Code:** `src/screwsynth/pipeline/`
- **Inputs:**
  - `recipes/<name>.yaml`;
  - `configs/config.yaml` and `configs/catalogue.yaml`;
  - `data/` (assets, rest poses, calibration).
- **Outputs:**
  - `output/<name>/`: renders, the merged `config.yaml`, `pipeline.log`, `pipeline_status.json`,
    `environment.txt`, `previews/`;
  - `datasets/<name>/`.

**Status: done.**

## Recipes

```yaml
name: dataset_v1                 # names output/<name> and datasets/<name>
description: First full dataset
pipeline:
  camera: nominal                # nominal | calibration (data/calibration/calibration.json)
  scenes: 3000                   # layouts; each gives scenegen.variants_per_layout images
  chunk: 200                     # scenes per Blender process (restarted in between)
  preview: false                 # true: scenegen.preview resolution/samples, a quick check
  steps: [catalogue, parts, restposes, render, postprocess, labels]
  formats: [seg, pose, coco, crops]
config:                          # only what differs from configs/config.yaml
  catalogue:
    extra_rules:                 # part selection, written by the GUI's Parts tab
      - {match: DIN7991, enabled: false}
      - {match: DIN934_M2, enabled: false}
  scenegen:
    objects: [20, 45]
```

- **Merged config:** the run writes `output/<name>/config.yaml` (config.yaml + `config:`). Every step is
  called with `--config` pointing to that file, so the file fully describes the dataset.
- **classes.yaml:** each run has its own (`output/<name>/classes.yaml`), because the classes depend on the
  part selection.
- **Part selection:** `catalogue.extra_rules` uses the same enable/disable rules as `catalogue.yaml`
  (`screwsynth-catalogue enable|disable`), applied after the file's own rules. Every tool reads the catalogue
  through `catalogue_from_config()`, so they all see the same selection.

## Steps

| Step | Commands (all with `--config output/<name>/config.yaml`) | Reused when re-run |
|---|---|---|
| catalogue | `screwsynth-catalogue validate`, `classes` | – |
| parts | `screwsynth-partgen build --all`, `contact-sheet` | parts already built and up to date |
| restposes | `screwsynth-restposes compute --all`, `contact-sheet` | poses already computed |
| render | `blenderproc run tools/scenegen/generate.py --start K --scenes N --resume` per chunk | finished scenes |
| postprocess | `screwsynth-post run --resume`, `show` (first scene) | processed scenes |
| labels | `screwsynth-labels export`, `check`, `preview` (train and val) | – (`datasets/<name>` is rebuilt) |

`screwsynth-dataset plan NAME` prints the exact commands.

## How a run works

- **Isolation:** each command runs in its own process, so a Blender crash cannot take the GUI down.
- **Failure:** the run stops at the first failed command. `pipeline_status.json` records which step failed.
- **Stop:** the GUI's Stop button kills the current command and its children (Blender).
- **Fresh start:** `--fresh` (the GUI's "start fresh") deletes the dataset's scenes, processed images, previews and
  exported dataset. It never deletes the shared `data/assets` or `data/rest_poses`.
- **Preflight checks:**
  - BlenderProc is installed;
  - the calibration exists when `camera: calibration`;
  - enough memory can be committed (Windows: RAM + page file; warning below 6 GB);
  - a note if the settings changed since the last run, because existing scenes would be kept.

## GUI

```bash
screwsynth-gui [--port 8080] [--no-browser]
```

- **Page:** a local NiceGUI web page. Only one run happens at a time; every open tab shows it.
- **URL parameters:** `/?recipe=preview&tab=run` opens a given recipe and tab.
- **All settings tab:** built from `configs/config.yaml` (`forms.py`):
  - the field type comes from the value (number, switch, range, text, YAML);
  - the help text comes from the `#` comment;
  - a comment like `a | b | c` turns a text field into a choice.

  New keys in config.yaml therefore appear without any GUI change.

## Tests

`pytest tests/pipeline` covers:
- recipe round trip and validation; merge/diff;
- part selection ↔ rules; generated fields and comments;
- the step plan and preflight;
- the runner: success, failure, stop, fresh start, environment snapshot; the CLI;
- two simulated GUI sessions (NiceGUI user simulation):
  - untick a family, save, and the recipe gets the rule;
  - press Run, the plan runs in the background and the status and log update.
