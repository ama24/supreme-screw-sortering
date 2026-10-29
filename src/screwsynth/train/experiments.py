"""Training experiments: configs/train/<name>.yaml -> Ultralytics train arguments.

name: stage1_nano            # optional, defaults to the file name
stage: seg                   # seg (stage 1: detect + segment + classify) | pose (stage 2 crops)
dataset: datasets/dataset_v1 # tool-07 output (seg.yaml / crops.yaml are used)
model: yolo26n-seg.pt        # pretrained weights, or a .yaml to train from scratch
imgsz: 1024
epochs: 100
batch: -1                    # -1 = auto-fit to the GPU memory (Ultralytics AutoBatch)
augment: {degrees: 180, ...} # see DEFAULT_AUGMENT
train: {}                    # any other Ultralytics train argument (overrides everything)
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from screwsynth.common.io import read_yaml
from screwsynth.common.paths import repo_root, resolve

STAGES = ("seg", "pose")
DATA_YAML = {"seg": "seg.yaml", "pose": "crops.yaml"}
REAL_YAML = {"seg": "seg_real.yaml", "pose": "crops_real.yaml"}

# A fixed overhead camera: any rotation and both flips are real, scale barely changes (the camera
# height is fixed: +-10 %), there is no perspective change or shear. Colour jitter covers finishes
# and lighting. Mosaic helps small objects; it is switched off for the last epochs.
DEFAULT_AUGMENT: dict[str, Any] = {
    "degrees": 180.0,
    "fliplr": 0.5,
    "flipud": 0.5,
    "scale": 0.1,
    "translate": 0.1,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


@dataclass
class Experiment:
    name: str
    stage: str
    dataset: Path
    model: str
    imgsz: int = 1024
    epochs: int = 100
    batch: float = -1
    patience: int = 30
    cache: str | bool = "ram"
    workers: int = 4
    amp: bool = True
    seed: int = 0
    augment: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_AUGMENT))
    train: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    @classmethod
    def load(cls, path: str | Path, dataset: str | Path | None = None) -> Experiment:
        p = Path(path)
        if not p.suffix:
            p = repo_root() / "configs" / "train" / f"{p}.yaml"
        p = resolve(p)
        data = read_yaml(p) or {}
        return cls.from_dict(data, p, dataset)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], source: Path | None = None, dataset: str | Path | None = None
    ) -> Experiment:
        stage = str(data.get("stage", "seg"))
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, not {stage!r}")
        ds = dataset or data.get("dataset")
        if not ds:
            raise ValueError("no dataset: set `dataset:` in the experiment or pass --dataset")
        augment = {**DEFAULT_AUGMENT, **dict(data.get("augment") or {})}
        return cls(
            name=str(data.get("name") or (source.stem if source else stage)),
            stage=stage,
            dataset=resolve(ds),
            model=str(
                data.get("model") or ("yolo26n-seg.pt" if stage == "seg" else "yolo26n-pose.pt")
            ),
            imgsz=int(data.get("imgsz", 1024 if stage == "seg" else 384)),
            epochs=int(data.get("epochs", 100)),
            batch=float(data.get("batch", -1)),
            patience=int(data.get("patience", 30)),
            cache=data.get("cache", "ram"),
            workers=int(data.get("workers", 4)),
            amp=bool(data.get("amp", True)),
            seed=int(data.get("seed", 0)),
            augment=augment,
            train=dict(data.get("train") or {}),
            source=source,
        )

    @property
    def data_yaml(self) -> Path:
        return self.dataset / DATA_YAML[self.stage]

    @property
    def real_yaml(self) -> Path:
        return self.dataset / REAL_YAML[self.stage]

    def ultralytics_args(self, project: Path, data: Path | None = None) -> dict[str, Any]:
        """Keyword arguments for ultralytics YOLO(model).train(**args)."""
        batch = int(self.batch) if self.batch == int(self.batch) else self.batch
        args: dict[str, Any] = {
            "data": str(data or self.data_yaml),
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "batch": batch,
            "patience": self.patience,
            "cache": self.cache,
            "workers": self.workers,
            "amp": self.amp,
            "seed": self.seed,
            "deterministic": True,
            "project": str(project),
            "name": self.name,
            "exist_ok": False,
            "plots": True,
            **copy.deepcopy(self.augment),
        }
        args.update(copy.deepcopy(self.train))
        return args

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "dataset": str(self.dataset),
            "model": self.model,
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "batch": self.batch,
            "patience": self.patience,
            "cache": self.cache,
            "workers": self.workers,
            "amp": self.amp,
            "seed": self.seed,
            "augment": dict(self.augment),
            "train": dict(self.train),
            "source": str(self.source) if self.source else None,
        }
