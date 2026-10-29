"""Annotations of real images and the CVAT round trip ("CVAT for images 1.1" XML).

One object = a polygon labelled with its class (DIN912_M4, UNKNOWN, ...) plus, for screws lying
flat, up to three point shapes labelled head_top / bearing / tip. The polygon and its points share a
`group_id`, which is how they stay together through the export -> correct in CVAT -> import trip.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from screwsynth.common.naming import KEYPOINTS


@dataclass
class Obj:
    label: str  # class name
    polygon: list[tuple[float, float]]
    keypoints: dict[str, tuple[float, float]] = field(default_factory=dict)  # labelled ones only
    score: float | None = None  # pre-annotation confidence (not exported)
    id: int = 0


@dataclass
class ImageAnn:
    name: str  # file name inside the session's images/ folder
    width: int
    height: int
    objects: list[Obj] = field(default_factory=list)


def to_json(anns: Sequence[ImageAnn]) -> list[dict[str, Any]]:
    return [
        {
            "name": a.name,
            "width": a.width,
            "height": a.height,
            "objects": [
                {
                    "id": o.id,
                    "label": o.label,
                    "score": o.score,
                    "polygon": [list(p) for p in o.polygon],
                    "keypoints": {k: list(v) for k, v in o.keypoints.items()},
                }
                for o in a.objects
            ],
        }
        for a in anns
    ]


def from_json(data: Iterable[Mapping[str, Any]]) -> list[ImageAnn]:
    out = []
    for a in data:
        objs = [
            Obj(
                o["label"],
                [tuple(p) for p in o["polygon"]],
                {k: tuple(v) for k, v in (o.get("keypoints") or {}).items()},
                o.get("score"),
                int(o.get("id", 0)),
            )
            for o in a["objects"]
        ]
        out.append(ImageAnn(a["name"], int(a["width"]), int(a["height"]), objs))
    return out


def _pts(points: Iterable[Sequence[float]]) -> str:
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _parse_pts(text: str) -> list[tuple[float, float]]:
    out = []
    for pair in text.strip().split(";"):
        if pair:
            x, y = pair.split(",")
            out.append((float(x), float(y)))
    return out


def export_cvat(
    anns: Sequence[ImageAnn], classes: Sequence[str], path: Path, task: str = "screwsynth"
) -> None:
    """Write CVAT-for-images 1.1 XML (import it into a CVAT task as pre-annotations)."""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(meta, "name").text = task
    labels = ET.SubElement(meta, "labels")
    for name in [*classes, *KEYPOINTS]:
        lab = ET.SubElement(labels, "label")
        ET.SubElement(lab, "name").text = name
        ET.SubElement(lab, "type").text = "points" if name in KEYPOINTS else "polygon"
    for i, a in enumerate(anns):
        im = ET.SubElement(
            root, "image", id=str(i), name=a.name, width=str(a.width), height=str(a.height)
        )
        for g, o in enumerate(a.objects, start=1):
            ET.SubElement(
                im,
                "polygon",
                label=o.label,
                source="auto" if o.score is not None else "manual",
                occluded="0",
                points=_pts(o.polygon),
                z_order="0",
                group_id=str(g),
            )
            for k in KEYPOINTS:
                if k in o.keypoints:
                    ET.SubElement(
                        im,
                        "points",
                        label=k,
                        source="auto",
                        occluded="0",
                        points=_pts([o.keypoints[k]]),
                        z_order="0",
                        group_id=str(g),
                    )
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def import_cvat(path: Path) -> list[ImageAnn]:
    """Read CVAT-for-images 1.1 XML (exported from CVAT after correcting).

    Points are attached to the polygon with the same group_id; a point without a group is attached
    to the screw polygon that contains it (CVAT drops group ids when shapes are drawn by hand).
    """
    root = ET.parse(path).getroot()
    out = []
    for im in root.iter("image"):
        objs: list[Obj] = []
        by_group: dict[str, Obj] = {}
        loose: list[tuple[str, tuple[float, float]]] = []
        for el in im:
            if el.tag == "polygon":
                o = Obj(el.get("label", ""), _parse_pts(el.get("points", "")), id=len(objs) + 1)
                objs.append(o)
                if el.get("group_id") not in (None, "", "0"):
                    by_group[el.get("group_id")] = o
            elif el.tag == "points" and el.get("label") in KEYPOINTS:
                pts = _parse_pts(el.get("points", ""))
                if not pts:
                    continue
                g = el.get("group_id")
                if g not in (None, "", "0") and g in by_group:
                    by_group[g].keypoints[el.get("label")] = pts[0]
                else:
                    loose.append((el.get("label"), pts[0]))
        for name, p in loose:
            hit = next((o for o in objs if _inside(p, o.polygon)), None)
            if hit is not None and name not in hit.keypoints:
                hit.keypoints[name] = p
        out.append(
            ImageAnn(im.get("name", ""), int(im.get("width", 0)), int(im.get("height", 0)), objs)
        )
    return out


def _inside(p: tuple[float, float], poly: Sequence[tuple[float, float]]) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def cvat_labels(classes: Sequence[str]) -> list[dict[str, Any]]:
    """Label definitions to paste into CVAT's 'Raw' label constructor when creating the project."""
    return [{"name": c, "type": "polygon", "attributes": []} for c in classes] + [
        {"name": k, "type": "points", "attributes": []} for k in KEYPOINTS
    ]
