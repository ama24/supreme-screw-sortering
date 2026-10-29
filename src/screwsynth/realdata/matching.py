"""Link every labelled screw to the inventory, so its true length comes from the calipers.

Per image and class:
- one inventory length for the class -> every screw of that class gets it (auto);
- several lengths -> each screw with labelled keypoints is measured (tool 08 geometry) and gets the
  nearest inventory length when that is clearly closer than the next one (within `margin` of the
  gap between them) and the inventory count allows it (auto); the rest is left `ambiguous` with
  its candidate lengths, for the operator to fill in (status `manual`).
Rows set to `manual` in matches.csv survive re-running the matching.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .session import InventoryRow

FIELDS = ("image", "object_id", "class_name", "measured_mm", "length_mm", "status", "candidates")


@dataclass
class Match:
    image: str
    object_id: int
    class_name: str
    measured_mm: float | None
    length_mm: float | None
    status: str  # auto | ambiguous | manual | not_in_inventory | over_count
    candidates: list[float] = field(default_factory=list)

    def row(self) -> dict[str, str]:
        return {
            "image": self.image,
            "object_id": str(self.object_id),
            "class_name": self.class_name,
            "measured_mm": "" if self.measured_mm is None else f"{self.measured_mm:.2f}",
            "length_mm": "" if self.length_mm is None else f"{self.length_mm:g}",
            "status": self.status,
            "candidates": " ".join(f"{c:g}" for c in self.candidates),
        }


def match_class(
    image: str,
    cls: str,
    screws: Sequence[tuple[int, float | None]],
    inventory: Sequence[InventoryRow],
    margin: float = 0.4,
) -> list[Match]:
    """screws: (object id, measured length or None) of one class in one image."""
    rows = [r for r in inventory if r.class_name == cls and r.length_mm is not None]
    if not rows:
        return [Match(image, oid, cls, m, None, "not_in_inventory") for oid, m in screws]
    capacity = Counter()
    for r in rows:
        capacity[r.length_mm] += r.count
    lengths = sorted(capacity)
    if len(lengths) == 1:
        L = lengths[0]
        status = "auto" if len(screws) <= capacity[L] else "over_count"
        return [Match(image, oid, cls, m, L, status, lengths) for oid, m in screws]
    out: list[Match] = []
    proposals = []
    for oid, m in screws:
        if m is None:
            out.append(Match(image, oid, cls, None, None, "ambiguous", lengths))
            continue
        near = sorted(lengths, key=lambda L: abs(L - m))
        best, second = near[0], near[1]
        gap = abs(second - best)
        if abs(m - best) <= margin * gap:
            proposals.append((abs(m - best), oid, m, best))
        else:
            out.append(Match(image, oid, cls, m, None, "ambiguous", lengths))
    used: Counter = Counter()
    for _, oid, m, best in sorted(proposals):  # most certain first
        if used[best] < capacity[best]:
            used[best] += 1
            out.append(Match(image, oid, cls, m, best, "auto", lengths))
        else:
            out.append(Match(image, oid, cls, m, None, "ambiguous", lengths))
    return sorted(out, key=lambda x: x.object_id)


def match_image(
    image: str,
    screws: Iterable[tuple[int, str, float | None]],
    inventory: Sequence[InventoryRow],
    margin: float = 0.4,
) -> list[Match]:
    """screws: (object id, class, measured length or None) of every screw in the image."""
    by_cls: dict[str, list] = defaultdict(list)
    for oid, cls, m in screws:
        by_cls[cls].append((oid, m))
    out = []
    for cls, items in sorted(by_cls.items()):
        out += match_class(image, cls, items, inventory, margin)
    return out


def read_matches(path: Path) -> dict[tuple[str, int], Match]:
    if not path.is_file():
        return {}
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        m = Match(
            r["image"],
            int(r["object_id"]),
            r["class_name"],
            float(r["measured_mm"]) if r.get("measured_mm") else None,
            float(r["length_mm"]) if r.get("length_mm") else None,
            r.get("status", ""),
            [float(c) for c in (r.get("candidates") or "").split()],
        )
        out[(m.image, m.object_id)] = m
    return out


def write_matches(
    path: Path, matches: Iterable[Match], keep: Mapping[tuple[str, int], Match] | None = None
) -> list[Match]:
    """Write matches.csv; rows the operator set to `manual` (with a length) are kept."""
    final = []
    for m in matches:
        old = (keep or {}).get((m.image, m.object_id))
        if old is not None and old.status == "manual" and old.length_mm is not None:
            m = Match(
                m.image,
                m.object_id,
                m.class_name,
                m.measured_mm,
                old.length_mm,
                "manual",
                m.candidates,
            )
        final.append(m)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for m in final:
            w.writerow(m.row())
    return final


def summary(matches: Sequence[Match]) -> dict[str, int]:
    return dict(Counter(m.status for m in matches))
