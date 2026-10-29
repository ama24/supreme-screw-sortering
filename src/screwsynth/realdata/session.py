"""Capture sessions and their inventory sheets.

data/real/<session>/
  session.json        id, time, operator, notes, camera settings / calibration fingerprints, recheck
  inventory.csv       what is on the plate (filled in by hand BEFORE capture): the length truth
  images/             img_0001.png + img_0001.json (capture sidecar), ...
  annotations/        preannotations.json, cvat.xml (export), corrected.xml (import), labels.json
  matches.csv         labelled screws <-> inventory lengths
  scenes/             one processed-format scene per labelled image (input of tool 07)
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from screwsynth.common.io import read_json, write_json
from screwsynth.common.naming import UNKNOWN, class_name, parse_part_id

INVENTORY_FIELDS = ("part_id", "finish", "length_mm", "count", "notes")
SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SessionError(ValueError):
    pass


@dataclass
class InventoryRow:
    part_id: str
    class_name: str
    finish: str
    length_mm: float | None
    count: int
    notes: str = ""


def fingerprint(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else None


def new_session(
    root: Path,
    name: str | None = None,
    operator: str = "",
    notes: str = "",
    camera_settings: Path | None = None,
    calibration: Path | None = None,
) -> Path:
    """Create data/real/<name>/ with session.json and an inventory template."""
    name = name or datetime.now().strftime("session_%Y%m%d_%H%M")
    if not SESSION_RE.match(name):
        raise SessionError(f"session name {name!r}: letters, digits, '_', '-', '.' only")
    d = root / name
    if (d / "session.json").exists():
        raise SessionError(f"session {name} already exists")
    for sub in ("images", "annotations", "scenes"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    write_json(
        d / "session.json",
        {
            "id": name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "operator": operator,
            "notes": notes,
            "camera_settings": str(camera_settings) if camera_settings else None,
            "camera_settings_sha": fingerprint(camera_settings) if camera_settings else None,
            "calibration_sha": fingerprint(calibration) if calibration else None,
            "recheck": None,
        },
    )
    write_inventory_template(d / "inventory.csv")
    return d


def load_session(d: Path) -> dict[str, Any]:
    if not (d / "session.json").is_file():
        raise SessionError(
            f"{d} is not a session (no session.json): create it with `screwsynth-real session new`"
        )
    return read_json(d / "session.json")


def write_inventory_template(path: Path, examples: Iterable[str] = ()) -> None:
    """Inventory sheet: one row per part type on the plate. Fill it in before capturing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(
            "# One row per kind of part on the plate. part_id as in the catalogue (DIN912_M4x16,\n"
            "# DIN934_M4, TNUT_DROP_S6_M4); for screws, length_mm is what the CALIPER says (it is the\n"
            "# ground truth for length). finish: black_oxide, zinc_clear, ... count: how many.\n"
            "# Junk that is not in the catalogue: part_id UNKNOWN. Lines starting with # are ignored.\n"
        )
        w = csv.writer(f)
        w.writerow(INVENTORY_FIELDS)
        for ex in examples:
            w.writerow([ex, "", "", "", ""])


def read_inventory(path: Path, finishes: Iterable[str] | None = None) -> list[InventoryRow]:
    """Read and check an inventory sheet; SessionError lists every problem."""
    if not path.is_file():
        raise SessionError(f"{path} not found: fill in the inventory before matching")
    lines = [
        ln
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    rows, problems = [], []
    known = set(finishes) if finishes is not None else None
    for i, r in enumerate(csv.DictReader(lines), start=2):
        pid = (r.get("part_id") or "").strip()
        if not pid:
            continue
        where = f"inventory line {i} ({pid})"
        try:
            cls = UNKNOWN if pid == UNKNOWN else class_name(pid)
            parsed = None if pid == UNKNOWN else parse_part_id(pid)
        except ValueError as exc:
            problems.append(f"{where}: {exc}")
            continue
        finish = (r.get("finish") or "").strip()
        if known is not None and finish and finish not in known:
            problems.append(f"{where}: unknown finish {finish!r}")
        raw_len = (r.get("length_mm") or "").strip()
        length = None
        if raw_len:
            try:
                length = float(raw_len)
            except ValueError:
                problems.append(f"{where}: length_mm {raw_len!r} is not a number")
        elif parsed is not None and parsed.is_screw:
            length = parsed.length  # nominal, if the caliper value was not written down
        try:
            count = int((r.get("count") or "1").strip() or 1)
        except ValueError:
            problems.append(f"{where}: count must be a whole number")
            count = 1
        rows.append(InventoryRow(pid, cls, finish, length, count, (r.get("notes") or "").strip()))
    if problems:
        raise SessionError("\n".join(problems))
    return rows


def session_images(d: Path) -> list[Path]:
    return sorted(p for p in (d / "images").glob("*.png"))


def finish_of_class(rows: Iterable[InventoryRow]) -> Mapping[str, str]:
    """class -> finish when every inventory row of that class has the same finish."""
    seen: dict[str, set[str]] = {}
    for r in rows:
        seen.setdefault(r.class_name, set()).add(r.finish or "unknown")
    return {c: next(iter(f)) for c, f in seen.items() if len(f) == 1}
