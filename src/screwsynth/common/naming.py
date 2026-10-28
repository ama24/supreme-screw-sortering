"""Part-id and class-name builders and parsers.

Part ids (see CLAUDE.md):
  screws        <FAMILY>_<SIZE>x<LENGTH>        DIN912_M4x16
  nuts/washers  <FAMILY>_<SIZE>                 DIN934_M4, DIN125A_M4
  T-nuts        TNUT_<SLIDE|DROP>_S<slot>_<SIZE> TNUT_DROP_S6_M4
Class name = part id without the length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KEYPOINTS: tuple[str, str, str] = ("head_top", "bearing", "tip")
TNUT_FAMILY = "TNUT"
TNUT_KINDS: tuple[str, str] = ("SLIDE", "DROP")

_NUM = r"\d+(?:\.\d+)?"
_SIZE = rf"M{_NUM}"
_FAMILY = r"[A-Z][A-Z0-9]*"
_SCREW_RE = re.compile(rf"^(?P<family>{_FAMILY})_(?P<size>{_SIZE})x(?P<length>{_NUM})$")
_TNUT_RE = re.compile(
    rf"^{TNUT_FAMILY}_(?P<kind>{'|'.join(TNUT_KINDS)})_S(?P<slot>\d+)_(?P<size>{_SIZE})$"
)
_PLAIN_RE = re.compile(rf"^(?P<family>{_FAMILY})_(?P<size>{_SIZE})$")


@dataclass(frozen=True)
class PartName:
    """Parsed part id."""

    family: str
    size: str
    length: float | None = None
    tnut_kind: str | None = None
    slot: int | None = None

    @property
    def is_screw(self) -> bool:
        return self.length is not None

    @property
    def is_tnut(self) -> bool:
        return self.tnut_kind is not None

    @property
    def part_id(self) -> str:
        if self.tnut_kind is not None:
            assert self.slot is not None
            return tnut_id(self.tnut_kind, self.slot, self.size)
        if self.length is not None:
            return screw_id(self.family, self.size, self.length)
        return plain_id(self.family, self.size)

    @property
    def class_name(self) -> str:
        if self.tnut_kind is not None:
            return self.part_id
        return plain_id(self.family, self.size)


def _fmt_num(x: float) -> str:
    """Format a number without a trailing .0 (16 -> '16', 2.5 -> '2.5')."""
    if x <= 0:
        raise ValueError(f"expected a positive number, got {x}")
    return f"{float(x):g}"


def format_size(size: str | float) -> str:
    """Normalise a thread size: 4, '4', 'm4', 'M4' -> 'M4'; 2.5 -> 'M2.5'."""
    s = str(size).strip().upper()
    if s.startswith("M"):
        s = s[1:]
    try:
        value = float(s)
    except ValueError:
        raise ValueError(f"invalid thread size: {size!r}") from None
    return "M" + _fmt_num(value)


def size_value(size: str | float) -> float:
    """Nominal diameter in mm: 'M2.5' -> 2.5."""
    return float(format_size(size)[1:])


def _check_family(family: str) -> str:
    if not re.fullmatch(_FAMILY, family):
        raise ValueError(
            f"invalid family {family!r}: use upper-case letters and digits, e.g. DIN912"
        )
    if family == TNUT_FAMILY:
        raise ValueError("use tnut_id() for T-nuts")
    return family


def screw_id(family: str, size: str | float, length: float) -> str:
    return f"{_check_family(family)}_{format_size(size)}x{_fmt_num(length)}"


def plain_id(family: str, size: str | float) -> str:
    """Id of a part without a length (nut, washer); also the class name of a screw."""
    return f"{_check_family(family)}_{format_size(size)}"


def tnut_id(kind: str, slot: int, size: str | float) -> str:
    kind = kind.upper()
    if kind not in TNUT_KINDS:
        raise ValueError(f"T-nut kind must be one of {TNUT_KINDS}, got {kind!r}")
    if int(slot) != slot or slot <= 0:
        raise ValueError(f"invalid slot width {slot!r}")
    return f"{TNUT_FAMILY}_{kind}_S{int(slot)}_{format_size(size)}"


def parse_part_id(part_id: str) -> PartName:
    """Parse a part id into its components; ValueError if it breaks the naming convention."""
    if m := _TNUT_RE.match(part_id):
        return PartName(
            family=TNUT_FAMILY,
            size=format_size(m["size"]),
            tnut_kind=m["kind"],
            slot=int(m["slot"]),
        )
    if m := _SCREW_RE.match(part_id):
        if m["family"] == TNUT_FAMILY:
            raise ValueError(f"invalid T-nut id: {part_id!r}")
        return PartName(family=m["family"], size=format_size(m["size"]), length=float(m["length"]))
    if m := _PLAIN_RE.match(part_id):
        if m["family"] == TNUT_FAMILY:
            raise ValueError(f"invalid T-nut id: {part_id!r}")
        return PartName(family=m["family"], size=format_size(m["size"]))
    raise ValueError(f"part id does not follow the naming convention: {part_id!r}")


def class_name(part_id: str) -> str:
    """Class name of a part: its id without the length (DIN912_M4x16 -> DIN912_M4)."""
    return parse_part_id(part_id).class_name
