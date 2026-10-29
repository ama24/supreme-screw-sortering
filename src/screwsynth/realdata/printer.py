"""Homing / parking the Ender 3 bed over USB serial before a capture.

Only homing Y and parking Y at the configured position are ever sent (plus waiting for moves to
finish). Any other G-code is refused: other motion commands need the operator's explicit approval
and are not part of this tool.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("screwsynth.realdata.printer")


class PrinterError(RuntimeError):
    pass


@dataclass
class PrinterSettings:
    port: str | None = None  # e.g. /dev/ttyUSB0 on the Pi, COM5 on Windows; None = no printer
    baud: int = 115200
    park_y_mm: float | None = None  # None: home only
    feed_mm_min: float = 3000.0
    timeout_s: float = 60.0

    @classmethod
    def from_config(cls, cfg: dict) -> PrinterSettings:
        p = dict(cfg or {})
        d = cls()
        return cls(
            port=p.get("port"),
            baud=int(p.get("baud", d.baud)),
            park_y_mm=None if p.get("park_y_mm") is None else float(p["park_y_mm"]),
            feed_mm_min=float(p.get("feed_mm_min", d.feed_mm_min)),
            timeout_s=float(p.get("timeout_s", d.timeout_s)),
        )


def allowed(line: str, st: PrinterSettings) -> bool:
    """The whitelist: home Y, park Y at the configured position, wait for moves, report position."""
    g = line.strip().upper()
    if g in ("G28 Y", "M400", "M114"):
        return True
    m = re.fullmatch(r"G0 Y(-?\d+(?:\.\d+)?) F(\d+(?:\.\d+)?)", g)
    return bool(m and st.park_y_mm is not None and abs(float(m[1]) - st.park_y_mm) < 1e-6)


def park_commands(st: PrinterSettings) -> list[str]:
    cmds = ["G28 Y", "M400"]
    if st.park_y_mm is not None:
        cmds += [f"G0 Y{st.park_y_mm:g} F{st.feed_mm_min:g}", "M400"]
    return cmds


class Printer:
    """Line-based G-code over serial, restricted to `allowed` commands."""

    def __init__(self, st: PrinterSettings, opener: Callable | None = None) -> None:
        self.st = st
        if not st.port:
            raise PrinterError("no printer port configured (realdata.printer.port)")
        if opener is None:
            try:
                import serial  # pyserial
            except ImportError as exc:  # pragma: no cover - depends on the Pi
                raise PrinterError("pip install pyserial to drive the printer") from exc
            opener = serial.Serial
        self.ser = opener(st.port, st.baud, timeout=1)
        time.sleep(0.1)

    def send(self, line: str) -> list[str]:
        if not allowed(line, self.st):
            raise PrinterError(f"refusing to send {line!r}: only homing/parking Y is allowed")
        self.ser.write((line.strip() + "\n").encode("ascii"))
        replies, t0 = [], time.monotonic()
        while time.monotonic() - t0 < self.st.timeout_s:
            raw = self.ser.readline()
            if not raw:
                continue
            text = raw.decode("ascii", errors="replace").strip()
            replies.append(text)
            if text.lower().startswith("ok"):
                return replies
            if text.lower().startswith(("error", "!!")):
                raise PrinterError(f"{line}: {text}")
        raise PrinterError(f"{line}: no 'ok' within {self.st.timeout_s:.0f} s")

    def home_and_park(self) -> None:
        for cmd in park_commands(self.st):
            log.info("printer: %s", cmd)
            self.send(cmd)

    def close(self) -> None:
        self.ser.close()
