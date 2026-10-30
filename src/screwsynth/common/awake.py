"""Keep the computer awake during long jobs (renders, training): Windows sleep kills them.

    with keep_awake("rendering dataset_v1"):
        ...

Windows: SetThreadExecutionState (the display may still turn off). macOS: caffeinate. Linux:
systemd-inhibit when available. Elsewhere it does nothing. Pure standard library.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator

log = logging.getLogger("screwsynth.awake")

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake(reason: str = "long job") -> Iterator[bool]:
    """Prevent system sleep inside the block. Yields True if a sleep blocker is active."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ok = bool(kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))
        if ok:
            log.info("keeping the computer awake while %s", reason)
        try:
            yield ok
        finally:
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        return
    cmd = None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", "-w", str(os.getpid())]
    elif shutil.which("systemd-inhibit"):
        cmd = ["systemd-inhibit", f"--why={reason}", "--what=sleep", "sleep", "infinity"]
    proc = None
    if cmd:
        with contextlib.suppress(OSError):
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield proc is not None
    finally:
        if proc is not None:
            proc.terminate()
