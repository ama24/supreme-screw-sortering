"""HDRI library: the .hdr/.exr files in paths.hdri_dir, each switched on or off (hdri.yaml).

Only local files are used, nothing is ever downloaded: the tool runs offline. Shared by all recipes,
like the surface photos. Pure Python: importable inside Blender.
"""

from __future__ import annotations

from pathlib import Path

from screwsynth.common.io import read_yaml, write_yaml

SUFFIXES = (".hdr", ".exr")
LIBRARY = "hdri.yaml"

# where to get HDRIs (downloaded by hand, once), shown in the GUI and the README
SOURCES = (
    ("Poly Haven", "https://polyhaven.com/hdris", "CC0: free for any use; 1K-16K, 2K-4K is enough"),
    ("ambientCG", "https://ambientcg.com/list?type=hdri", "CC0: free for any use"),
    (
        "sIBL Archive",
        "http://www.hdrlabs.com/sibl/archive.html",
        "free for non-commercial use; check each file's licence",
    ),
)


def files(hdri_dir: Path) -> list[str]:
    """Every HDRI file in the folder, sorted by name."""
    d = Path(hdri_dir)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)


def disabled(hdri_dir: Path) -> set[str]:
    path = Path(hdri_dir) / LIBRARY
    if not path.is_file():
        return set()
    return set((read_yaml(path) or {}).get("disabled") or [])


def set_enabled(hdri_dir: Path, name: str, on: bool) -> None:
    off = disabled(hdri_dir)
    if on:
        off.discard(name)
    else:
        off.add(name)
    Path(hdri_dir).mkdir(parents=True, exist_ok=True)
    write_yaml(Path(hdri_dir) / LIBRARY, {"disabled": sorted(off)})


def usable(hdri_dir: Path) -> list[str]:
    """The files that are switched on, sorted by name (the renderer spreads layouts over them)."""
    off = disabled(hdri_dir)
    return [f for f in files(hdri_dir) if f not in off]
