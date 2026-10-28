"""Minimal PNG writer/reader (stdlib zlib + numpy) usable inside Blender.

Supports 8-bit RGB / grayscale and 16-bit grayscale (instance masks), no interlacing.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: str | os.PathLike[str], image: np.ndarray, level: int = 3) -> None:
    """Write uint8 (H, W), uint8 (H, W, 3) or uint16 (H, W) as PNG."""
    img = np.ascontiguousarray(image)
    if img.ndim == 2:
        colour_type = 0
    elif img.ndim == 3 and img.shape[2] == 3:
        colour_type = 2
    else:
        raise ValueError(f"unsupported shape {img.shape}")
    if img.dtype == np.uint8:
        depth = 8
    elif img.dtype == np.uint16 and colour_type == 0:
        depth = 16
        img = img.astype(">u2")
    else:
        raise ValueError(f"unsupported dtype {img.dtype} for shape {img.shape}")
    h, w = img.shape[:2]
    rows = img.reshape(h, -1).view(np.uint8)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()  # filter 0 per row
    header = struct.pack(">IIBBBBB", w, h, depth, colour_type, 0, 0, 0)
    data = (
        _SIG
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, level))
        + _chunk(b"IEND", b"")
    )
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _unfilter(raw: bytes, h: int, stride: int, bpp: int) -> np.ndarray:
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    pos = 0
    for y in range(h):
        ftype = raw[pos]
        line = np.frombuffer(raw, np.uint8, stride, pos + 1).astype(np.int32)
        pos += stride + 1
        if ftype == 0:
            cur = line
        elif ftype == 2:
            cur = (line + prev) & 0xFF
        else:  # 1 (sub), 3 (average), 4 (paeth) need sequential decoding
            cur = np.zeros(stride, np.int32)
            for x in range(stride):
                a = cur[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                if ftype == 1:
                    pred = a
                elif ftype == 3:
                    pred = (a + b) // 2
                else:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur[x] = (line[x] + pred) & 0xFF
        out[y] = cur
        prev = cur
    return out


def read_png(path: str | os.PathLike[str]) -> np.ndarray:
    """Read the PNGs written by write_png (and other non-interlaced 8/16-bit gray/RGB/RGBA PNGs)."""
    data = Path(path).read_bytes()
    if data[:8] != _SIG:
        raise ValueError("not a PNG")
    pos, idat = 8, b""
    w = h = depth = colour_type = 0
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + n]
        pos += 12 + n
        if tag == b"IHDR":
            w, h, depth, colour_type, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if interlace:
                raise ValueError("interlaced PNG not supported")
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[colour_type]
    bpp = channels * depth // 8
    rows = _unfilter(zlib.decompress(idat), h, w * bpp, bpp)
    if depth == 16:
        img = rows.view(">u2").astype(np.uint16).reshape(h, w, channels)
    else:
        img = rows.reshape(h, w, channels)
    return img[..., 0] if channels == 1 else img
