"""Seeding helpers. Every random choice in the project takes a seed."""

from __future__ import annotations

import hashlib
import random


def seed_everything(seed: int) -> None:
    """Seed Python's random module and, when installed, numpy and torch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % 2**32)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def rng(seed: int) -> random.Random:
    """An independent random.Random instance (pure Python, usable inside Blender)."""
    return random.Random(seed)


def derive_seed(base: int, *keys: object) -> int:
    """Deterministic sub-seed, e.g. derive_seed(run_seed, "scene", 42). Stable across processes."""
    text = "/".join([str(base), *map(str, keys)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 2**63
