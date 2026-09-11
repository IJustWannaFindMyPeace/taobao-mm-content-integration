"""Versioned, label-independent user sampling over signed 64-bit IDs."""
import numpy as np


def mix64(ids, seed):
    x = np.asarray(ids, dtype=np.int64).view(np.uint64) ^ np.uint64(seed)
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def user_masks(ids):
    selected = mix64(ids, 42) % 1000 == 0
    validation = mix64(ids, 2026) % 5 == 0
    return selected & ~validation, selected & validation
