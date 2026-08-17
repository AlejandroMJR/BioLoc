"""Internal microphone-pair and bounds helpers."""

from __future__ import annotations

import numpy as np

from .models import Pair


def nearest_neighbor_pairs(
    microphone_positions: np.ndarray,
    k_neighbors: int,
) -> list[Pair]:
    positions = np.asarray(microphone_positions, dtype=float)
    n_mics = positions.shape[0]
    if n_mics < 2:
        raise ValueError("at least two microphones are required")
    k_neighbors = min(int(k_neighbors), n_mics - 1)
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    pair_set: set[Pair] = set()
    for first in range(n_mics):
        distances = np.linalg.norm(positions - positions[first], axis=1)
        for second in np.argsort(distances)[1 : k_neighbors + 1]:
            pair_set.add(
                (first, int(second))
                if first < second
                else (int(second), first)
            )
    return sorted(pair_set)


def recording_environment_bounds(
    recording,
    *,
    margin_m: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return square search bounds around the microphone array."""
    positions = np.asarray(recording.microphone_positions, dtype=float)
    margin = float(margin_m)
    if margin < 0:
        raise ValueError("environment_margin_m must be non-negative")
    position_min = np.min(positions, axis=0)
    position_max = np.max(positions, axis=0)
    center = 0.5 * (position_min + position_max)
    side = max(float(np.max(position_max - position_min)) + 2.0 * margin, 1.0)
    return center - side / 2.0, center + side / 2.0
