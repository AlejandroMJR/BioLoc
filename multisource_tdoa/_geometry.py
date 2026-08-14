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


def default_environment_bounds(
    microphone_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative data-derived bounds when none are supplied."""
    positions = np.asarray(microphone_positions, dtype=float)
    lower = np.min(positions, axis=0)
    upper = np.max(positions, axis=0)
    span = np.maximum(upper - lower, 1.0)
    margin = np.max(span)
    return lower - margin, upper + margin
