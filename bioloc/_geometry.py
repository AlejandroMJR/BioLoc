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
    fallback_margin_m: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the localization bounds prepared by the recording loader."""
    map_info = recording.metadata.get("map_info")
    if map_info is not None and "plot_min" in map_info and "plot_max" in map_info:
        return (
            np.asarray(map_info["plot_min"], dtype=float),
            np.asarray(map_info["plot_max"], dtype=float),
        )
    positions = np.asarray(recording.microphone_positions, dtype=float)
    lower = np.min(positions, axis=0)
    upper = np.max(positions, axis=0)
    margin = float(fallback_margin_m)
    return lower - margin, upper + margin
