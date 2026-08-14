"""Compact deterministic snapshots for legacy/new pipeline comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pipeline_snapshot(result) -> dict:
    """Return JSON-serializable intermediate values suitable for comparison."""
    return {
        "roi_active_cells_by_pair": {
            _pair_name(pair): int(np.count_nonzero(mask))
            for pair, mask in result.roi.masks_by_pair.items()
        },
        "local_tdoa_counts_by_pair": {
            _pair_name(pair): int(values.delays.size)
            for pair, values in result.local_tdoas.pairs.items()
        },
        "local_tdoa_first_values_by_pair": {
            _pair_name(pair): np.asarray(values.delays[:10], dtype=float).tolist()
            for pair, values in result.local_tdoas.pairs.items()
        },
        "peak_delays_by_pair": {
            _pair_name(pair): [float(peak.delay_samples) for peak in values.peaks]
            for pair, values in result.peaks.pairs.items()
        },
        "associated_sources": [
            [
                {
                    "pair": list(member.pair),
                    "peak_index": member.peak_index,
                    "delay_samples": member.delay_samples,
                }
                for member in source.members
            ]
            for source in result.association.sources
        ],
        "locations": [
            {
                "position": (
                    None
                    if source.position is None
                    else np.asarray(source.position).tolist()
                ),
                "uncertainty_m": source.uncertainty_m,
                "residual_samples": source.residual_samples,
                "boundary_contact": source.boundary_contact,
            }
            for source in result.localization.sources
        ],
        "reconstruction": [
            {
                "reference_microphone": source.reference_microphone,
                "mask_shape": list(source.reconstruction_mask.shape),
                "mask_sum": float(np.sum(source.reconstruction_mask)),
                "signal_samples": int(source.signal.size),
                "signal_rms": float(np.sqrt(np.mean(np.square(source.signal)))),
            }
            for source in result.reconstruction.sources
        ],
        "classification": [
            {
                "label": source.label,
                "confidence": source.confidence,
            }
            for source in result.classification.sources
        ],
    }


def save_pipeline_snapshot(result, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(pipeline_snapshot(result), indent=2, sort_keys=True) + "\n")
    return path


def _pair_name(pair) -> str:
    return f"{pair[0]}-{pair[1]}"
