"""Region-of-interest extraction."""

from __future__ import annotations

import numpy as np

from ._kernels.tdoa import build_shared_maad_roi_masks

from ._geometry import nearest_neighbor_pairs
from .config import LocalTDOAConfig, ROIConfig
from .local_tdoa import _filter_signals
from .models import ROIResult, Recording


def extract_regions_of_interest(
    recording: Recording,
    config: ROIConfig | None = None,
    local_tdoa_config: LocalTDOAConfig | None = None,
) -> ROIResult:
    """Extract MAAD time-frequency ROI masks for each selected mic pair.

    Pair masks use the union of both microphone masks, matching the current
    pipeline's ``pair_mode='max'`` behavior. Frequency bands and algorithm
    frame times intentionally belong to :func:`estimate_local_tdoas`.
    """
    config = config or ROIConfig()
    local_tdoa_config = local_tdoa_config or LocalTDOAConfig()
    filtered_signals = _filter_signals(
        recording.signals,
        recording.sample_rate,
        local_tdoa_config,
    )
    pair_variant_masks, microphone_masks = build_shared_maad_roi_masks(
        filtered_signals,
        recording.sample_rate,
        nperseg_maad=config.nperseg,
        noverlap_maad=config.noverlap,
        dB_max=config.db_range,
        smooth_std=config.smooth_std,
        pair_roi_kwargs={
            "bin_std": config.bin_std,
            "bin_per": config.bin_per,
            "min_roi": config.min_roi,
            "max_roi": config.max_roi,
        },
        mic_roi_kwargs={
            "bin_std": config.bin_std,
            "bin_per": config.bin_per,
            "min_roi": config.min_roi,
            "max_roi": config.max_roi,
        },
    )
    pairs = nearest_neighbor_pairs(
        recording.microphone_positions,
        local_tdoa_config.k_neighbors,
    )
    masks_by_pair = {
        pair: np.asarray(
            pair_variant_masks[pair[0]] | pair_variant_masks[pair[1]],
            dtype=bool,
        )
        for pair in pairs
    }
    return ROIResult(
        masks_by_pair=masks_by_pair,
        microphone_masks=[np.asarray(mask, dtype=bool) for mask in microphone_masks],
    )
