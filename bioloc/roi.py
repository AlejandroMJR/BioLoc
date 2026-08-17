"""Region-of-interest extraction."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ._kernels.tdoa import build_shared_maad_roi_masks

from .config import GeneralConfig, ROIConfig
from ._defaults import (
    ROI_DB_RANGE,
    ROI_MAX_ROI,
    ROI_NOVERLAP,
    ROI_NPERSEG,
    ROI_SMOOTH_STD,
)
from .local_tdoa import _filter_signals
from .models import ROIResult, Recording


def extract_regions_of_interest(
    recording: Recording,
    config: ROIConfig | None = None,
    general_config: GeneralConfig | None = None,
) -> ROIResult:
    """Extract MAAD time-frequency ROI masks for each selected mic pair.

    Pair masks use the union of both microphone masks, matching the current
    pipeline's ``pair_mode='max'`` behavior. Frequency bands and algorithm
    frame times intentionally belong to :func:`estimate_local_tdoas`.
    """
    config = config or ROIConfig()
    general_config = general_config or GeneralConfig()
    filtered_signals = _filter_signals(
        recording.signals,
        recording.sample_rate,
        general_config,
    )
    pair_variant_masks, microphone_masks = build_shared_maad_roi_masks(
        filtered_signals,
        recording.sample_rate,
        nperseg_maad=ROI_NPERSEG,
        noverlap_maad=ROI_NOVERLAP,
        dB_max=ROI_DB_RANGE,
        smooth_std=ROI_SMOOTH_STD,
        pair_roi_kwargs={
            "bin_std": config.bin_std,
            "bin_per": config.bin_per,
            "min_roi": config.min_roi,
            "max_roi": ROI_MAX_ROI,
        },
        mic_roi_kwargs={
            "bin_std": config.bin_std,
            "bin_per": config.bin_per,
            "min_roi": config.min_roi,
            "max_roi": ROI_MAX_ROI,
        },
    )
    pairs = list(combinations(range(recording.n_microphones), 2))
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
        filtered_signals=filtered_signals,
        lowcut_hz=general_config.lowcut_hz,
        highcut_hz=general_config.highcut_hz,
        speed_of_sound_m_s=general_config.speed_of_sound_m_s,
        microphone_position_uncertainty_m=(
            general_config.microphone_position_uncertainty_m
        ),
        environment_margin_m=general_config.environment_margin_m,
    )
