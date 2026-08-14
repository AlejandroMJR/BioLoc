"""Smoothed TDOA histograms and matching-pursuit peak extraction."""

from __future__ import annotations

import numpy as np

from ._kernels.tdoa import (
    matching_pursuit_tdoa,
    smooth_tdoa_histogram_gaussian_fast,
)
from ._timefreq import attach_timefreq_info_to_peaks
from .config import PeakExtractionConfig
from .models import (
    LocalTDOAResult,
    PairPeakResult,
    PeakExtractionResult,
    TDOAPeak,
)


def extract_tdoa_peaks(
    local_tdoas: LocalTDOAResult,
    config: PeakExtractionConfig | None = None,
) -> PeakExtractionResult:
    """Build one delay histogram per pair and extract supported peaks."""
    config = config or PeakExtractionConfig()
    n_bands = len(local_tdoas.frequency_bands_hz)
    n_frames = len(local_tdoas.frame_times_seconds)
    results = {}
    for pair, observations in local_tdoas.pairs.items():
        if observations.delays.size == 0:
            continue
        max_delay = int(observations.max_delay_samples)
        length = 2 * max_delay + 1
        t_min, t_max = -max_delay, max_delay + 1
        narrow = max(
            2,
            int(2 * max_delay / config.narrow_width_divisor),
        )
        wide = max(narrow, int(narrow * config.wide_to_narrow_ratio))
        histogram, delay_bins = smooth_tdoa_histogram_gaussian_fast(
            tau_obs=np.asarray(observations.delays, dtype=np.float32),
            L=length,
            t_min=t_min,
            t_max=t_max,
            sigma_tau=wide / config.histogram_sigma_divisor,
        )
        histogram = _normalize_histogram(histogram)
        peak_delays, _, residual, _ = matching_pursuit_tdoa(
            y=histogram,
            L=length,
            QN=narrow,
            QW=wide,
            t_min=t_min,
            t_max=t_max,
            verbose=False,
        )
        tolerance = max(0, int(narrow / config.support_tolerance_divisor))
        raw_peaks = attach_timefreq_info_to_peaks(
            tau_obs=observations.delays,
            bands=observations.band_indices,
            frames=observations.frame_indices,
            peak_taus=peak_delays,
            tol_tau=tolerance,
            n_bands=n_bands,
            n_frames=n_frames,
        )
        peaks = [
            TDOAPeak(
                delay_samples=float(raw["tau"]),
                supporting_observation_indices=np.asarray(raw["obs_idx"], dtype=int),
                band_support=_band_support(raw, n_bands),
                time_frequency_support=np.asarray(
                    raw.get("tf_support_map", np.zeros((n_bands, n_frames))),
                    dtype=np.float32,
                ),
                raw=raw,
            )
            for raw in raw_peaks
        ]
        if not peaks:
            continue
        results[pair] = PairPeakResult(
            pair=pair,
            histogram=np.asarray(histogram, dtype=np.float32),
            delay_bins=np.asarray(delay_bins, dtype=np.float32),
            peaks=peaks,
            assignment_tolerance_samples=float(tolerance),
            matching_pursuit_residual=np.asarray(residual, dtype=np.float32),
        )
    return PeakExtractionResult(pairs=results)


def _normalize_histogram(histogram: np.ndarray) -> np.ndarray:
    output = np.asarray(histogram, dtype=float)
    output = output / (np.max(output) + 1e-12)
    output = output - np.median(output)
    output = output / (np.max(output) + 1e-12)
    return np.maximum(output, 0.0)


def _band_support(raw_peak: dict, n_bands: int) -> np.ndarray:
    support = np.zeros(n_bands, dtype=np.float32)
    for band, count in raw_peak.get("band_counts", {}).items():
        if 0 <= int(band) < n_bands:
            support[int(band)] = float(count)
    return support
