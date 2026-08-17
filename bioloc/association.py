"""Feature construction and cross-pair TDOA association."""

from __future__ import annotations

import numpy as np

from ._kernels.association import (
    associate_tdoa_peaks_by_band_cosine_constrained,
    associate_tdoa_peaks_by_timefreq_candidates_geometric,
    associate_tdoa_peaks_by_timefreq_cosine_constrained,
    peak_band_vector,
    peak_timefreq_vector,
)
from ._kernels.localization import prune_geometrically_inconsistent_sources

from ._geometry import recording_environment_bounds
from .config import AssociationConfig
from ._defaults import (
    ASSOCIATION_FRAMES_PER_BIN,
    ASSOCIATION_MAX_EXACT_COMBINATIONS,
)
from .models import (
    AssociatedSource,
    AssociatedTDOA,
    AssociationResult,
    LocalTDOAResult,
    PeakExtractionResult,
    PeakFeature,
    Recording,
)


def build_feature_vectors(
    peaks: PeakExtractionResult,
    local_tdoas: LocalTDOAResult,
    config: AssociationConfig | None = None,
) -> list[PeakFeature]:
    """Return the exact feature vector used to associate every TDOA peak."""
    config = config or AssociationConfig()
    n_bands = len(local_tdoas.frequency_bands_hz)
    n_frames = len(local_tdoas.frame_times_seconds)
    features = []
    for pair, pair_result in peaks.pairs.items():
        for peak_index, peak in enumerate(pair_result.peaks):
            if config.feature == "frequency":
                vector = peak_band_vector(peak.raw, n_bands)
            else:
                vector = peak_timefreq_vector(
                    peak.raw,
                    n_bands=n_bands,
                    n_frames=n_frames,
                    frames_per_bin=ASSOCIATION_FRAMES_PER_BIN,
                )
            features.append(
                PeakFeature(
                    pair=pair,
                    peak_index=peak_index,
                    delay_samples=peak.delay_samples,
                    vector=np.asarray(vector, dtype=np.float32),
                )
            )
    return features


def associate_tdoas(
    peaks: PeakExtractionResult,
    local_tdoas: LocalTDOAResult,
    recording: Recording,
    config: AssociationConfig | None = None,
) -> AssociationResult:
    """Build peak features and group compatible pairwise delays into sources."""
    config = config or AssociationConfig()
    features = build_feature_vectors(peaks, local_tdoas, config)
    raw_pairs = _raw_peak_mapping(peaks)
    n_bands = len(local_tdoas.frequency_bands_hz)
    n_frames = len(local_tdoas.frame_times_seconds)
    minimum_members = max(1, min(recording.n_microphones - 1, 5))
    maximum_delay = max(
        (item.max_delay_samples for item in local_tdoas.pairs.values()),
        default=0,
    )
    max_shift_bins = int(
        np.ceil(
            maximum_delay
            / float(local_tdoas.hop_size * ASSOCIATION_FRAMES_PER_BIN)
        )
    )
    lower, upper = recording_environment_bounds(recording)
    span = float(np.max(upper - lower))
    grid_spacing = max(span / 4.0, 1.0)

    if config.feature == "time_frequency_candidates":
        raw_sources = associate_tdoa_peaks_by_timefreq_candidates_geometric(
            raw_pairs,
            n_bands=n_bands,
            n_frames=n_frames,
            mic_positions=recording.microphone_positions,
            fs=recording.sample_rate,
            c=local_tdoas.speed_of_sound_m_s,
            env_bounds=(lower, upper),
            frames_per_bin=ASSOCIATION_FRAMES_PER_BIN,
            max_shift_bins=max_shift_bins,
            alpha=config.frequency_weight,
            cosine_thresh=config.cosine_threshold,
            inpair_cosine_thresh=config.same_pair_cosine_threshold,
            min_tdoas_per_source=minimum_members,
            max_exact_combinations=ASSOCIATION_MAX_EXACT_COMBINATIONS,
            grid_spacing=grid_spacing,
            n_starts=5,
            max_sources=None,
        )
    elif config.feature == "time_frequency":
        raw_sources = associate_tdoa_peaks_by_timefreq_cosine_constrained(
            raw_pairs,
            n_bands=n_bands,
            n_frames=n_frames,
            frames_per_bin=ASSOCIATION_FRAMES_PER_BIN,
            max_shift_bins=max_shift_bins,
            alpha=config.frequency_weight,
            cosine_thresh=config.cosine_threshold,
            min_tdoas_per_source=minimum_members,
            max_sources=None,
        )
    elif config.feature == "frequency":
        raw_sources = associate_tdoa_peaks_by_band_cosine_constrained(
            raw_pairs,
            n_bands=n_bands,
            cosine_thresh=config.cosine_threshold,
            min_tdoas_per_source=minimum_members,
            max_sources=None,
        )
    else:
        raise ValueError(f"unknown association feature: {config.feature}")

    pruning_reports = []
    if raw_sources and config.prune_geometric_outliers:
        wide_widths = [
            max(2.0, result.assignment_tolerance_samples * 9.0)
            for result in peaks.pairs.values()
        ]
        raw_sources, pruning_reports = prune_geometrically_inconsistent_sources(
            raw_sources,
            mic_positions=recording.microphone_positions,
            fs=recording.sample_rate,
            c=local_tdoas.speed_of_sound_m_s,
            env_bounds=(lower, upper),
            min_tdoas_per_source=minimum_members,
            min_relative_improvement=config.prune_min_relative_improvement,
            max_residual_samples=float(np.mean(wide_widths)) if wide_widths else 3.0,
        )
    sources = _public_sources(raw_sources, features)
    return AssociationResult(
        features=features,
        sources=sources,
        raw_sources=raw_sources,
        pruning_reports=pruning_reports,
        speed_of_sound_m_s=local_tdoas.speed_of_sound_m_s,
        microphone_position_uncertainty_m=(
            local_tdoas.microphone_position_uncertainty_m
        ),
    )


def _raw_peak_mapping(peaks: PeakExtractionResult) -> dict:
    return {
        pair: {
            "peak_taus": np.asarray(
                [peak.delay_samples for peak in result.peaks], dtype=np.float32
            ),
            "peaks": [dict(peak.raw) for peak in result.peaks],
            "hist": result.histogram,
            "bins": result.delay_bins,
            "support_tolerance_samples": result.assignment_tolerance_samples,
        }
        for pair, result in peaks.pairs.items()
    }


def _public_sources(
    raw_sources: list[dict],
    features: list[PeakFeature],
) -> list[AssociatedSource]:
    feature_lookup = {
        (feature.pair, feature.peak_index): feature.vector for feature in features
    }
    output = []
    for source_id, raw_source in enumerate(raw_sources):
        members = []
        for raw_member in raw_source.get("members", []):
            pair = tuple(int(value) for value in raw_member["pair"])
            peak_index = int(raw_member["peak_idx"])
            members.append(
                AssociatedTDOA(
                    pair=pair,
                    peak_index=peak_index,
                    delay_samples=float(raw_member["tau"]),
                    feature=np.asarray(
                        feature_lookup.get((pair, peak_index), []),
                        dtype=np.float32,
                    ),
                    raw=raw_member,
                )
            )
        output.append(
            AssociatedSource(
                source_id=source_id,
                members=members,
                raw=raw_source,
            )
        )
    return output
