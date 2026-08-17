"""Source-mask reconstruction and classification."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ._kernels.reconstruction import (
    build_source_tf_maps_aligned_to_mic,
    shift_source_maps_between_mics,
    solve_relative_delays_from_source,
)
from ._kernels.tdoa import (
    build_soft_masks_from_projected_maps,
    fill_outside_source_mask_with_white_noise,
    morph_source_maps,
)
from .association import _raw_peak_mapping
from ._timefreq import hz_seconds_to_stft_sigma, hz_seconds_to_tf_size
from .config import ReconstructionConfig
from .models import (
    AssociationResult,
    ClassificationResult,
    LocalizationResult,
    LocalTDOAResult,
    PeakExtractionResult,
    ReconstructionResult,
    ReconstructedSource,
    Recording,
    SourceClassification,
)

_MORPH_THRESHOLD = 0.0
_MORPH_CORE_GAIN = 1.0
_MORPH_CONTEXT_GAIN = 1.0
_MORPH_MIN_AREA = 4
_MORPH_POST_CLOSE_FREQ_HZ = 800.0
_MORPH_POST_CLOSE_TIME_SECONDS = 0.1
_MASK_TAPER_FREQ_HZ = 0.0
_MASK_TAPER_TIME_SECONDS = 0.0
_FILL_MODE = "white_noise"
_NOISE_LEVEL = 0.0
_TARGET_SNR_DB = 0.0
_WHITE_NOISE_LEVEL_MODE = "inside_noise_floor"
_RANDOM_STATE = 0


class SourceClassifier(Protocol):
    """Minimal interface accepted by :func:`classify_reconstructed_sources`."""

    name: str

    def classify(self, signal: np.ndarray, sample_rate: float) -> Any:
        ...


def reconstruct_sources(
    recording: Recording,
    peaks: PeakExtractionResult,
    local_tdoas: LocalTDOAResult,
    association: AssociationResult,
    localization: LocalizationResult,
    config: ReconstructionConfig | None = None,
) -> ReconstructionResult:
    """Create final aligned masks and reconstruct every associated source."""
    config = config or ReconstructionConfig()
    if not association.raw_sources:
        return ReconstructionResult(
            sources=[],
            sample_rate=recording.sample_rate,
            mask_hop_samples=local_tdoas.hop_size,
            mask_frequency_bands_hz=local_tdoas.frequency_bands_hz,
            display_masks=[],
        )

    raw_peak_mapping = _raw_peak_mapping(peaks)
    positions = [source.position for source in localization.sources]
    n_bands = len(local_tdoas.frequency_bands_hz)
    n_frames = len(local_tdoas.frame_times_seconds)
    source_maps, source_reference_mics, alignment_debug = (
        build_source_tf_maps_aligned_to_mic(
            sources=association.raw_sources,
            tdoas_hat_pairs=raw_peak_mapping,
            n_bands=n_bands,
            n_frames=n_frames,
            n_mics=recording.n_microphones,
            hop=local_tdoas.hop_size,
            mic_positions=recording.microphone_positions,
            source_positions=positions,
            fs=recording.sample_rate,
            c=local_tdoas.speed_of_sound_m_s,
            reject_ambiguous_pair_tf=config.reject_ambiguous_pair_tf,
            pair_ambiguity_tolerance_samples=(
                config.pair_ambiguity_tolerance_samples
            ),
        )
    )
    rejected_counts = [
        int(item.get("n_ambiguous_pair_tf_rejected", 0))
        for item in alignment_debug
    ]
    fallback_maps = None
    if any(rejected_counts):
        fallback_maps, _, _ = build_source_tf_maps_aligned_to_mic(
            sources=association.raw_sources,
            tdoas_hat_pairs=raw_peak_mapping,
            n_bands=n_bands,
            n_frames=n_frames,
            n_mics=recording.n_microphones,
            hop=local_tdoas.hop_size,
            mic_positions=recording.microphone_positions,
            source_positions=positions,
            fs=recording.sample_rate,
            c=local_tdoas.speed_of_sound_m_s,
            reject_ambiguous_pair_tf=False,
        )

    final_maps = _morph_maps(
        source_maps,
        local_tdoas,
        recording.sample_rate,
        config,
        minimal=True,
    )
    soft_masks = build_soft_masks_from_projected_maps(final_maps, power=1.0)
    display_maps = _morph_maps(
        source_maps,
        local_tdoas,
        recording.sample_rate,
        config,
        minimal=False,
    )
    fallback_soft_masks = None
    fallback_display_maps = None
    if fallback_maps is not None:
        fallback_final_maps = _morph_maps(
            fallback_maps,
            local_tdoas,
            recording.sample_rate,
            config,
            minimal=True,
        )
        fallback_soft_masks = build_soft_masks_from_projected_maps(
            fallback_final_maps,
            power=1.0,
        )
        fallback_display_maps = _morph_maps(
            fallback_maps,
            local_tdoas,
            recording.sample_rate,
            config,
            minimal=False,
        )

    source_delays = []
    used_mics_by_source = []
    for source in association.raw_sources:
        delays, used_mics = solve_relative_delays_from_source(
            source,
            n_mics=recording.n_microphones,
            reference="mean",
        )
        if delays is None:
            delays = np.zeros(recording.n_microphones, dtype=float)
        source_delays.append(np.asarray(delays, dtype=float))
        used_mics_by_source.append(list(used_mics))
    source_delays_array = np.stack(source_delays, axis=0)

    reference_mics = _closest_reference_microphones(
        recording,
        positions,
        used_mics_by_source,
        source_reference_mics,
    )
    reconstruction_signals = (
        local_tdoas.filtered_signals
        if local_tdoas.filtered_signals is not None
        else recording.signals
    )
    selected_signals = [reconstruction_signals[index] for index in reference_mics]
    aligned_masks, _ = shift_source_maps_between_mics(
        soft_masks,
        association.raw_sources,
        from_mic=source_reference_mics,
        target_mics=reference_mics,
        n_mics=recording.n_microphones,
        hop=local_tdoas.hop_size,
    )
    display_masks, _ = shift_source_maps_between_mics(
        display_maps,
        association.raw_sources,
        from_mic=source_reference_mics,
        target_mics=reference_mics,
        n_mics=recording.n_microphones,
        hop=local_tdoas.hop_size,
    )
    aligned_fallback_masks = None
    aligned_fallback_display_masks = None
    if fallback_soft_masks is not None:
        aligned_fallback_masks, _ = shift_source_maps_between_mics(
            fallback_soft_masks,
            association.raw_sources,
            from_mic=source_reference_mics,
            target_mics=reference_mics,
            n_mics=recording.n_microphones,
            hop=local_tdoas.hop_size,
        )
    if fallback_display_maps is not None:
        aligned_fallback_display_masks, _ = shift_source_maps_between_mics(
            fallback_display_maps,
            association.raw_sources,
            from_mic=source_reference_mics,
            target_mics=reference_mics,
            n_mics=recording.n_microphones,
            hop=local_tdoas.hop_size,
        )

    reconstruction_frame_size = int(config.reconstruction_frame_size)
    if reconstruction_frame_size <= 0:
        raise ValueError("reconstruction_frame_size must be positive")
    reconstruction_overlap = reconstruction_frame_size // 2
    reconstruction_hop = reconstruction_frame_size - reconstruction_overlap
    taper_frequency, taper_time = hz_seconds_to_stft_sigma(
        _MASK_TAPER_FREQ_HZ,
        _MASK_TAPER_TIME_SECONDS,
        recording.sample_rate,
        reconstruction_frame_size,
        reconstruction_hop,
    )
    reconstructed, diagnostics = fill_outside_source_mask_with_white_noise(
        selected_signals,
        soft_masks,
        recording.sample_rate,
        reconstruction_frame_size,
        reconstruction_overlap,
        reconstruction_frame_size,
        Omega_list=local_tdoas.algorithm_frequency_bins,
        source_delays=source_delays_array,
        mask_reference_mic=source_reference_mics,
        target_mics=reference_mics,
        mask_hop=local_tdoas.hop_size,
        mask_nperseg=local_tdoas.frame_size,
        mask_nfft=local_tdoas.frame_size,
        mask_threshold=0.0,
        border_smooth_freq_bins=taper_frequency,
        border_smooth_time_frames=taper_time,
        noise_level=_NOISE_LEVEL,
        target_snr_db=_TARGET_SNR_DB,
        replacement_mode=_FILL_MODE,
        white_noise_level_mode=_WHITE_NOISE_LEVEL_MODE,
        mic_roi_masks_high_res=None,
        roi_nperseg=1024,
        roi_noverlap=512,
        random_state=_RANDOM_STATE,
        fallback_source_masks=fallback_soft_masks,
        fallback_source_mask_allowed=[count > 0 for count in rejected_counts],
        return_diagnostics=True,
    )
    for diagnostic in diagnostics:
        if diagnostic.get("mask_fallback_used") and aligned_fallback_masks is not None:
            source_id = int(diagnostic["source_idx"])
            aligned_masks[source_id] = aligned_fallback_masks[source_id]
            if aligned_fallback_display_masks is not None:
                display_masks[source_id] = aligned_fallback_display_masks[source_id]

    sources = [
        ReconstructedSource(
            source_id=source_id,
            signal=np.asarray(reconstructed[source_id], dtype=float),
            reconstruction_mask=np.asarray(aligned_masks[source_id], dtype=np.float32),
            reference_microphone=int(reference_mics[source_id]),
            relative_delays_samples=source_delays_array[source_id],
        )
        for source_id in range(len(reconstructed))
    ]
    return ReconstructionResult(
        sources=sources,
        sample_rate=recording.sample_rate,
        mask_hop_samples=local_tdoas.hop_size,
        mask_frequency_bands_hz=local_tdoas.frequency_bands_hz,
        display_masks=[np.asarray(mask, dtype=np.float32) for mask in display_masks],
    )


def classify_reconstructed_sources(
    reconstruction: ReconstructionResult,
    classifier: SourceClassifier,
) -> ClassificationResult:
    """Classify reconstructed signals without coupling classification to the pipeline."""
    if hasattr(classifier, "classify_source"):
        classifier.mask_hop_samples = reconstruction.mask_hop_samples or 1
    outputs = []
    for source in reconstruction.sources:
        if hasattr(classifier, "classify_source"):
            raw = classifier.classify_source(source, reconstruction.sample_rate)
        else:
            raw = classifier.classify(source.signal, reconstruction.sample_rate)
        label, confidence, detections = _normalize_classification(raw)
        outputs.append(
            SourceClassification(
                source_id=source.source_id,
                label=label,
                confidence=confidence,
                detections=detections,
                input_signal=(
                    np.asarray(raw.get("input_signal"), dtype=float)
                    if isinstance(raw, dict) and raw.get("input_signal") is not None
                    else source.signal
                ),
            )
        )
    return ClassificationResult(
        sources=outputs,
        classifier_name=getattr(classifier, "name", classifier.__class__.__name__),
        sample_rate=reconstruction.sample_rate,
    )


def _normalize_classification(raw: Any) -> tuple[str | None, float | None, Any]:
    if isinstance(raw, dict):
        return raw.get("label"), raw.get("confidence"), raw.get("detections", raw)
    if isinstance(raw, tuple) and len(raw) >= 2:
        return str(raw[0]), float(raw[1]), raw[2] if len(raw) > 2 else raw
    return None, None, raw


def _morph_maps(
    maps: list[np.ndarray],
    local_tdoas: LocalTDOAResult,
    sample_rate: float,
    config: ReconstructionConfig,
    *,
    minimal: bool,
) -> list[np.ndarray]:
    def size(frequency_hz: float, time_seconds: float) -> tuple[int, int]:
        return hz_seconds_to_tf_size(
            frequency_hz,
            time_seconds,
            local_tdoas.frequency_bands_hz,
            local_tdoas.hop_size,
            sample_rate,
        )

    if minimal:
        open_size = close_size = dilate_size = post_close_size = (0, 0)
        min_area = 2
    else:
        open_size = close_size = dilate_size = (0, 0)
        post_close_size = size(
            _MORPH_POST_CLOSE_FREQ_HZ,
            _MORPH_POST_CLOSE_TIME_SECONDS,
        )
        min_area = _MORPH_MIN_AREA
    return morph_source_maps(
        maps,
        threshold=_MORPH_THRESHOLD,
        open_size=open_size,
        close_size=close_size,
        min_area=min_area,
        area_connectivity=2,
        area_filter_position="after_close",
        dilate_size=dilate_size,
        post_area_close_size=post_close_size,
        core_gain=_MORPH_CORE_GAIN,
        context_gain=_MORPH_CONTEXT_GAIN,
    )


def _closest_reference_microphones(
    recording: Recording,
    positions: list[np.ndarray | None],
    used_mics_by_source: list[list[int]],
    fallback_mics: list[int | None],
) -> list[int]:
    output = []
    for source_id, position in enumerate(positions):
        used = used_mics_by_source[source_id]
        if position is None or not used:
            fallback = fallback_mics[source_id]
            output.append(0 if fallback is None else int(fallback))
            continue
        used_array = np.asarray(used, dtype=int)
        distances = np.linalg.norm(
            recording.microphone_positions[used_array] - np.asarray(position)[None, :],
            axis=1,
        )
        output.append(int(used_array[int(np.argmin(distances))]))
    return output
