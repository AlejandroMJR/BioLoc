"""Local sub-band TDOA estimation."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index as integer_index

import numpy as np
from scipy.signal import butter, filtfilt

from ._kernels.tdoa import (
    build_frequency_bands,
    build_maad_roi_valid_mask_from_signals,
    compute_frame_starts,
    extract_subband_tdoas_windowed_search,
)

from ._geometry import nearest_neighbor_pairs
from .config import GeneralConfig, LocalTDOAConfig
from ._defaults import ROI_NOVERLAP, ROI_NPERSEG
from .models import LocalTDOAResult, Pair, PairLocalTDOAs, ROIResult, Recording


def estimate_local_tdoas(
    recording: Recording,
    roi: ROIResult | None = None,
    config: LocalTDOAConfig | None = None,
    *,
    pairs: Sequence[Pair] | None = None,
) -> LocalTDOAResult:
    """Estimate one local delay observation per valid pair/band/frame cell.

    ``pairs`` selects zero-based microphone pairs instead of using
    ``config.k_neighbors``. Reversed pairs are normalized to ascending index
    order and duplicates are removed, preserving the first occurrence.
    An explicit selection must contain at least one valid pair.
    """
    config = config or LocalTDOAConfig()
    if pairs is None:
        pairs = nearest_neighbor_pairs(
            recording.microphone_positions, config.k_neighbors,
        )
    else:
        selected = []
        for pair in pairs:
            try:
                first, second = pair
                first, second = integer_index(first), integer_index(second)
            except (TypeError, ValueError) as exc:
                raise ValueError("each pair must contain two integer microphone indices") from exc
            if not (0 <= first < recording.n_microphones and
                    0 <= second < recording.n_microphones):
                raise ValueError("microphone pair index is out of range")
            if first == second:
                raise ValueError("a microphone pair must contain two distinct indices")
            normalized = (min(first, second), max(first, second))
            if normalized not in selected:
                selected.append(normalized)
        if not selected:
            raise ValueError("pairs must contain at least one microphone pair")
        pairs = selected
    if roi is not None and roi.filtered_signals is not None:
        signals = np.asarray(roi.filtered_signals, dtype=float)
        general = GeneralConfig(
            lowcut_hz=roi.lowcut_hz,
            highcut_hz=roi.highcut_hz,
            speed_of_sound_m_s=roi.speed_of_sound_m_s,
            microphone_position_uncertainty_m=(
                roi.microphone_position_uncertainty_m
            ),
            environment_margin_m=roi.environment_margin_m,
        )
    else:
        general = GeneralConfig()
        signals = _filter_signals(recording.signals, recording.sample_rate, general)
    if general.speed_of_sound_m_s <= 0:
        raise ValueError("speed_of_sound_m_s must be positive")
    if general.microphone_position_uncertainty_m < 0:
        raise ValueError("microphone_position_uncertainty_m must be non-negative")
    if general.environment_margin_m < 0:
        raise ValueError("environment_margin_m must be non-negative")
    iu = np.asarray([pair[0] for pair in pairs], dtype=int)
    ju = np.asarray([pair[1] for pair in pairs], dtype=int)

    uncertainty = np.sqrt(2.0) * general.microphone_position_uncertainty_m
    distances = np.linalg.norm(
        recording.microphone_positions[iu] - recording.microphone_positions[ju],
        axis=1,
    )
    maximum_delays = np.ceil(
        distances / general.speed_of_sound_m_s * recording.sample_rate
    ).astype(int)
    maximum_delays += int(
        uncertainty / general.speed_of_sound_m_s * recording.sample_rate
    )

    frame_size = int(config.frame_size)
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    hop_size = frame_size // 2
    overlap_hz = config.band_width_hz / 2.0
    omega_list, frequency_bands = build_frequency_bands(
        fs=recording.sample_rate,
        nfft=frame_size,
        fmin=0.0 if general.lowcut_hz is None else general.lowcut_hz,
        fmax=(
            recording.sample_rate / 2.0
            if general.highcut_hz is None
            else general.highcut_hz
        ),
        omega_size_hz=config.band_width_hz,
        overlap_hz=overlap_hz,
    )
    if omega_list:
        full_width = max(len(omega) for omega in omega_list)
        complete = [
            (omega, limits)
            for omega, limits in zip(omega_list, frequency_bands)
            if len(omega) == full_width
        ]
        omega_list = [omega for omega, _ in complete]
        frequency_bands = [limits for _, limits in complete]
    if not omega_list:
        raise ValueError("no complete frequency bands remain")

    frame_starts = compute_frame_starts(
        recording.n_samples,
        frame_size,
        hop_size,
        force_last_frame=False,
    )
    frame_times = np.asarray(frame_starts, dtype=float) / recording.sample_rate

    projected_roi_mask = None
    if roi is not None:
        # The projection helper expects one native mask per microphone. Pair
        # masks remain public; microphone masks are retained only for this step.
        if roi.microphone_masks is None:
            raise ValueError("ROIResult lacks microphone masks needed for projection")
        projected_roi_mask = build_maad_roi_valid_mask_from_signals(
            signals_win=signals,
            fs=recording.sample_rate,
            Omega_list=omega_list,
            iu=iu,
            ju=ju,
            nfft_alg=frame_size,
            hop_alg=hop_size,
            nperseg_maad=ROI_NPERSEG,
            noverlap_maad=ROI_NOVERLAP,
            pair_mode="max",
            force_last_frame=False,
            roi_masks=np.asarray(roi.microphone_masks),
        )

    obs_tau, obs_band, obs_frame = extract_subband_tdoas_windowed_search(
        signals=signals,
        fs=recording.sample_rate,
        band_ranges_hz=frequency_bands,
        iu=iu,
        ju=ju,
        pair_max_tdoa=maximum_delays,
        frameSize=frame_size,
        hop=hop_size,
        B=None,
        valid_mask=(projected_roi_mask if config.use_roi_mask else None),
        force_last_frame=False,
        filter_mode=config.filter_mode,
        correlation_mode="norm_cc",
        workers=config.workers,
    )

    pair_results = {}
    for index, pair in enumerate(pairs):
        pair_results[pair] = PairLocalTDOAs(
            pair=pair,
            delays=np.asarray(obs_tau[index], dtype=np.float32),
            band_indices=np.asarray(obs_band[index], dtype=np.int16),
            frame_indices=np.asarray(obs_frame[index], dtype=np.int32),
            max_delay_samples=int(maximum_delays[index]),
        )
    return LocalTDOAResult(
        pairs=pair_results,
        frequency_bands_hz=tuple(
            (float(limits[0]), float(limits[1])) for limits in frequency_bands
        ),
        frame_times_seconds=frame_times,
        frame_size=frame_size,
        hop_size=hop_size,
        sample_rate=recording.sample_rate,
        algorithm_frequency_bins=tuple(np.asarray(x, dtype=int) for x in omega_list),
        projected_roi_mask=projected_roi_mask,
        filtered_signals=signals,
        speed_of_sound_m_s=general.speed_of_sound_m_s,
        microphone_position_uncertainty_m=(
            general.microphone_position_uncertainty_m
        ),
        environment_margin_m=general.environment_margin_m,
    )


def _filter_signals(
    signals: np.ndarray,
    sample_rate: float,
    config: GeneralConfig,
) -> np.ndarray:
    lowcut, highcut = config.lowcut_hz, config.highcut_hz
    if lowcut is not None and lowcut < 0:
        raise ValueError("lowcut_hz must be non-negative")
    if highcut is not None and highcut <= 0:
        raise ValueError("highcut_hz must be positive")
    if highcut is not None and highcut >= sample_rate / 2:
        raise ValueError("highcut_hz must be below Nyquist")
    if lowcut is not None and highcut is not None and lowcut >= highcut:
        raise ValueError("lowcut_hz must be below highcut_hz")
    output = np.asarray(signals, dtype=float).copy()
    nyquist = sample_rate / 2.0
    if lowcut is not None and lowcut > 0:
        b, a = butter(4, lowcut / nyquist, btype="highpass")
        output = filtfilt(b, a, output, axis=1)
    if highcut is not None:
        b, a = butter(4, highcut / nyquist, btype="lowpass")
        output = filtfilt(b, a, output, axis=1)
    return output
