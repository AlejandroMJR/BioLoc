"""Plots for public pipeline result objects."""

from __future__ import annotations

import numpy as np

from .standard import make_standard_plots


def plot_roi(roi, pair, *, ax=None):
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    mask = roi.for_pair(*pair)
    image = ax.imshow(mask, origin="lower", aspect="auto", interpolation="nearest")
    ax.set(title=f"ROI mask, pair {tuple(pair)}", xlabel="Time frame", ylabel="Frequency bin")
    return ax.figure, ax, image


def plot_local_tdoas(local_tdoas, pair, *, ax=None):
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    values = local_tdoas.for_pair(*pair)
    times = local_tdoas.frame_times_seconds[values.frame_indices]
    centers = np.asarray(
        [(low + high) / 2.0 for low, high in local_tdoas.frequency_bands_hz]
    )
    scatter = ax.scatter(times, centers[values.band_indices], c=values.delays, cmap="coolwarm")
    ax.set(title=f"Local TDOAs, pair {tuple(pair)}", xlabel="Time (s)", ylabel="Frequency (Hz)")
    return ax.figure, ax, scatter


def plot_pair_histogram(peaks, pair, *, ax=None):
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    result = peaks.for_pair(*pair)
    ax.plot(result.delay_bins, result.histogram)
    for peak in result.peaks:
        ax.axvline(peak.delay_samples, color="tab:red", alpha=0.8)
    ax.set(title=f"TDOA peaks, pair {tuple(pair)}", xlabel="Delay (samples)", ylabel="Normalized support")
    return ax.figure, ax


def plot_feature_support(association, pair, peak_index=0, *, ax=None):
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    feature = association.features_for_pair(*pair)[peak_index]
    ax.plot(feature.vector)
    ax.set(title=f"Association feature, pair {tuple(pair)}, peak {peak_index}", xlabel="Feature index", ylabel="Support")
    return ax.figure, ax


def plot_reconstruction_mask(reconstruction, source_id=0, *, ax=None):
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    source = reconstruction.sources[source_id]
    image = ax.imshow(source.reconstruction_mask, origin="lower", aspect="auto")
    ax.set(title=f"Final reconstruction mask, source {source_id}", xlabel="Time frame", ylabel="Frequency band")
    return ax.figure, ax, image


def plot_reconstructed_signal(reconstruction, source_id=0, *, ax=None):
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    source = reconstruction.sources[source_id]
    times = np.arange(source.signal.size) / reconstruction.sample_rate
    ax.plot(times, source.signal)
    ax.set(title=f"Reconstructed source {source_id}", xlabel="Time (s)", ylabel="Amplitude")
    return ax.figure, ax


def plot_localizations(recording, localization, *, ax=None):
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots()
    positions = recording.microphone_positions
    ax.scatter(positions[:, 0], positions[:, 1], marker="^", label="Microphones")
    for source in localization.sources:
        if source.position is not None:
            ax.scatter(source.position[0], source.position[1], marker="x", s=80)
            ax.annotate(f"S{source.source_id}", source.position[:2])
    ax.set(title="Estimated source locations", xlabel="x (m)", ylabel="y (m)")
    ax.axis("equal")
    return ax.figure, ax


def plot_reference_spectrogram(recording, microphone=0, **kwargs):
    """Plot the existing high-resolution reference spectrogram."""
    legacy = _legacy_plotting()
    return legacy.plot_reference_spectrogram(
        signals=recording.signals,
        fs=recording.sample_rate,
        mic_idx=int(microphone),
        **kwargs,
    )


def plot_associated_hyperbolas(
    recording,
    association,
    environment_size,
    *,
    speed_of_sound_m_s=333.0,
    ax=None,
    **kwargs,
):
    """Plot grouped TDOA hyperbolas using the established plotting routine."""
    plt = _pyplot()
    legacy = _legacy_plotting()
    if ax is None:
        _, ax = plt.subplots()
    legacy.plot_sources_hyperbolas(
        association.raw_sources,
        recording.microphone_positions,
        recording.sample_rate,
        environment_size,
        ax=ax,
        c=speed_of_sound_m_s,
        **kwargs,
    )
    return ax.figure, ax


def plot_peak_timefrequency_support(
    peaks,
    local_tdoas,
    association,
    *,
    frames_per_bin=1,
    colors=None,
):
    """Return the established per-pair peak-support figures."""
    from bioloc.association import _raw_peak_mapping

    legacy = _legacy_plotting()
    if colors is None:
        colors = legacy.build_cluster_color_map(association.raw_sources)
    return legacy.plot_peak_timefreq_support_by_pair(
        _raw_peak_mapping(peaks),
        n_bands=len(local_tdoas.frequency_bands_hz),
        n_frames=len(local_tdoas.frame_times_seconds),
        hop=local_tdoas.hop_size,
        fs=local_tdoas.sample_rate,
        frames_per_bin=frames_per_bin,
        sources=association.raw_sources,
        cid2color=colors,
    )


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "The package's plotting dependencies are missing. Reinstall "
            "`bioloc` to restore them."
        ) from error
    return plt


def _legacy_plotting():
    try:
        from bioloc._kernels import plotting as utils
    except ImportError as error:
        raise ImportError(
            "The package's plotting dependencies are missing. Reinstall "
            "`bioloc` to restore them."
        ) from error
    return utils


__all__ = [
    "plot_feature_support",
    "plot_associated_hyperbolas",
    "plot_local_tdoas",
    "plot_localizations",
    "plot_peak_timefrequency_support",
    "plot_pair_histogram",
    "plot_reconstructed_signal",
    "plot_reconstruction_mask",
    "plot_reference_spectrogram",
    "plot_roi",
    "make_standard_plots",
]
