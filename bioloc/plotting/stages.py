"""Original-style plots shared by notebooks and the standard report."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal

from bioloc._kernels import plotting as legacy
from bioloc._geometry import recording_environment_bounds


def plot_roi_stage(recording, roi, config, *, pair=None, show=True):
    """Plot a native ROI mask over the spectrogram used for ROI extraction."""
    import matplotlib.pyplot as plt

    legacy.set_plot_style(base_font=24)
    pairs = list(roi.masks_by_pair)
    if not pairs:
        return []
    pair = _select_pair(pairs, pair)
    mask = np.asarray(roi.masks_by_pair[pair], dtype=bool)
    signals = (
        roi.filtered_signals
        if roi.filtered_signals is not None
        else recording.signals
    )
    frequencies, times, spectrum = signal.spectrogram(
        signals[pair[0]],
        fs=recording.sample_rate,
        nperseg=1024,
        noverlap=512,
        mode="magnitude",
    )
    reference_db = float(20.0 * np.log10(np.max(spectrum) + 1e-12))
    decibels = 20.0 * np.log10(spectrum + 1e-12)
    figure, axis = plt.subplots(figsize=(20, 10), layout="constrained")
    axis.pcolormesh(
        times,
        frequencies,
        decibels,
        shading="auto",
        cmap="gray_r",
        vmin=reference_db - 96.0,
        vmax=reference_db,
    )
    overlay = np.ma.masked_where(~mask, mask)
    axis.imshow(
        overlay,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
        alpha=0.35,
        cmap="autumn",
        interpolation="nearest",
        zorder=2,
    )
    axis.set_ylim(
        0.0 if config.general.lowcut_hz is None else config.general.lowcut_hz,
        recording.sample_rate / 2.0
        if config.general.highcut_hz is None
        else config.general.highcut_hz,
    )
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Frequency [Hz]")
    axis.set_title(f"ROI mask over spectrogram - Pair {pair}")
    if show:
        plt.show()
    return [(figure, f"roi_mask_pair_{pair[0]}_{pair[1]}")]


def plot_local_tdoa_stage(recording, local_tdoas, config, *, pair=None, show=True):
    """Plot the original local-TDOA colour map."""
    plt, context = _prepare(recording, local_tdoas, config)
    figures = []
    pairs = list(local_tdoas.pairs)
    if pairs:
        requested_pair = _select_pair(pairs, pair)
        observations = list(local_tdoas.pairs.values())
        figure = legacy.plot_local_tdoa_colormap(
            [item.delays for item in observations],
            [item.band_indices for item in observations],
            [item.frame_indices for item in observations],
            local_tdoas.frequency_bands_hz,
            np.asarray([pair[0] for pair in pairs], dtype=int),
            np.asarray([pair[1] for pair in pairs], dtype=int),
            local_tdoas.hop_size,
            context["sample_rate"],
            max_abs_tdoa=max(item.max_delay_samples for item in observations),
            pair=requested_pair,
        )
        figures.append((figure, "local_tdoa_colormap"))
    if show:
        plt.show()
    return figures


def plot_histogram_stage(peaks, local_tdoas, config, *, show=True):
    """Plot peak histograms with their matching-pursuit Qn and Qw regions."""
    import matplotlib.pyplot as plt

    legacy.set_plot_style(base_font=24)
    figures = []
    for pair, pair_result in peaks.pairs.items():
        figure = plt.figure(figsize=(15, 10))
        plt.subplots_adjust(bottom=0.15, left=0.18)
        axis = plt.gca()
        axis.plot(pair_result.delay_bins, pair_result.histogram, label=f"Pair {pair}")
        maximum_delay = local_tdoas.pairs[pair].max_delay_samples
        narrow = max(2, int(2 * maximum_delay / config.peaks.narrow_width_divisor))
        wide = max(narrow, int(narrow * config.peaks.wide_to_narrow_ratio))
        for peak_index, peak in enumerate(pair_result.peaks):
            axis.axvspan(
                peak.delay_samples - wide / 2,
                peak.delay_samples + wide / 2,
                alpha=0.10,
                color="tab:orange",
                label="Qw" if peak_index == 0 else None,
            )
            axis.axvspan(
                peak.delay_samples - narrow / 2,
                peak.delay_samples + narrow / 2,
                alpha=0.22,
                color="tab:blue",
                label="Qn" if peak_index == 0 else None,
            )
            value = pair_result.histogram[
                int(np.argmin(np.abs(pair_result.delay_bins - peak.delay_samples)))
            ]
            axis.scatter(
                [peak.delay_samples],
                [value],
                color="black",
                edgecolors="black",
                s=90,
                zorder=5,
            )
        axis.set_title(f"TDOA histogram + matching-pursuit peaks - Pair {pair}")
        axis.set_xlabel("TDOA lag (samples)")
        axis.set_ylabel("Normalized count")
        axis.legend()
        figures.append((figure, f"tdoa_hist_pair_{pair[0]}_{pair[1]}"))
    if show:
        plt.show()
    return figures


def plot_association_stage(
    peaks,
    local_tdoas,
    association,
    config,
    *,
    show=True,
):
    """Plot only the time-frequency supports produced by association."""
    import matplotlib.pyplot as plt

    legacy.set_plot_style(base_font=24)
    colors = _cluster_colors(association)
    figures = []

    raw_peak_mapping = _raw_peak_mapping(peaks)
    if raw_peak_mapping:
        support_figures = legacy.plot_peak_timefreq_support_by_pair(
            raw_peak_mapping,
            n_bands=len(local_tdoas.frequency_bands_hz),
            n_frames=len(local_tdoas.frame_times_seconds),
            hop=local_tdoas.hop_size,
            fs=local_tdoas.sample_rate,
            frames_per_bin=1,
            sources=association.raw_sources,
            cid2color=colors,
        )
        for pair, figure in support_figures.items():
            figures.append((figure, f"peak_tf_support_pair_{pair[0]}_{pair[1]}"))
    if show:
        plt.show()
    return figures


def plot_reconstruction_stage(
    recording,
    local_tdoas,
    association,
    reconstruction,
    config,
    *,
    show=True,
):
    """Plot original soft-mask overlays and reconstructed spectrograms."""
    plt, context = _prepare(recording, local_tdoas, config)
    colors = _cluster_colors(association)
    figures = []
    for source in reconstruction.sources:
        if source.source_id >= len(reconstruction.display_masks):
            continue
        mask = reconstruction.display_masks[source.source_id]
        figure, _, _, _, _ = legacy.plot_highres_spectrogram_with_projected_clusters(
            signals=context["signals"],
            fs=context["sample_rate"],
            mic_idx=source.reference_microphone,
            cid2color={0: colors.get(source.source_id, f"C{source.source_id % 10}")},
            projected_maps=[mask],
            t_start=0.0,
            t_end=recording.n_samples / context["sample_rate"],
            nperseg=1024,
            noverlap=512,
            nfft=1024,
            alpha=0.44,
            cmap="gray_r",
            title=(
                f"Spectrogram with soft mask source {source.source_id} "
                f"(mic {source.reference_microphone})"
            ),
            fmin=context["frequency_min"],
            fmax=context["frequency_max"],
            ref_db=context["reference_db"],
        )
        figures.append((figure, f"spectrogram_soft_mask_source_{source.source_id}"))

    if reconstruction.sources:
        reconstructed = np.stack([source.signal for source in reconstruction.sources])
        for source in reconstruction.sources:
            figure, _ = legacy.plot_reference_spectrogram(
                signals=reconstructed,
                fs=reconstruction.sample_rate,
                mic_idx=source.source_id,
                t_start=0.0,
                t_end=source.signal.size / reconstruction.sample_rate,
                nperseg=1024,
                noverlap=512,
                nfft=1024,
                title=f"Reconstructed source {source.source_id}",
                fmin=context["frequency_min"],
                fmax=context["frequency_max"],
                ref_db=context["reference_db"],
            )
            figures.append(
                (figure, f"source_{source.source_id}_reconstructed_spectrogram")
            )
    if show:
        plt.show()
    return figures


def plot_localization_stage(
    recording,
    association,
    localization,
    config,
    *,
    show=True,
):
    """Plot associated hyperbolas and final source estimates."""
    import matplotlib.pyplot as plt

    legacy.set_plot_style(base_font=24)
    figures = []
    if association.sources:
        colors = _cluster_colors(association)
        figure, axis = plt.subplots(figsize=(20, 20))
        lower, upper = recording_environment_bounds(
            recording,
            margin_m=config.general.environment_margin_m,
        )
        environment_size = float(upper[0] - lower[0])
        plot_microphones = recording.microphone_positions - lower
        plot_positions = localization.positions - lower
        legacy.plot_sources_hyperbolas(
            association.raw_sources,
            plot_microphones,
            recording.sample_rate,
            environment_size,
            ax=axis,
            cid2color=colors,
            alpha=0.5,
            linewidth=10.0,
            c=association.speed_of_sound_m_s,
            clip_impossible=True,
        )
        _add_satellite_background_if_available(
            axis,
            recording,
            coordinate_shift=lower,
            environment_size=environment_size,
        )
        legacy.plot_truth_vs_estimates(
            micPos=plot_microphones,
            estPos=plot_positions,
            srcPos=None,
            roomDims=environment_size,
            ax=axis,
            cid2color=colors,
        )
        figures.append((figure, "truth_vs_estimates"))
    if show:
        plt.show()
    return figures


def _prepare(recording, local_tdoas, config):
    import matplotlib.pyplot as plt

    legacy.set_plot_style(base_font=24)
    plt.rcParams["figure.max_open_warning"] = 0
    signals = (
        local_tdoas.filtered_signals
        if local_tdoas.filtered_signals is not None
        else recording.signals
    )
    sample_rate = recording.sample_rate
    reference_mic = min(3, recording.n_microphones - 1)
    lowcut = config.general.lowcut_hz
    highcut = config.general.highcut_hz
    return plt, {
        "signals": signals,
        "sample_rate": sample_rate,
        "reference_mic": reference_mic,
        "reference_db": _global_reference_db(signals[reference_mic], sample_rate),
        "frequency_min": 0.0 if lowcut is None else lowcut,
        "frequency_max": sample_rate / 2.0 if highcut is None else highcut,
    }


def _cluster_colors(association):
    return legacy.build_cluster_color_map(
        association.raw_sources,
        cmap_name=("tab10" if len(association.sources) <= 10 else "tab20"),
    )


def _raw_peak_mapping(peaks):
    from bioloc.association import _raw_peak_mapping as convert

    return convert(peaks)


def _select_pair(available_pairs, requested_pair):
    if not available_pairs:
        raise ValueError("No microphone pairs are available to plot")
    if requested_pair is None:
        return available_pairs[0]
    requested_pair = tuple(map(int, requested_pair))
    if requested_pair not in available_pairs:
        raise ValueError(
            f"Microphone pair {requested_pair} is unavailable; "
            f"choose from {available_pairs}"
        )
    return requested_pair


def _global_reference_db(values, sample_rate):
    _, _, spectrum = signal.stft(
        np.asarray(values, dtype=float),
        fs=sample_rate,
        nperseg=1024,
        noverlap=512,
        nfft=1024,
        boundary=None,
        padded=False,
    )
    return float(20.0 * np.log10(np.max(np.abs(spectrum)) + 1e-12))


def _add_satellite_background_if_available(
    axis,
    recording,
    *,
    coordinate_shift=None,
    environment_size=None,
):
    metadata = recording.metadata
    required = ("gps_groups_for_map", "local_groups_for_map", "map_info")
    if not all(name in metadata for name in required):
        return
    try:
        local_groups = np.asarray(metadata["local_groups_for_map"], dtype=float)
        map_info = dict(metadata["map_info"])
        if coordinate_shift is not None:
            local_groups = local_groups - np.asarray(coordinate_shift, dtype=float)
        if environment_size is not None:
            map_info["plot_min"] = np.asarray([0.0, 0.0])
            map_info["plot_max"] = np.asarray(
                [environment_size, environment_size], dtype=float
            )
            map_info["environment_size"] = float(environment_size)
        legacy.add_satellite_background_to_local_axes(
            axis,
            gps_groups=metadata["gps_groups_for_map"],
            local_groups_centered=local_groups,
            map_info=map_info,
            zoom=18,
            alpha=0.4,
        )
    except Exception as error:
        warnings.warn(
            f"Satellite background unavailable; plotting local coordinates only: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


__all__ = [
    "plot_association_stage",
    "plot_histogram_stage",
    "plot_local_tdoa_stage",
    "plot_localization_stage",
    "plot_reconstruction_stage",
    "plot_roi_stage",
]
