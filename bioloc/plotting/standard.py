"""Fixed, original-style standard plot report for the public pipeline."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from scipy import signal


def make_standard_plots(
    recording,
    result,
    config,
    *,
    show: bool,
    save: bool,
):
    """Create the fixed set of scientifically useful original-style figures."""
    import matplotlib.pyplot as plt

    from bioloc._kernels import plotting as legacy

    legacy.set_plot_style(base_font=24)
    plt.rcParams["figure.max_open_warning"] = 0
    output_directory = Path("Figures") / "BioLoc"
    if save:
        output_directory.mkdir(parents=True, exist_ok=True)

    signals = (
        result.local_tdoas.filtered_signals
        if result.local_tdoas.filtered_signals is not None
        else recording.signals
    )
    sample_rate = recording.sample_rate
    reference_mic = min(3, recording.n_microphones - 1)
    global_reference_db = _global_reference_db(
        signals[reference_mic],
        sample_rate,
    )
    lowcut = config.local_tdoa.lowcut_hz
    highcut = config.local_tdoa.highcut_hz
    frequency_min = 0.0 if lowcut is None else lowcut
    frequency_max = sample_rate / 2.0 if highcut is None else highcut

    figures = []

    # ROI validity on the same reference spectrogram used by the old pipeline.
    valid_mask = result.local_tdoas.projected_roi_mask
    if valid_mask is not None:
        figure, _ = legacy.plot_valid_mask_on_reference_spectrogram(
            signal_ref=signals[reference_mic],
            fs=sample_rate,
            valid_mask=np.any(valid_mask, axis=0),
            band_ranges_hz=result.local_tdoas.frequency_bands_hz,
            mask_times=result.local_tdoas.frame_times_seconds,
            nperseg=1024,
            noverlap=512,
            nfft=1024,
            fmin=frequency_min,
            fmax=frequency_max,
            ref_db=global_reference_db,
            db_floor=80.0,
            cmap="gray_r",
            fixed_clim=True,
        )
        figures.append((figure, "valid_mask_on_reference_spectrogram"))

    # Original local-TDOA colour map, using its established diagnostic pair.
    pairs = list(result.local_tdoas.pairs)
    if pairs:
        requested_pair = (1, 3)
        if requested_pair not in result.local_tdoas.pairs:
            requested_pair = pairs[0]
        observations = list(result.local_tdoas.pairs.values())
        figure = legacy.plot_local_tdoa_colormap(
            [item.delays for item in observations],
            [item.band_indices for item in observations],
            [item.frame_indices for item in observations],
            result.local_tdoas.frequency_bands_hz,
            np.asarray([pair[0] for pair in pairs], dtype=int),
            np.asarray([pair[1] for pair in pairs], dtype=int),
            result.local_tdoas.hop_size,
            sample_rate,
            max_abs_tdoa=max(item.max_delay_samples for item in observations),
            pair=requested_pair,
        )
        figures.append((figure, "local_tdoa_colormap"))

    raw_peak_mapping = _raw_peak_mapping(result.peaks)
    colors = legacy.build_cluster_color_map(
        result.association.raw_sources,
        cmap_name=("tab10" if len(result.association.sources) <= 10 else "tab20"),
    )

    # Original associated histogram presentation for every available pair.
    peak_to_source = {}
    for source_id, source in enumerate(result.association.raw_sources):
        for member in source.get("members", []):
            peak_to_source[(tuple(member["pair"]), int(member["peak_idx"]))] = source_id
    for pair, pair_result in result.peaks.pairs.items():
        figure = plt.figure(figsize=(15, 10))
        plt.subplots_adjust(bottom=0.15, left=0.18)
        axis = plt.gca()
        axis.plot(pair_result.delay_bins, pair_result.histogram, label=f"Pair {pair}")
        maximum_delay = result.local_tdoas.pairs[pair].max_delay_samples
        narrow = max(2, int(2 * maximum_delay / 40))
        wide = max(narrow, narrow * 3)
        for peak_index, peak in enumerate(pair_result.peaks):
            source_id = peak_to_source.get((pair, peak_index))
            color = "0.5"
            axis.axvspan(
                peak.delay_samples - wide / 2,
                peak.delay_samples + wide / 2,
                alpha=0.10,
                color=color,
            )
            axis.axvspan(
                peak.delay_samples - narrow / 2,
                peak.delay_samples + narrow / 2,
                alpha=0.22,
                color=color,
            )
            value = pair_result.histogram[
                int(np.argmin(np.abs(pair_result.delay_bins - peak.delay_samples)))
            ]
            axis.scatter(
                [peak.delay_samples],
                [value],
                color=color,
                edgecolors="black",
                s=90,
                zorder=5,
            )
            axis.text(
                peak.delay_samples,
                1.03,
                f"S{source_id}" if source_id is not None else "U",
                color=color,
                ha="center",
                va="bottom",
            )
        axis.set_title(f"TDOA histogram + associated MP peaks - Pair {pair}")
        axis.set_xlabel("TDOA lag (samples)")
        axis.set_ylabel("Normalized count")
        axis.legend()
        figures.append((figure, f"tdoa_hist_pair_{pair[0]}_{pair[1]}_associated"))

    # Original per-pair peak time-frequency support plots.
    if raw_peak_mapping:
        support_figures = legacy.plot_peak_timefreq_support_by_pair(
            raw_peak_mapping,
            n_bands=len(result.local_tdoas.frequency_bands_hz),
            n_frames=len(result.local_tdoas.frame_times_seconds),
            hop=result.local_tdoas.hop_size,
            fs=sample_rate,
            frames_per_bin=config.association.frames_per_bin,
            sources=result.association.raw_sources,
            cid2color=colors,
        )
        for pair, figure in support_figures.items():
            figures.append((figure, f"peak_tf_support_pair_{pair[0]}_{pair[1]}"))

    # Final masks on the microphone timeline actually used for each source.
    for source in result.reconstruction.sources:
        if source.source_id >= len(result.reconstruction.display_masks):
            continue
        mask = result.reconstruction.display_masks[source.source_id]
        figure, _, _, _, _ = legacy.plot_highres_spectrogram_with_projected_clusters(
            signals=signals,
            fs=sample_rate,
            mic_idx=source.reference_microphone,
            cid2color={0: colors.get(source.source_id, f"C{source.source_id % 10}")},
            projected_maps=[mask],
            t_start=0.0,
            t_end=recording.n_samples / sample_rate,
            nperseg=1024,
            noverlap=512,
            nfft=1024,
            alpha=0.44,
            cmap="gray_r",
            title=(
                f"Spectrogram with soft mask source {source.source_id} "
                f"(mic {source.reference_microphone})"
            ),
            fmin=frequency_min,
            fmax=frequency_max,
            ref_db=global_reference_db,
        )
        figures.append((figure, f"spectrogram_soft_mask_source_{source.source_id}"))

    # Spectrograms of the actual signals returned by the package.
    if result.reconstruction.sources:
        reconstructed = np.stack(
            [source.signal for source in result.reconstruction.sources], axis=0
        )
        for source in result.reconstruction.sources:
            figure, _ = legacy.plot_reference_spectrogram(
                signals=reconstructed,
                fs=result.reconstruction.sample_rate,
                mic_idx=source.source_id,
                t_start=0.0,
                t_end=source.signal.size / result.reconstruction.sample_rate,
                nperseg=1024,
                noverlap=512,
                nfft=1024,
                title=f"Reconstructed source {source.source_id}",
                fmin=frequency_min,
                fmax=frequency_max,
                ref_db=global_reference_db,
            )
            figures.append((figure, f"source_{source.source_id}_reconstructed_spectrogram"))

    # Grouped hyperbolas, optional satellite context, and final estimates.
    if result.association.sources:
        figure, axis = plt.subplots(figsize=(20, 20))
        environment_size = _environment_size(recording, config)
        legacy.plot_sources_hyperbolas(
            result.association.raw_sources,
            recording.microphone_positions,
            sample_rate,
            environment_size,
            ax=axis,
            cid2color=colors,
            alpha=0.5,
            linewidth=10.0,
            c=config.local_tdoa.speed_of_sound_m_s,
            clip_impossible=True,
        )
        _add_satellite_background_if_available(legacy, axis, recording)
        legacy.plot_truth_vs_estimates(
            micPos=recording.microphone_positions,
            estPos=result.localization.positions,
            srcPos=None,
            roomDims=environment_size,
            ax=axis,
            cid2color=colors,
        )
        figures.append((figure, "truth_vs_estimates"))

    for figure, filename in figures:
        if save:
            legacy.savefig(
                figure,
                str(output_directory / filename),
                save_pdf=False,
                save_png=True,
                close=not show,
            )
    if show:
        plt.show()
    elif not save:
        for figure, _ in figures:
            plt.close(figure)
    return [figure for figure, _ in figures]


def _raw_peak_mapping(peaks):
    from bioloc.association import _raw_peak_mapping as convert

    return convert(peaks)


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


def _environment_size(recording, config):
    map_info = recording.metadata.get("map_info")
    if map_info is not None and "environment_size" in map_info:
        return float(map_info["environment_size"])
    bounds = config.localization.environment_bounds
    if bounds is not None:
        return float(np.max(np.asarray(bounds[1]) - np.asarray(bounds[0])))
    positions = recording.microphone_positions
    span = np.max(positions, axis=0) - np.min(positions, axis=0)
    return float(max(np.max(span), 1.0))


def _add_satellite_background_if_available(legacy, axis, recording):
    metadata = recording.metadata
    required = ("gps_groups_for_map", "local_groups_for_map", "map_info")
    if not all(name in metadata for name in required):
        return
    try:
        legacy.add_satellite_background_to_local_axes(
            axis,
            gps_groups=metadata["gps_groups_for_map"],
            local_groups_centered=metadata["local_groups_for_map"],
            map_info=metadata["map_info"],
            zoom=18,
            alpha=0.4,
        )
    except Exception as error:
        warnings.warn(
            f"Satellite background unavailable; plotting local coordinates only: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
