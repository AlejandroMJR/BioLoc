import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .association import peak_band_vector, peak_timefreq_vector
import xml.etree.ElementTree as ET
import re
import librosa
from pathlib import Path
from datetime import datetime, timedelta
import contextily as cx
import xyzservices.providers as xyz
from pyproj import Transformer
from scipy.signal import spectrogram

# ============================================================
# File / folder utilities
# ============================================================

def clear_folder(folder_path, pattern="*"):
    """
    Delete files inside `folder_path` (non-recursive).

    Parameters
    ----------
    folder_path : str
        Path to the directory to clear.
    pattern : str
        Glob pattern for which files to delete (default: '*').
        Examples: '*.png', '*.pdf'
    """
    if not os.path.isdir(folder_path):
        return

    files = glob.glob(os.path.join(folder_path, pattern))
    for file in files:
        if not os.path.isfile(file):
            continue
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting {file}: {e}")


def ensure_dir(folder_path):
    """Create folder if it does not exist."""
    os.makedirs(folder_path, exist_ok=True)


# ============================================================
# Feature utilities
# ============================================================
def build_cluster_color_map(sources, cmap_name="tab10"):
    """
    Returns cid2color: dict {cluster_id: rgba}
    """
    if not sources:
        return {}

    cmap = plt.get_cmap(cmap_name)
    n = len(sources)
    return {cid: cmap(cid % cmap.N) for cid in range(n)}


def build_peak_cluster_map(sources):
    """
    Build lookup (pair, peak_idx) -> cluster_id for coloring / post-processing.

    Parameters
    ----------
    sources : list of dict
        Output of associate_tdoa_peaks_by_band_cosine_constrained(...).
        Each dict has "members", and each member contains:
          - "pair": (i,j)
          - "peak_idx": int

    Returns
    -------
    peak2cid : dict
        Maps ((i,j), peak_idx) -> cluster_id
    """
    peak2cid = {}
    if not sources:
        return peak2cid

    for cid, src in enumerate(sources):
        for m in src.get("members", []):
            peak2cid[(tuple(m["pair"]), int(m["peak_idx"]))] = cid
    return peak2cid

def theoretical_tdoas(src_positions, mic_positions, c=343.0, fs=None):
    src_positions = np.asarray(src_positions)
    mic_positions = np.asarray(mic_positions)
    src_positions = src_positions.reshape(-1, src_positions.shape[-1])
    mic_positions = mic_positions.reshape(-1, mic_positions.shape[-1])

    dists = np.linalg.norm(src_positions[:, None, :] - mic_positions[None, :, :], axis=-1)
    tof = dists / c
    tdoa_seconds = tof[:, :, None] - tof[:, None, :]

    if fs is not None:
        return tdoa_seconds, tdoa_seconds * fs
    return tdoa_seconds, None


# ============================================================
# Geometry plots
# ============================================================

def set_plot_style(base_font=24):
    """Set consistent matplotlib style for plots."""
    plt.rcParams.update({
        "font.size": base_font,
        "axes.titlesize": base_font,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 2,
        "ytick.labelsize": base_font - 2,
        "legend.fontsize": base_font - 2,
        "axes.linewidth": 1.4,
    })


def savefig(fig, path_no_ext, *, save_pdf=True, save_png=False, dpi=300, close=True):
    """
    Save a matplotlib figure with consistent settings.
    """
    import os
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)

    if save_pdf:
        fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", pad_inches=0.02)
    if save_png:
        fig.savefig(path_no_ext + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    if close:
        plt.close(fig)


def plot_local_tdoa_colormap(
    obs_tau,
    obs_band,
    obs_frame,
    band_ranges_hz,
    iu,
    ju,
    hop,
    fs,
    max_abs_tdoa=None,
    cmap="hsv",
    pair=(0, 1),
):
    """Plot local TDOA estimates for one microphone pair.

    Parameters
    ----------
    pair : tuple of int, default=(0, 1)
        Microphone pair to show. The corresponding entry is located from
        ``iu`` and ``ju`` rather than assuming a particular pair ordering.
    """
    pair = tuple(map(int, pair))
    pair_indices = np.flatnonzero(
        (np.asarray(iu, dtype=int) == pair[0])
        & (np.asarray(ju, dtype=int) == pair[1])
    )
    if pair_indices.size == 0:
        available = list(zip(np.asarray(iu, dtype=int), np.asarray(ju, dtype=int)))
        raise ValueError(f"Microphone pair {pair} is unavailable; choose from {available}")
    p = int(pair_indices[0])

    # Match the large-format diagnostic plots used elsewhere in the paper.
    # Constrained layout keeps the colorbar label and axis labels from being
    # clipped when a large global font size is active.
    fig, ax = plt.subplots(figsize=(20, 10), layout="constrained")

    values = np.asarray(obs_tau[p], dtype=float)
    finite_taus = values[np.isfinite(values)]
    if max_abs_tdoa is None:
        max_abs_tdoa = np.max(np.abs(finite_taus)) if finite_taus.size else 1.0
    max_abs_tdoa = max(float(max_abs_tdoa), 1.0)
    norm = TwoSlopeNorm(vmin=-max_abs_tdoa, vcenter=0.0, vmax=max_abs_tdoa)

    band_centers_hz = np.asarray(
        [(float(low) + float(high)) / 2.0 for low, high in band_ranges_hz],
        dtype=float,
    )
    taus = np.asarray(obs_tau[p], dtype=float)
    bands = np.asarray(obs_band[p], dtype=int)
    frames = np.asarray(obs_frame[p], dtype=float)
    valid = (
        np.isfinite(taus)
        & np.isfinite(frames)
        & (bands >= 0)
        & (bands < len(band_centers_hz))
    )

    if np.any(valid):
        # Render complete time--subband cells rather than fixed-size scatter
        # markers. Marker sizes are expressed in screen points and therefore
        # leave frequency-dependent white gaps when the figure is resized.
        frame_ids = frames[valid].astype(int)
        n_frames = int(np.max(frame_ids)) + 1
        tdoa_grid = np.full(
            (len(band_centers_hz), n_frames),
            np.nan,
            dtype=float,
        )
        tdoa_grid[bands[valid], frame_ids] = taus[valid]

        time_step = float(hop) / float(fs)
        time_edges = (
            np.arange(n_frames + 1, dtype=float) - 0.5
        ) * time_step

        if len(band_centers_hz) == 1:
            low, high = map(float, band_ranges_hz[0])
            frequency_edges = np.array([low, high], dtype=float)
        else:
            frequency_edges = np.empty(len(band_centers_hz) + 1, dtype=float)
            frequency_edges[1:-1] = 0.5 * (
                band_centers_hz[:-1] + band_centers_hz[1:]
            )
            frequency_edges[0] = max(
                0.0,
                band_centers_hz[0]
                - 0.5 * (band_centers_hz[1] - band_centers_hz[0]),
            )
            frequency_edges[-1] = (
                band_centers_hz[-1]
                + 0.5 * (band_centers_hz[-1] - band_centers_hz[-2])
            )

        scatter = ax.pcolormesh(
            time_edges,
            frequency_edges,
            tdoa_grid,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="none",
            linewidth=0,
            antialiased=False,
            rasterized=True,
        )
        ax.set_xlim(time_edges[0], time_edges[-1])
        ax.set_ylim(frequency_edges[0], frequency_edges[-1])
    else:
        ax.text(
            0.5,
            0.5,
            "No local TDOA estimates",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        scatter = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    ax.set_title(f"Microphone pair ({pair[0]}, {pair[1]})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sub-band center frequency (Hz)")
    ax.grid(alpha=0.2)
    fig.colorbar(
        scatter,
        ax=ax,
        label="Local TDOA (samples)",
        shrink=0.88,
        pad=0.02,
    )
    return fig


def plot_room(micPos, srcPos, roomDims, ax=None,
              mic_label="Microphones", src_label="Sources",
              mic_color="green", src_color="black",
              mic_marker="o", src_marker="x",
              mic_size=300, src_size=300,
              show_legend=True,
              **scatter_kwargs):
    """
    Plot microphones and sources in a 2D room.


    Parameters
    ----------
    micPos : (M,2) array
        Microphone positions.
    srcPos : (S,2) array
        Source positions (can be empty with shape (0,2)).
    roomDims : float or (W,H)
        Room size. If float -> square [0,roomDims]x[0,roomDims].
    ax : matplotlib Axes or None
        If None, creates a new figure/axes.
    scatter_kwargs : dict
        Extra kwargs passed to ax.scatter.
    """
    micPos = np.asarray(micPos, dtype=float)
    srcPos = np.asarray(srcPos, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Room bounds
    if np.isscalar(roomDims):
        xlim = (0.0, float(roomDims))
        ylim = (0.0, float(roomDims))
    else:
        xlim = (0.0, float(roomDims[0]))
        ylim = (0.0, float(roomDims[1]))

    # Microphones
    ax.scatter(
        micPos[:, 0], micPos[:, 1],
        s=mic_size, marker=mic_marker, c=mic_color,
        linewidths=4,
        label=mic_label,
        **scatter_kwargs
    )

    # Sources (optional)
    if srcPos.size > 0:
        ax.scatter(
            srcPos[:, 0], srcPos[:, 1],
            s=src_size, marker=src_marker, c=src_color,
            linewidths=1.5,
            label=src_label,
            **scatter_kwargs
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best")

    return ax


def plot_tdoa_hyperbola(m0, m1, tdoa,
                        fs=None, c=343.0,
                        xlim=(0, 1), ylim=(0, 1),
                        n_points=400, ax=None,
                        color=None, linestyle="--", linewidths=1.0, alpha=1.0,
                        degenerate_tol_m=0.05,
                        clip_to_physical=False,
                        return_status=False,
                        **contour_kwargs):
    """
    Plot the 2D TDOA hyperbola for a microphone pair as a zero-level contour.

    Hyperbola definition:
        ||p - m0|| - ||p - m1|| = c * tau

    Parameters
    ----------
    m0, m1 : (2,) array-like
        Microphone positions [x,y] in meters.
    tdoa : float
        If fs is None -> seconds.
        If fs is given -> samples (converted to seconds).
    fs : float or None
        Sampling frequency.
    c : float
        Speed of sound (m/s).
    xlim, ylim : (min,max)
        Plot bounds in meters.
    n_points : int
        Grid resolution used for contouring.
    ax : matplotlib Axes or None
        Axis to plot on.
    color : any matplotlib color or None
        Can be a string ("C0") OR an RGBA array from a colormap.
    linewidths, alpha : float
        Contour style.
    degenerate_tol_m : float
        Tolerance in meters for near-maximum TDOAs. When
        abs(c * tau) is very close to the microphone spacing, the hyperbola
        degenerates into a ray. In that case matplotlib contour may not draw
        anything because the zero level only touches the grid. This function
        explicitly draws the degenerate ray instead.
    contour_kwargs : dict
        Passed to ax.contour.
    """
    m0 = np.asarray(m0, dtype=float)
    m1 = np.asarray(m1, dtype=float)

    # Convert to seconds if needed
    tau = (float(tdoa) / float(fs)) if fs is not None else float(tdoa)
    delta_d = c * tau  # distance difference in meters
    mic_dist = float(np.linalg.norm(m1 - m0))
    was_clipped = False
    if clip_to_physical and mic_dist > 0.0 and abs(delta_d) > mic_dist:
        delta_d = np.sign(delta_d) * mic_dist
        was_clipped = True

    xs = np.linspace(xlim[0], xlim[1], n_points)
    ys = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y], axis=-1)

    d0 = np.linalg.norm(P - m0, axis=-1)
    d1 = np.linalg.norm(P - m1, axis=-1)
    F = d0 - d1 - delta_d

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    default_kwargs = dict(linestyles=linestyle, linewidths=linewidths, alpha=alpha)
    if color is not None:
        default_kwargs["colors"] = [color]
    default_kwargs.update(contour_kwargs)

    # If |delta_d| is very close to the microphone spacing, the hyperbola is
    # degenerate: it becomes a ray along the microphone-pair axis. In this case
    # F may only touch zero without changing sign, so ax.contour can draw
    # nothing even though the TDOA is physically valid.
    is_near_degenerate = (
        mic_dist > 0.0
        and abs(abs(delta_d) - mic_dist) <= float(degenerate_tol_m)
    )

    crosses_grid = (float(np.nanmin(F)) <= 0.0 <= float(np.nanmax(F)))

    if is_near_degenerate:
        u = (m1 - m0) / mic_dist

        if delta_d >= 0:
            # d0 - d1 = +D: ray starts at m1 and continues away from m0.
            ray_start = m1
            ray_dir = u
        else:
            # d0 - d1 = -D: ray starts at m0 and continues away from m1.
            ray_start = m0
            ray_dir = -u

        candidates = []
        eps = 1e-12

        if abs(ray_dir[0]) > eps:
            for xb in (float(xlim[0]), float(xlim[1])):
                s = (xb - ray_start[0]) / ray_dir[0]
                yb = ray_start[1] + s * ray_dir[1]
                if s >= 0 and float(ylim[0]) <= yb <= float(ylim[1]):
                    candidates.append((s, np.array([xb, yb], dtype=float)))

        if abs(ray_dir[1]) > eps:
            for yb in (float(ylim[0]), float(ylim[1])):
                s = (yb - ray_start[1]) / ray_dir[1]
                xb = ray_start[0] + s * ray_dir[0]
                if s >= 0 and float(xlim[0]) <= xb <= float(xlim[1]):
                    candidates.append((s, np.array([xb, yb], dtype=float)))

        if candidates:
            _, ray_end = max(candidates, key=lambda item: item[0])
            line_kwargs = dict(linewidth=linewidths, alpha=alpha, linestyle=linestyle)
            if color is not None:
                line_kwargs["color"] = color
            ax.plot([ray_start[0], ray_end[0]], [ray_start[1], ray_end[1]], **line_kwargs)
            status = "drawn_clipped" if was_clipped else "drawn_degenerate"
        else:
            status = "outside_plot"
        return (ax, status) if return_status else ax

    if not crosses_grid:
        status = (
            "physically_impossible"
            if abs(delta_d) > mic_dist + float(degenerate_tol_m)
            else "outside_plot"
        )
        return (ax, status) if return_status else ax

    ax.contour(X, Y, F, levels=[0.0], **default_kwargs)
    status = "drawn"
    return (ax, status) if return_status else ax


# ============================================================
# Band-support histogram plots
# ============================================================

def plot_peak_timefreq_support_by_pair(
    tdoas_hat_pairs,
    n_bands,
    n_frames,
    hop,
    fs,
    frames_per_bin=1,
    frame_start=0,
    sources=None,
    cid2color=None,
    unassigned_color="0.5",
):
    """
    Plot one time-frequency support panel per detected peak and microphone pair.

    Panel borders and titles identify candidate-selection status:
      - solid source color: final selected peak
      - dashed source color: rejected same-pair candidate
      - dotted gray: unrelated or unassigned peak

    Returns
    -------
    figures : dict
        Mapping ``pair -> matplotlib Figure``.
    """
    peak2cid = build_peak_cluster_map(sources)
    rejected_peak2cid = {}
    for cluster_id, source in enumerate(sources or []):
        selection = source.get("candidate_selection", {})
        for member in selection.get("rejected_members", []):
            rejected_peak2cid[
                (tuple(member["pair"]), int(member["peak_idx"]))
            ] = cluster_id

    if cid2color is None:
        cmap_name = "tab10" if len(sources or []) <= 10 else "tab20"
        cid2color = build_cluster_color_map(sources, cmap_name=cmap_name)

    n_time_bins = int(np.ceil(n_frames / frames_per_bin))
    time_start = frame_start * hop / float(fs)
    time_stop = (frame_start + n_frames) * hop / float(fs)
    figures = {}

    for pair, pair_data in sorted(tdoas_hat_pairs.items()):
        peaks = pair_data.get("peaks", [])
        if not peaks:
            continue

        fig, axes = plt.subplots(
            len(peaks),
            1,
            figsize=(14, max(3.2, 2.8 * len(peaks))),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        axes = axes[:, 0]

        for peak_idx, (ax, peak) in enumerate(zip(axes, peaks)):
            vector = peak_timefreq_vector(
                peak,
                n_bands=n_bands,
                n_frames=n_frames,
                frames_per_bin=frames_per_bin,
                frame_start=frame_start,
            )
            support = vector.reshape(n_bands, n_time_bins)
            image = ax.imshow(
                support,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(time_start, time_stop, -0.5, n_bands - 0.5),
                cmap="magma",
                vmin=0.0,
            )

            peak_key = (tuple(pair), peak_idx)
            cluster_id = peak2cid.get(peak_key)
            rejected_cluster_id = rejected_peak2cid.get(peak_key)
            if cluster_id is not None:
                color = cid2color[cluster_id]
                cluster_label = f"source {cluster_id}, final selected"
                status_label = "SELECTED"
                spine_style = "-"
                spine_width = 4.0
            elif rejected_cluster_id is not None:
                color = cid2color[rejected_cluster_id]
                cluster_label = (
                    f"source {rejected_cluster_id}, rejected candidate"
                )
                status_label = "CANDIDATE"
                spine_style = "--"
                spine_width = 3.0
            else:
                color = unassigned_color
                cluster_label = "unassigned"
                status_label = "UNASSIGNED"
                spine_style = ":"
                spine_width = 2.0

            tau = float(peak.get("tau", np.nan))
            ax.set_title(
                f"Peak {peak_idx}: tau={tau:.1f} samples, {cluster_label}",
                color=color,
                loc="left",
            )
            ax.text(
                0.99,
                0.92,
                status_label,
                color=color,
                ha="right",
                va="top",
                fontweight="bold",
                transform=ax.transAxes,
                bbox={
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.85,
                    "boxstyle": "round,pad=0.25",
                },
            )
            ax.set_ylabel("Band")
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linestyle(spine_style)
                spine.set_linewidth(spine_width)
            fig.colorbar(image, ax=ax, label="Support count", pad=0.01)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Peak time-subband support - microphone pair {tuple(pair)}")
        figures[tuple(pair)] = fig

    return figures


def plot_single_peak_timefreq_support_and_frequency_projection(
    tdoas_hat_pairs,
    pair,
    peak_idx,
    n_bands,
    n_frames,
    hop,
    fs,
    frames_per_bin=1,
    frame_start=0,
    color="tab:blue",
):
    """Plot one peak's TF support and its frequency projection separately."""
    pair = tuple(map(int, pair))
    peak_idx = int(peak_idx)
    pair_data = tdoas_hat_pairs.get(pair)
    if pair_data is None:
        raise ValueError(f"Microphone pair {pair} not found.")
    peaks = pair_data.get("peaks", [])
    if not 0 <= peak_idx < len(peaks):
        raise ValueError(
            f"Peak {peak_idx} is unavailable for microphone pair {pair}."
        )

    peak = peaks[peak_idx]
    n_time_bins = int(np.ceil(n_frames / frames_per_bin))
    time_start = frame_start * hop / float(fs)
    time_stop = (frame_start + n_frames) * hop / float(fs)
    support = peak_timefreq_vector(
        peak,
        n_bands=n_bands,
        n_frames=n_frames,
        frames_per_bin=frames_per_bin,
        frame_start=frame_start,
    ).reshape(n_bands, n_time_bins)

    fig_tf, ax_tf = plt.subplots(figsize=(14, 5), constrained_layout=True)
    image = ax_tf.imshow(
        support,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(time_start, time_stop, -0.5, n_bands - 0.5),
        cmap="magma",
        vmin=0.0,
    )
    ax_tf.set_title(
        f"Time-subband support - pair {pair}, peak {peak_idx}"
    )
    ax_tf.set_xlabel("Time (s)")
    ax_tf.set_ylabel("Frequency zone index")
    fig_tf.colorbar(image, ax=ax_tf, label="Support", pad=0.01)

    frequency_projection = peak_band_vector(peak, n_bands)
    fig_frequency, ax_frequency = plt.subplots(
        figsize=(14, 5),
        constrained_layout=True,
    )
    band_indices = np.arange(n_bands)
    ax_frequency.bar(
        band_indices,
        frequency_projection,
        width=1.0,
        color=color,
        linewidth=0,
    )
    ax_frequency.set_title(
        f"Frequency projection - pair {pair}, peak {peak_idx}"
    )
    ax_frequency.set_xlabel("Frequency zone index")
    ax_frequency.set_ylabel("Supporting time regions")
    ax_frequency.set_xlim(-0.5, n_bands - 0.5)
    ax_frequency.grid(axis="y", alpha=0.25)

    return fig_tf, fig_frequency


def plot_tdoa_peak_band_histograms_three_pairs(
    tdoas_hat_pairs,
    pair_left,
    pair_mid,
    pair_right,
    n_bands,
    band_freqs=None,
    out_dir="Figures",
    base_font=24,
    tick_step=15,
    sources=None,
    cid2color=None,
    unassigned_color="0.75",
):
    """
    Plot band-support histograms for peaks from three microphone pairs (3 columns).

    Coloring
    --------
    If `sources` (association output) is provided:
      - peaks belonging to the same cluster are assigned the same color across pairs.
      - peaks not assigned to any cluster are plotted in `unassigned_color`.

    Expected structure
    ------------------
    tdoas_hat_pairs[(i,j)] is a dict with:
      - "peaks": list of peaks
    Each peak must contain "band_counts" or data usable by `peak_band_vector`.

    Returns
    -------
    fig, axes
    """
    ensure_dir(out_dir)

    def _get_peaks(pair):
        info = tdoas_hat_pairs.get(pair)
        if info is None:
            raise ValueError(f"Microphone pair {pair} not found in tdoas_hat_pairs.")
        return info.get("peaks", [])

    peaks_L = _get_peaks(pair_left)
    peaks_M = _get_peaks(pair_mid)
    peaks_R = _get_peaks(pair_right)

    n_rows = max(len(peaks_L), len(peaks_M), len(peaks_R))
    if n_rows == 0:
        raise ValueError("No peaks found for the provided microphone pairs.")

    # x-axis
    x = np.arange(n_bands) if band_freqs is None else np.asarray(band_freqs)
    x_label = "Frequency zone index" if band_freqs is None else "Frequency (Hz)"
    bar_width = 1 if band_freqs is None else (x[1] - x[0] if len(x) > 1 else 1.0)

    # Local rcParams
    with plt.rc_context({
        "font.size": base_font,
        "axes.titlesize": base_font,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 2,
        "ytick.labelsize": base_font - 2,
        "legend.fontsize": base_font - 2,
        "axes.linewidth": 1.4,
    }):
        fig_w = 20
        fig_h = max(9, 3.6 * n_rows)
        fig, axes = plt.subplots(
            n_rows, 3,
            figsize=(fig_w, fig_h),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if n_rows == 1:
            axes = np.array([axes])

        # Cluster coloring
        peak2cid = build_peak_cluster_map(sources) if sources is not None else {}
        if cid2color is None:
            cid2color = build_cluster_color_map(sources, cmap_name=("tab10" if len(sources or []) <= 10 else "tab20"))

        def color_for(pair, peak_idx):
            cid = peak2cid.get((tuple(pair), int(peak_idx)))
            if cid is None:
                return unassigned_color
            return cid2color[cid]

        def annotate_peak(ax, k):
            ax.text(
                0.72, 0.92, f"Peak {k}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=base_font,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
            )

        pairs = [pair_left, pair_mid, pair_right]
        peaks_all = [peaks_L, peaks_M, peaks_R]

        for k in range(n_rows):
            for col in range(3):
                ax = axes[k, col]
                pair = pairs[col]
                peaks = peaks_all[col]

                if k >= len(peaks):
                    ax.axis("off")
                    continue

                vec = peak_band_vector(peaks[k], n_bands)
                ax.bar(x, vec, width=bar_width, color=color_for(pair, k), linewidth=0)
                annotate_peak(ax, k)
                ax.grid(alpha=0.25, linewidth=1.0)

        # Column titles
        axes[0, 0].set_title(f"Pair {pair_left}")
        axes[0, 1].set_title(f"Pair {pair_mid}")
        axes[0, 2].set_title(f"Pair {pair_right}")

        # x labels only at bottom row
        for c in range(3):
            axes[-1, c].set_xlabel(x_label)

        # Reduce x tick clutter for band indices
        if band_freqs is None:
            ticks = np.arange(0, n_bands, tick_step)
            for c in range(3):
                axes[-1, c].set_xticks(ticks)

    return fig


# ============================================================
# Hyperbola plots
# ============================================================

HYPERBOLA_MIC_SIZE = 600
HYPERBOLA_LINEWIDTH = 10.0

def plot_sources_hyperbolas(
    sources,
    mic_positions,
    fs,
    env_size,
    ax=None,
    c=343.0,
    alpha=0.8,
    linewidth=HYPERBOLA_LINEWIDTH,
    show_legend=True,
    cid2color=None,
    report_undrawn=True,
    clip_impossible=False,
):
    """
    Plot hyperbolas for each associated source hypothesis (one color per hypothesis).

    Parameters
    ----------
    sources : list of dict
        Association output. Each source has "members".
        Each member must have:
          - "pair": (i,j)
          - "tau": TDOA in samples
    mic_positions : (M,2)
    fs : float
    env_size : float
    cid2color : dict or None
        Optional mapping {cluster_id: color}. If None, colors are generated internally.
    """
    mic_positions = np.asarray(mic_positions, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    plot_room(
        mic_positions,
        np.zeros((0, 2)),
        env_size,
        ax=ax,
        mic_size=HYPERBOLA_MIC_SIZE,
src_size=HYPERBOLA_MIC_SIZE,
    )

    if cid2color is None:
        cid2color = build_cluster_color_map(sources, cmap_name=("tab10" if len(sources) <= 10 else "tab20"))

    for s_idx, src in enumerate(sources):
        col = cid2color[s_idx]

        for member in src.get("members", []):
            i, j = member["pair"]
            tau = member["tau"]

            _, status = plot_tdoa_hyperbola(
                m0=mic_positions[i],
                m1=mic_positions[j],
                tdoa=tau,
                fs=fs,
                c=c,
                xlim=(0, env_size),
                ylim=(0, env_size),
                ax=ax,
                color=col,
                linewidths=linewidth,
                alpha=alpha,
                clip_to_physical=clip_impossible,
                return_status=True,
            )
            if report_undrawn and status == "drawn_clipped":
                pair_distance = float(np.linalg.norm(
                    mic_positions[j] - mic_positions[i]
                ))
                original_difference = abs(
                    float(c) * float(tau) / float(fs)
                )
                print(
                    f"[HYPERBOLA] Source {s_idx}, pair ({i}, {j}), "
                    f"tau={float(tau):.2f} samples clipped for plotting: "
                    f"|c*tau/fs|={original_difference:.2f} m -> "
                    f"{pair_distance:.2f} m."
                )
            elif report_undrawn and status not in {
                "drawn",
                "drawn_degenerate",
            }:
                pair_distance = float(np.linalg.norm(
                    mic_positions[j] - mic_positions[i]
                ))
                distance_difference = abs(float(c) * float(tau) / float(fs))
                print(
                    f"[HYPERBOLA] Source {s_idx}, pair ({i}, {j}), "
                    f"tau={float(tau):.2f} samples not drawn: {status}; "
                    f"|c*tau/fs|={distance_difference:.2f} m, "
                    f"pair distance={pair_distance:.2f} m."
                )

        # ax.plot([], [], color=col, label=rf"$\mathcal{{A}}_{{{s_idx+1}}}$")

    ax.set_title("Hyperbolas grouped by associated source")
    ax.set_xlim(0, env_size)
    ax.set_ylim(0, env_size)
    ax.set_aspect("equal")

    if show_legend:
        ax.legend(loc="best")

    return ax


def plot_truth_vs_estimates(micPos, srcPos, estPos, roomDims, ax=None, cid2color=None):
    """
    Plot microphones, true sources, and estimated sources.
    """
    micPos = np.asarray(micPos, float)
    estPos = np.asarray(estPos, float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if srcPos is None:
        plot_room(
            micPos,
            np.zeros((0, 2)),
            roomDims,
            ax=ax,
            mic_size=HYPERBOLA_MIC_SIZE,
            src_size=HYPERBOLA_MIC_SIZE,
        )
    else:
        plot_room(
            micPos,
            srcPos,
            roomDims,
            ax=ax,
            mic_size=HYPERBOLA_MIC_SIZE,
            src_size=HYPERBOLA_MIC_SIZE,
        )

    if estPos.size > 0:
        if cid2color is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(estPos)))
        else:
            colors = [cid2color[k] for k in range(len(estPos))]

        ax.scatter(
            estPos[:, 0], estPos[:, 1],
            s=600,
            marker="o",
            linewidths=8,
            alpha=1.0,
            color=colors,
            label="Estimated sources",
        )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")

    ax.set_title("Estimated source positions")
    return ax


def plot_all_pairs_hyperbolas(
    tdoas_hat_pairs,
    micPos,
    fs,
    env_size,
    srcPos=None,
    ax=None,
    c=343.0,
    alpha=0.5,
    linewidth=HYPERBOLA_LINEWIDTH,
    color_by="none",
    max_hyperbolas=None,
    sources=None,
    cid2color=None,
    unassigned_color="0.55",
):
    """
    Plot hyperbolas for all pairs and all estimated TDOA peaks.

    Expected structure
    ------------------
    tdoas_hat_pairs[(i,j)] contains:
      - "tdoas": list/array of TDOA peaks (in samples)

    Parameters
    ----------
    color_by : "pair", "source", or "none"
        If "pair", each mic pair uses a different color.
        If "source", peaks assigned to the same source cluster use the same color.
    max_hyperbolas : int or None
        If set, stops plotting after this many hyperbolas.
    sources : list of dict or None
        Association output. Required for color_by="source".
    cid2color : dict or None
        Optional mapping {cluster_id: color}. If None, colors are generated internally.
    """
    micPos = np.asarray(micPos, float)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    # Plot microphones and (optionally) true sources
    if srcPos is None:
        plot_room(
            micPos,
            np.zeros((0, 2)),
            env_size,
            ax=ax,
            mic_size=HYPERBOLA_MIC_SIZE,
src_size=HYPERBOLA_MIC_SIZE,
        )
    else:
        plot_room(
            micPos,
            srcPos,
            env_size,
            ax=ax,
            mic_size=HYPERBOLA_MIC_SIZE,
src_size=HYPERBOLA_MIC_SIZE,
        )

    pair_keys = list(tdoas_hat_pairs.keys())
    if color_by == "pair":
        pair_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pair_keys), 1)))
    else:
        pair_colors = [None] * len(pair_keys)

    if color_by == "source":
        peak2cid = build_peak_cluster_map(sources) if sources is not None else {}
        if cid2color is None:
            cid2color = build_cluster_color_map(sources, cmap_name=("tab10" if len(sources or []) <= 10 else "tab20"))
    else:
        peak2cid = {}

    count = 0
    for idx, (pair, info) in enumerate(tdoas_hat_pairs.items()):
        i, j = pair
        taus = np.asarray(
            info.get("peak_taus", info.get("tdoas", [])),
            dtype=float
        )
        if taus.size == 0:
            continue

        for peak_idx, tau in enumerate(taus):
            if color_by == "source":
                cid = peak2cid.get((tuple(pair), int(peak_idx)))
                line_color = cid2color[cid] if cid is not None else unassigned_color
            elif color_by == "pair":
                line_color = pair_colors[idx]
            else:
                line_color = None

            plot_tdoa_hyperbola(
                m0=micPos[i],
                m1=micPos[j],
                tdoa=tau,
                fs=fs,
                c=c,
                xlim=(0, env_size),
                ylim=(0, env_size),
                ax=ax,
                color=line_color,
                linewidths=linewidth,
                alpha=alpha,
            )
            count += 1
            if max_hyperbolas is not None and count >= int(max_hyperbolas):
                ax.set_title(f"Hyperbolas (stopped at {max_hyperbolas})")
                return ax

    ax.set_title("All hyperbolas from all pairs (all estimated TDOA peaks)")
    return ax


def load_gpx_positions(gpx_path):
    """
    Load AudioMoth GPS positions from a GPX file and sort by device number.

    Returns
    -------
    coords : ndarray (n,2)
        [lat, lon] coordinates sorted by AudioMoth number.
    names : list
        Waypoint names sorted the same way.
    """

    tree = ET.parse(gpx_path)
    root = tree.getroot()

    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}

    entries = []

    for wpt in root.findall("gpx:wpt", ns):

        lat = float(wpt.attrib["lat"])
        lon = float(wpt.attrib["lon"])

        name_elem = wpt.find("gpx:name", ns)
        name = name_elem.text if name_elem is not None else ""

        # extract AudioMoth number
        m = re.search(r"(\d+)$", name)
        num = int(m.group(1)) if m else -1

        entries.append((num, name, lat, lon))

    # sort by AudioMoth number
    entries.sort(key=lambda x: x[0])

    names = [e[1] for e in entries]
    coords = np.array([[e[2], e[3]] for e in entries])

    return coords, names


def gps_to_local(coords, ref_lat=None, ref_lon=None, return_ref=False):
    """
    Convert GPS coordinates [lat, lon] to local Cartesian coordinates [x, y] in meters.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
        GPS coordinates as [latitude, longitude].
    ref_lat, ref_lon : float or None
        Shared GPS reference/origin. If None, the mean of coords is used.
    return_ref : bool, default False
        If True, return (local_coords, ref_lat, ref_lon).

    Returns
    -------
    local_coords : ndarray, shape (n, 2)
        Local coordinates in meters as [east, north].
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")

    if ref_lat is None:
        ref_lat = float(coords[:, 0].mean())
    if ref_lon is None:
        ref_lon = float(coords[:, 1].mean())

    R = 6371000.0
    lat = np.deg2rad(coords[:, 0])
    lon = np.deg2rad(coords[:, 1])
    lat0 = np.deg2rad(ref_lat)
    lon0 = np.deg2rad(ref_lon)

    x = (lon - lon0) * np.cos(lat0) * R
    y = (lat - lat0) * R
    local_coords = np.c_[x, y]

    if return_ref:
        return local_coords, ref_lat, ref_lon

    return local_coords


def gps_groups_to_local(*gps_groups, ref_group=1, return_ref=False):
    """
    Convert several GPS position arrays to one shared local meter coordinate system.

    This prevents the common mistake of converting microphones and sources separately,
    which would give each group a different origin.

    Parameters
    ----------
    *gps_groups : arrays
        Any number of arrays with shape (n, 2), each containing [lat, lon].
        Example: gps_groups_to_local(micGPS, person1GPS, person2GPS)
    ref_group : int or 'all', default 0
        Which group defines the reference origin.
        - 0 means use the mean of the first group, usually the microphones.
        - 'all' means use the mean of all points together.
    return_ref : bool, default False
        If True, also return (ref_lat, ref_lon).

    Returns
    -------
    local_groups : tuple of ndarrays
        Converted arrays, in the same order as the inputs.
    ref_lat, ref_lon : float, optional
        Returned only if return_ref=True.
    """
    if len(gps_groups) == 0:
        raise ValueError("At least one GPS group must be provided.")

    gps_groups = [np.asarray(g, dtype=float) for g in gps_groups]

    for k, g in enumerate(gps_groups):
        if g.ndim != 2 or g.shape[1] != 2:
            raise ValueError(f"GPS group {k} must have shape (n, 2), got {g.shape}")

    if ref_group == "all":
        ref_coords = np.vstack(gps_groups)
    else:
        ref_coords = gps_groups[int(ref_group)]

    ref_lat = float(ref_coords[:, 0].mean())
    ref_lon = float(ref_coords[:, 1].mean())

    local_groups = tuple(
        gps_to_local(g, ref_lat=ref_lat, ref_lon=ref_lon)
        for g in gps_groups
    )

    if return_ref:
        return (*local_groups, ref_lat, ref_lon)

    return local_groups


def _normalize_audiomoth_id(audiomoth_id, prefix="AUD"):
    """
    Normalize AudioMoth identifiers to names like 'AUD26'.

    Examples
    --------
    26 -> 'AUD26'
    '26' -> 'AUD26'
    'AUD26' -> 'AUD26'
    ' aud26 ' -> 'AUD26'
    """
    s = str(audiomoth_id).strip().upper()
    prefix = str(prefix).strip().upper()

    if s.startswith(prefix):
        return s

    m = re.search(r"(\d+)$", s)
    if m is None:
        raise ValueError(f"Could not extract AudioMoth number from ID: {audiomoth_id!r}")

    return f"{prefix}{int(m.group(1))}"


def read_audiomoth_gps_csv(
    csv_path,
    desired_ids,
    *,
    return_dict=False,
    sort_like_requested=True,
    has_header="auto",
    lat_col=None,
    lon_col=None,
    id_col=None,
    prefix="AUD",
    missing_ok=False,
):
    import pandas as pd

    """
    Read a simple AudioMoth GPS CSV and return coordinates only for selected IDs.

    The expected CSV can be either headerless, for example:

        49.8399506,14.0997564,AUD26
        49.8396086,14.0996661,AUD27

    or it can have named columns such as:

        Latitude,Longitude,Device
        49.8399506,14.0997564,AUD26

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file.
    desired_ids : iterable
        AudioMoth names/IDs to load. Typical usage is
        ['AUD1', 'AUD2', 'AUD4'].
        Integers [1, 2, 4] are still accepted for backward compatibility.
    return_dict : bool, default False
        If False, returns an array with shape (n_selected, 2).
        If True, returns a dict {'AUD26': np.array([lat, lon]), ...}.
    sort_like_requested : bool, default True
        If True, output follows the order of desired_ids.
        If False, output follows the order found in the CSV.
    has_header : {'auto', True, False}, default 'auto'
        Whether the CSV has a header. 'auto' handles both common cases.
    lat_col, lon_col, id_col : str or int or None
        Optional explicit column names/indices. If None, the function tries to infer them.
    prefix : str, default 'AUD'
        Prefix used to normalize device IDs.
    missing_ok : bool, default False
        If True, requested AudioMoths missing from the CSV are skipped with a
        warning instead of raising an exception.

    Returns
    -------
    coords : ndarray, shape (n_selected, 2)
        GPS coordinates as [latitude, longitude], if return_dict=False.
    coord_dict : dict
        Mapping from normalized AudioMoth ID to [latitude, longitude], if return_dict=True.
    """
    desired_names = [_normalize_audiomoth_id(x, prefix=prefix) for x in desired_ids]
    desired_set = set(desired_names)

    def read_csv_auto(path, **kwargs):
        return pd.read_csv(path, sep=None, engine="python", **kwargs)

    if has_header == "auto":
        df_try = read_csv_auto(csv_path)
        cols_lower = [str(c).strip().lower() for c in df_try.columns]
        looks_like_header = any(c in cols_lower for c in ["latitude", "lat"])
        looks_like_header = looks_like_header or any(c in cols_lower for c in ["longitude", "lon", "long"])

        if looks_like_header:
            df = df_try
        else:
            df = read_csv_auto(csv_path, header=None)
    elif has_header:
        df = read_csv_auto(csv_path)
    else:
        df = read_csv_auto(csv_path, header=None)

    if lat_col is None or lon_col is None or id_col is None:
        if df.columns.dtype == "int64":
            inferred_lat_col = 0
            inferred_lon_col = 1
            inferred_id_col = 2
        else:
            col_lookup = {str(c).strip().lower(): c for c in df.columns}

            def pick_col(candidates, kind):
                for cand in candidates:
                    if cand in col_lookup:
                        return col_lookup[cand]
                raise ValueError(
                    f"Could not infer {kind} column. Available columns are: {list(df.columns)}"
                )

            inferred_lat_col = pick_col(["latitude", "lat"], "latitude")
            inferred_lon_col = pick_col(["longitude", "lon", "long"], "longitude")
            inferred_id_col = pick_col(["device", "id", "name", "audiomoth", "audiomoth_id"], "AudioMoth ID")
    else:
        inferred_lat_col = lat_col
        inferred_lon_col = lon_col
        inferred_id_col = id_col

    df = df.copy()
    df["_lat"] = pd.to_numeric(df[inferred_lat_col], errors="coerce")
    df["_lon"] = pd.to_numeric(df[inferred_lon_col], errors="coerce")
    # Remove empty spreadsheet rows before normalizing IDs. Otherwise a blank
    # ID is read as NaN and reported as a malformed AudioMoth identifier even
    # though the row contains no GPS record.
    df = df[df["_lat"].notna() & df["_lon"].notna()]
    df["_id"] = df[inferred_id_col].apply(
        lambda x: _normalize_audiomoth_id(x, prefix=prefix)
    )
    df = df[df["_id"].isin(desired_set)]

    found = set(df["_id"].tolist())
    missing = [name for name in desired_names if name not in found]
    if missing:
        if missing_ok:
            print(f"[WARNING] Missing requested AudioMoth IDs in {csv_path}: {missing}")
            desired_names = [name for name in desired_names if name in found]
        else:
            raise ValueError(f"Missing requested AudioMoth IDs in {csv_path}: {missing}")

    if df["_id"].duplicated().any():
        duplicated = sorted(df.loc[df["_id"].duplicated(), "_id"].unique().tolist())
        raise ValueError(f"Duplicate AudioMoth IDs found in {csv_path}: {duplicated}")

    if len(desired_names) == 0:
        raise ValueError("None of the requested AudioMoth IDs were found in the CSV.")
    coord_dict = {
        row["_id"]: np.array([float(row["_lat"]), float(row["_lon"])], dtype=float)
        for _, row in df.iterrows()
    }

    if return_dict:
        if sort_like_requested:
            return {name: coord_dict[name] for name in desired_names}
        return coord_dict

    if sort_like_requested:
        return np.array([coord_dict[name] for name in desired_names], dtype=float)

    return df[["_lat", "_lon"]].to_numpy(dtype=float)


# ============================================================
# Generic GPS position CSV reader
# ============================================================

def read_gps_position_csv(
    csv_path,
    desired_ids=None,
    *,
    return_dict=False,
    sort_like_requested=True,
    has_header="auto",
    lat_col=None,
    lon_col=None,
    id_col=None,
):
    import pandas as pd

    """
    Read a generic GPS position CSV and return selected source/location coordinates.

    This is useful for source/person positions where IDs are simple numbers,
    for example a headerless CSV like:

        49.85103,14.11028,1
        49.85121,14.11061,2
        49.85151,14.11085,3

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file.
    desired_ids : iterable or None, default None
        IDs to load. If None, all rows are returned.
    return_dict : bool, default False
        If False, returns an array with shape (n_selected, 2).
        If True, returns a dict {id: np.array([lat, lon]), ...}.
    sort_like_requested : bool, default True
        If True and desired_ids is not None, output follows desired_ids order.
        If False, output follows CSV order.
    has_header : {'auto', True, False}, default 'auto'
        Whether the CSV has a header. For files like Person1_REAL.csv, use
        has_header=False or leave as 'auto'.
    lat_col, lon_col, id_col : str or int or None
        Optional explicit column names/indices. For headerless files, defaults
        are lat_col=0, lon_col=1, id_col=2.

    Returns
    -------
    coords : ndarray, shape (n_selected, 2)
        GPS coordinates as [latitude, longitude], if return_dict=False.
    coord_dict : dict
        Mapping from source/location ID to [latitude, longitude], if return_dict=True.
    """
    if has_header == "auto":
        df_try = pd.read_csv(csv_path)
        cols_lower = [str(c).strip().lower() for c in df_try.columns]
        looks_like_header = any(c in cols_lower for c in ["latitude", "lat"])
        looks_like_header = looks_like_header or any(c in cols_lower for c in ["longitude", "lon", "long"])

        if looks_like_header:
            df = df_try
        else:
            df = pd.read_csv(csv_path, header=None)
    elif has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)

    if lat_col is None or lon_col is None or id_col is None:
        if df.columns.dtype == "int64":
            inferred_lat_col = 0
            inferred_lon_col = 1
            inferred_id_col = 2
        else:
            col_lookup = {str(c).strip().lower(): c for c in df.columns}

            def pick_col(candidates, kind):
                for cand in candidates:
                    if cand in col_lookup:
                        return col_lookup[cand]
                raise ValueError(
                    f"Could not infer {kind} column. Available columns are: {list(df.columns)}"
                )

            inferred_lat_col = pick_col(["latitude", "lat"], "latitude")
            inferred_lon_col = pick_col(["longitude", "lon", "long"], "longitude")
            inferred_id_col = pick_col(["id", "source", "source_id", "location", "location_id", "name"], "position ID")
    else:
        inferred_lat_col = lat_col
        inferred_lon_col = lon_col
        inferred_id_col = id_col

    df = df.copy()
    df["_lat"] = pd.to_numeric(df[inferred_lat_col], errors="coerce")
    df["_lon"] = pd.to_numeric(df[inferred_lon_col], errors="coerce")
    df["_id"] = df[inferred_id_col]

    df = df[df["_lat"].notna() & df["_lon"].notna() & df["_id"].notna()]

    # Keep numeric IDs as integers when possible. Otherwise keep strings.
    id_numeric = pd.to_numeric(df["_id"], errors="coerce")
    if id_numeric.notna().all():
        df["_id_norm"] = id_numeric.astype(int)
    else:
        df["_id_norm"] = df["_id"].astype(str).str.strip()

    if desired_ids is not None:
        desired_norm = []
        for x in desired_ids:
            try:
                desired_norm.append(int(x))
            except (TypeError, ValueError):
                desired_norm.append(str(x).strip())

        desired_set = set(desired_norm)
        df = df[df["_id_norm"].isin(desired_set)]

        found = set(df["_id_norm"].tolist())
        missing = [x for x in desired_norm if x not in found]
        if missing:
            raise ValueError(f"Missing requested IDs in {csv_path}: {missing}")
    else:
        desired_norm = None

    if df["_id_norm"].duplicated().any():
        duplicated = sorted(df.loc[df["_id_norm"].duplicated(), "_id_norm"].unique().tolist())
        raise ValueError(f"Duplicate IDs found in {csv_path}: {duplicated}")

    coord_dict = {
        row["_id_norm"]: np.array([float(row["_lat"]), float(row["_lon"])], dtype=float)
        for _, row in df.iterrows()
    }

    if return_dict:
        if desired_norm is not None and sort_like_requested:
            return {idx: coord_dict[idx] for idx in desired_norm}
        return coord_dict

    if desired_norm is not None and sort_like_requested:
        return np.array([coord_dict[idx] for idx in desired_norm], dtype=float)

    return df[["_lat", "_lon"]].to_numpy(dtype=float)

def attach_timefreq_info_to_peaks(
    tau_obs,
    bands,
    frames,
    peak_taus,
    tol_tau,
    n_bands=None,
    n_frames=None,
):
    """
    Old-format-compatible peak attachment.

    Keeps the old keys:
        - "tau"
        - "bands"
        - "band_counts"

    Adds new keys:
        - "frames"
        - "tf_cells"
        - "tf_support_map" (optional)
        - "obs_idx"
        - "n_support"
    """
    tau_obs = np.asarray(tau_obs)
    bands = np.asarray(bands)
    frames = np.asarray(frames)
    peak_taus = np.asarray(peak_taus)

    peaks = []

    for k, tau_hat in enumerate(peak_taus):
        mask = np.abs(tau_obs - tau_hat) <= tol_tau
        idx = np.flatnonzero(mask)

        peak_bands = bands[idx].copy()
        peak_frames = frames[idx].copy()

        band_counts = {}
        if peak_bands.size > 0:
            uniq_b, cnt_b = np.unique(peak_bands, return_counts=True)
            band_counts = {int(b): int(c) for b, c in zip(uniq_b, cnt_b)}

        peak_dict = {
            "peak_idx": k,
            "tau": float(tau_hat),          # keep OLD expected key
            "bands": peak_bands,            # keep OLD expected key
            "band_counts": band_counts,     # keep OLD expected key

            # new TF info
            "frames": peak_frames,
            "obs_idx": idx,
            "n_support": int(idx.size),
            "tf_cells": np.stack([peak_bands, peak_frames], axis=1) if idx.size > 0 else np.zeros((0, 2), dtype=int),
        }

        if n_bands is not None and n_frames is not None:
            tf_support_map = np.zeros((n_bands, n_frames), dtype=np.float32)
            for b, t in peak_dict["tf_cells"]:
                if 0 <= b < n_bands and 0 <= t < n_frames:
                    tf_support_map[b, t] += 1.0
            peak_dict["tf_support_map"] = tf_support_map

        peaks.append(peak_dict)

    return peaks

def build_source_tf_maps(
    sources,
    tdoas_hat_pairs,
    n_bands,
    n_frames,
):
    """
    Build TF support map per source cluster.
    Returns list of (n_bands, n_frames) arrays.
    """
    source_maps = []

    for src in sources:

        tf_map = np.zeros((n_bands, n_frames), dtype=float)

        for m in src["members"]:

            pair = tuple(m["pair"])
            peak_idx = int(m["peak_idx"])

            peak = tdoas_hat_pairs[pair]["peaks"][peak_idx]

            # tf_cells = [(band, frame), ...]
            for b, t in peak.get("tf_cells", []):
                tf_map[b, t] += 1

        if tf_map.max() > 0:
            tf_map /= tf_map.max()

        source_maps.append(tf_map)

    return source_maps


from scipy.signal import stft

def plot_reference_spectrogram(
    signals,
    fs,
    mic_idx=0,
    t_start=0.0,
    t_end=None,
    nperseg=1024,
    noverlap=None,
    nfft=None,
    db_floor=80.0,
    fmin=0.0,
    fmax=None,
    ax=None,
    title=None,
    cmap="gray_r",
    ref_db=None,
    fixed_clim=True,
):
    """
    Plot a standard spectrogram for one recorder over a chosen time interval.

    Parameters
    ----------
    signals : ndarray, shape (n_mics, n_samples)
    fs : float
    mic_idx : int
        Recorder / microphone index to plot.
    t_start, t_end : float
        Time interval in seconds.
    nperseg : int
        STFT window length.
    noverlap : int or None
        Overlap between windows. If None, uses nperseg // 2.
    nfft : int or None
        FFT size. If None, uses nperseg.
    db_floor : float
        Dynamic range shown in dB below peak.
    ax : matplotlib Axes or None
    title : str or None
    cmap : str

    Returns
    -------
    fig, ax
    """
    x = np.asarray(signals[mic_idx], dtype=float)

    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    n_samples = len(x)

    s0 = max(0, int(round(t_start * fs)))
    s1 = n_samples if t_end is None else min(n_samples, int(round(t_end * fs)))

    if s1 <= s0:
        raise ValueError("Chosen interval is empty.")

    x_seg = x[s0:s1]

    freqs, times, Zxx = stft(
        x_seg,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )

    S = np.abs(Zxx)
    S_db = 20 * np.log10(S + 1e-12)
    if ref_db is None:
        ref_db = S_db.max()
    S_db = S_db - ref_db
    if fixed_clim:
        S_db = np.clip(S_db, -db_floor, 0)

    keep_f = freqs >= float(fmin)
    if fmax is not None:
        keep_f &= freqs <= fmax
    freqs = freqs[keep_f]
    S_db = S_db[keep_f, :]
    if freqs.size == 0:
        raise ValueError("No spectrogram bins remain between fmin and fmax")

    times = times + t_start  # shift back to global time axis

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
    else:
        fig = ax.figure

    ax.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=cmap,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        title or f"Recorder {mic_idx} spectrogram (nperseg={nperseg}, nfft={nfft})"
    )

    return fig, ax


def save_birdnet_input_spectrograms(
    signals,
    fs,
    output_dir,
    *,
    source_ids=None,
    window_info=None,
    show_plots=False,
):
    """Save spectrograms of the exact signal arrays passed to BirdNET."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals = list(signals)
    if source_ids is None:
        source_ids = list(range(len(signals)))
    if window_info is None:
        window_info = [{} for _ in signals]

    for local_idx, x in enumerate(signals):
        x = np.asarray(x)
        if x.size == 0:
            continue

        source_id = source_ids[local_idx]
        info = window_info[local_idx] if local_idx < len(window_info) else {}
        selected_start = float(info.get("start_s", 0.0))
        selected_end = float(
            info.get("end_s", selected_start + x.size / float(fs))
        )
        fig, _ = plot_reference_spectrogram(
            signals=x[None, :],
            fs=fs,
            mic_idx=0,
            t_start=0.0,
            t_end=x.size / float(fs),
            nperseg=1024,
            noverlap=512,
            nfft=1024,
            title=(
                f"BirdNET input source {source_id} "
                f"(selected {selected_start:.2f}-{selected_end:.2f} s)"
            ),
            fmin=0.0,
            fmax=fs / 2.0,
            ref_db=None,
        )
        savefig(
            fig,
            str(output_dir / f"source_{source_id}_birdnet_input"),
            save_pdf=False,
            save_png=True,
            close=not show_plots,
        )

    if show_plots and signals:
        plt.show()


def hz_seconds_to_tf_size(freq_hz, time_sec, band_ranges_hz, hop, fs):
    """
    Convert a morphology structuring-element span from Hz/seconds to TF cells.

    The source maps use frequency bands on axis 0 and algorithm frames on axis 1.
    Returned sizes are suitable for morph_source_maps(..., *_size=(freq, time)).
    """
    freq_hz = float(freq_hz)
    time_sec = float(time_sec)
    hop = float(hop)
    fs = float(fs)

    if freq_hz <= 0:
        freq_cells = 0
    else:
        centers = []
        widths = []
        for lo, hi in band_ranges_hz:
            lo = float(lo)
            hi = float(hi)
            centers.append(0.5 * (lo + hi))
            widths.append(max(hi - lo, 1e-12))
        if len(centers) >= 2:
            freq_step = float(np.median(np.diff(np.sort(centers))))
        elif len(widths) == 1:
            freq_step = float(widths[0])
        else:
            freq_step = max(freq_hz, 1.0)
        freq_cells = max(1, int(np.ceil(freq_hz / max(freq_step, 1e-12))))

    if time_sec <= 0:
        time_cells = 0
    else:
        time_step = hop / fs
        time_cells = max(1, int(np.ceil(time_sec / max(time_step, 1e-12))))

    return (freq_cells, time_cells)


def hz_seconds_to_stft_sigma(freq_hz, time_sec, fs, nfft, hop):
    """Convert taper fade radius from Hz/seconds to Gaussian sigma.

    The public Hz/sec controls are intended to mean "the fade mostly dies out
    by this distance", not "this distance is one Gaussian standard deviation".
    Use a 3-sigma radius so the visible border stays close to the requested
    physical span.
    """
    freq_sigma = max(0.0, float(freq_hz)) / (3.0 * (float(fs) / float(nfft)))
    time_sigma = max(0.0, float(time_sec)) / (3.0 * (float(hop) / float(fs)))
    return freq_sigma, time_sigma


def project_source_maps_to_visual_grid(
    source_maps,
    Omega_list_alg,
    fs,
    nfft_alg,
    hop_alg,
    n_frames_alg,
    nfft_rec,
    hop_rec,
    n_frames_rec,
):
    """
    Project source maps from the algorithm TF grid to a high-resolution
    visualization TF grid.

    Parameters
    ----------
    source_maps : list of arrays
        Each entry has shape (n_bands_alg, n_frames_alg).
    Omega_list_alg : list
        Algorithm band definition. Omega_list_alg[b] contains FFT-bin indices
        in the algorithm STFT corresponding to band b.
    fs : float
        Sampling rate.
    nfft_alg : int
        FFT size of the algorithm STFT.
    hop_alg : int
        Hop size of the algorithm STFT.
    n_frames_alg : int
        Number of algorithm STFT frames.
    nfft_vis : int
        FFT size of the visualization STFT.
    hop_vis : int
        Hop size of the visualization STFT.
    n_frames_vis : int
        Number of visualization STFT frames.

    Returns
    -------
    projected_maps : list of arrays
        Each entry has shape (n_freq_vis, n_frames_vis).
    """
    n_freq_vis = nfft_rec // 2 + 1
    projected_maps = []

    # Frequencies of the algorithm FFT bins
    freqs_alg = np.arange(nfft_alg // 2 + 1) * fs / nfft_alg
    freqs_vis = np.arange(n_freq_vis) * fs / nfft_rec

    # Times of the frame centers
    times_alg = np.arange(n_frames_alg) * hop_alg / fs
    times_vis = np.arange(n_frames_rec) * hop_rec / fs

    # For each band, estimate its frequency span
    band_ranges_hz = []
    for Omega_idx in Omega_list_alg:
        Omega_idx = np.asarray(Omega_idx, dtype=int)
        if Omega_idx.size == 0:
            band_ranges_hz.append((None, None))
            continue
        fmin = freqs_alg[Omega_idx.min()]
        fmax = freqs_alg[Omega_idx.max()]
        band_ranges_hz.append((fmin, fmax))

    for band_map in source_maps:
        n_bands_alg, _ = band_map.shape

        # --------
        # 1) time interpolation: (n_bands_alg, n_frames_alg) -> (n_bands_alg, n_frames_vis)
        # --------
        band_time_interp = np.zeros((n_bands_alg, n_frames_rec), dtype=float)

        for b in range(n_bands_alg):
            y = band_map[b]
            if np.all(y == 0):
                continue

            band_time_interp[b] = np.interp(
                times_vis,
                times_alg,
                y,
                left=0.0,
                right=0.0,
            )

        # --------
        # 2) frequency expansion: band axis -> high-res FFT bins
        # --------
        freq_map_vis = np.zeros((n_freq_vis, n_frames_rec), dtype=float)

        for b in range(n_bands_alg):
            fmin, fmax = band_ranges_hz[b]
            if fmin is None:
                continue

            freq_mask = (freqs_vis >= fmin) & (freqs_vis <= fmax)
            if not np.any(freq_mask):
                continue

            freq_map_vis[freq_mask, :] = np.maximum(
                freq_map_vis[freq_mask, :],
                band_time_interp[b][None, :]
            )

        projected_maps.append(freq_map_vis)

    return projected_maps


def compute_highres_spectrogram(
    signals,
    fs,
    mic_idx,
    t_start=0.0,
    t_end=None,
    nperseg=1024,
    noverlap=512,
    nfft=1024,
):
    """Compute reusable STFT data for high-resolution spectrogram plots."""
    x = np.asarray(signals[mic_idx], dtype=float)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    n_samples = len(x)
    s0 = max(0, int(round(t_start * fs)))
    s1 = n_samples if t_end is None else min(
        n_samples,
        int(round(t_end * fs)),
    )
    if s1 <= s0:
        raise ValueError("Chosen interval is empty.")

    freqs, times, Zxx = stft(
        x[s0:s1],
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    return freqs, times + t_start, Zxx


def plot_highres_spectrogram_with_projected_clusters(
    signals,
    fs,
    mic_idx,
    cid2color,
    projected_maps,
    t_start=0.0,
    t_end=None,
    nperseg=1024,
    noverlap=512,
    nfft=1024,
    alpha=0.5,
    db_floor=80.0,
    fmin=0.0,
    fmax=None,
    title="High-resolution spectrogram with projected source regions",
    cmap="gray_r",
    ref_db=None,
    fixed_clim=True,
    spectrogram_data=None,

):
    """
    Plot a high-resolution spectrogram and overlay projected cluster colors.
    """
    if spectrogram_data is None:
        freqs, times, Zxx = compute_highres_spectrogram(
            signals,
            fs,
            mic_idx,
            t_start=t_start,
            t_end=t_end,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
        )
    else:
        freqs, times, Zxx = spectrogram_data
        freqs = np.asarray(freqs)
        times = np.asarray(times)
        Zxx = np.asarray(Zxx)

    S = np.abs(Zxx)
    S_db = 20 * np.log10(S + 1e-12)
    if ref_db is None:
        ref_db = S_db.max()
    S_db = S_db - ref_db
    if fixed_clim:
        S_db = np.clip(S_db, -db_floor, 0)

    keep_f = freqs >= float(fmin)
    if fmax is not None:
        keep_f &= freqs <= fmax
    n_freq_full = freqs.size
    freqs = freqs[keep_f]
    S_db = S_db[keep_f, :]
    if freqs.size == 0:
        raise ValueError("No spectrogram bins remain between fmin and fmax")

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.imshow(
        S_db,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap=cmap,
    )

    n_freq_vis, n_frames_vis = S_db.shape

    for cid, freq_map in enumerate(projected_maps):
        freq_map = np.asarray(freq_map, dtype=float)

        if freq_map.ndim != 2:
            raise ValueError(f"Projected map for cluster {cid} must be 2D, got shape {freq_map.shape}")

        if freq_map.shape[0] == n_freq_full:
            freq_map = freq_map[keep_f, :]
        else:
            freq_map = freq_map[:min(freq_map.shape[0], n_freq_vis), :]

        if np.max(freq_map) <= 0:
            continue

        freq_map = freq_map / (np.max(freq_map) + 1e-12)

        # Same philosophy as plot_valid_mask_on_reference_spectrogram:
        # the overlay does not need to have the same number of time/frequency
        # pixels as the spectrogram. imshow uses the extent to stretch the map
        # over the same physical time/frequency axes.
        rgba = np.zeros((freq_map.shape[0], freq_map.shape[1], 4), dtype=float)
        rgba[..., :3] = cid2color[cid][:3]
        rgba[..., 3] = alpha * freq_map

        ax.imshow(
            rgba,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            interpolation="nearest",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    return fig, ax, freqs, times, Zxx


def read_audiomoth_wavs(
    root_path,
    desired_ids,
    wav_pattern="*.WAV",
    prefix="AUD",
    mono=True,
    trim_to_shortest=True,
    return_names=True,
    return_dict=False,
):
    """
    Read one WAV from each requested AUDxx folder.

    Returns
    -------
    signals : ndarray, shape (n_audiomoths, n_samples)
    fs : int
    names : list[str]
    """

    root_path = Path(root_path)
    names = [_normalize_audiomoth_id(x, prefix=prefix) for x in desired_ids]

    loaded = []
    fs = None

    for name in names:
        folder = root_path / name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        wavs = sorted(folder.glob(wav_pattern))
        if not wavs and wav_pattern == "*.WAV":
            wavs = sorted(folder.glob("*.wav"))

        if len(wavs) == 0:
            raise FileNotFoundError(f"No WAV found in {folder}")

        if len(wavs) > 1:
            raise ValueError(f"Expected one WAV in {folder}, found {len(wavs)}: {wavs}")

        wav_path = wavs[0]

        if fs is None:
            x, fs = librosa.load(wav_path, sr=None, mono=mono)
        else:
            x, _ = librosa.load(wav_path, sr=fs, mono=mono)

        loaded.append(np.asarray(x, dtype=np.float32))

    lengths = np.array([len(x) for x in loaded])

    if len(set(lengths)) != 1:
        if trim_to_shortest:
            min_len = lengths.min()
            loaded = [x[:min_len] for x in loaded]
        else:
            raise ValueError(f"Different signal lengths: {dict(zip(names, lengths))}")

    if return_dict:
        signal_dict = {
            name: sig
            for name, sig in zip(names, loaded)
        }

        if return_names:
            return signal_dict, fs, names

        return signal_dict, fs

    signals = np.stack(loaded, axis=0)

    if return_names:
        return signals, fs, names

    return signals, fs


# ============================================================
# AudioMoth date/time-based WAV reading helpers
# ============================================================

def _normalize_recording_date_and_time(recording_date, recording_time):
    """
    Normalize AudioMoth recording date/time inputs.

    Parameters
    ----------
    recording_date : str or int
        Either YYYYMMDD, for example 20260225, or YYMMDD, for example 260225.
        The returned folder date is YYYYMMDD because the recording folders use
        the same date string as the filename.
    recording_time : str or int
        HHMMSS. Values such as 60001 are padded to 060001.

    Returns
    -------
    file_date : str
        YYYYMMDD date used in AudioMoth filenames.
    folder_date : str
        YYYYMMDD date used by the date folder.
    file_time : str
        HHMMSS time used in AudioMoth filenames.
    """
    date_str = str(recording_date).strip()
    time_str = str(recording_time).strip()

    if not date_str.isdigit():
        raise ValueError(f"recording_date must contain only digits, got {recording_date!r}")
    if not time_str.isdigit():
        raise ValueError(f"recording_time must contain only digits, got {recording_time!r}")

    if len(date_str) == 8:
        file_date = date_str
        folder_date = date_str
    elif len(date_str) == 6:
        file_date = "20" + date_str
        folder_date = file_date
    else:
        raise ValueError(
            f"recording_date must be YYYYMMDD or YYMMDD, got {recording_date!r}"
        )

    file_time = time_str.zfill(6)
    if len(file_time) != 6:
        raise ValueError(
            f"recording_time must fit HHMMSS after zero-padding, got {recording_time!r}"
        )

    return file_date, folder_date, file_time


def read_audiomoth_wavs_by_datetime(
    root_path,
    desired_ids,
    recording_date,
    recording_time,
    *,
    require_sync=True,
    prefix="AUD",
    mono=True,
    trim_to_shortest=True,
    return_names=True,
    return_dict=False,
    return_paths=False,
):
    """
    Read one AudioMoth WAV per requested recorder for a specific recording date/time.

    Expected folder/file organization
    ---------------------------------
    root_path/
        AUD1/
            YYYYMMDD/
                242A260463FB62A5_YYYYMMDD_HHMMSS_SYNC.WAV
        AUD2/
            YYYYMMDD/
                ...

    Example
    -------
    For filename:
        242A260463FB62A5_20260225_060001_SYNC.WAV

    call:
        read_audiomoth_wavs_by_datetime(root, [1, 2, 3], 20260225, 60001)

    Parameters
    ----------
    root_path : str or pathlib.Path
        Root directory containing AUD folders.
    desired_ids : iterable
        AudioMoth IDs, for example [1, 2, 3], ["AUD1", "AUD2"], or mixed.
    recording_date : str or int
        Date in YYYYMMDD or YYMMDD format. The folder is assumed to use YYYYMMDD.
    recording_time : str or int
        Time in HHMMSS format. Values like 60001 are interpreted as 060001.
    require_sync : bool, default True
        If True, only files ending in _SYNC before the extension are accepted.
        If False, both synced and non-synced files with the requested date/time are accepted.
    prefix : str, default "AUD"
        Prefix used to normalize AudioMoth folder names.
    mono : bool, default True
        Passed to librosa.load.
    trim_to_shortest : bool, default True
        If recordings differ in length, trim all to the shortest length.
    return_names : bool, default True
        If True, return the normalized AudioMoth names.
    return_dict : bool, default False
        If True, return a dict {"AUDx": signal}. This is recommended when some
        recorders may be missing because skipped recorders are omitted without
        changing the remaining IDs.
    return_paths : bool, default False
        If True, also return selected WAV paths.

    Returns
    -------
    signals : ndarray or dict
        If return_dict=False: shape (n_audiomoths, n_samples).
        If return_dict=True: dict mapping normalized AudioMoth IDs to signals.
        Missing AudioMoths are skipped with a warning instead of raising an
        exception.
    fs : int
        Sampling rate.
    names : list[str], optional
        Returned if return_names=True.
    paths : list[pathlib.Path] or dict, optional
        Returned if return_paths=True.
    """
    root_path = Path(root_path)
    names = [_normalize_audiomoth_id(x, prefix=prefix) for x in desired_ids]
    file_date, folder_date, file_time = _normalize_recording_date_and_time(
        recording_date,
        recording_time,
    )

    loaded = []
    loaded_names = []
    selected_paths = []
    fs = None

    for name in names:
        folder = root_path / name / folder_date

        if not folder.exists():
            raise FileNotFoundError(f"Missing date folder for {name}: {folder}")

        candidates = []
        for ext in (".WAV", ".wav"):
            if require_sync:
                patterns = (
                    f"{file_date}_{file_time}_SYNC{ext}",
                    f"*_{file_date}_{file_time}_SYNC{ext}",
                )
            else:
                patterns = (
                    f"{file_date}_{file_time}{ext}",
                    f"{file_date}_{file_time}_SYNC{ext}",
                    f"*_{file_date}_{file_time}*{ext}",
                )
            for pattern in patterns:
                candidates.extend(sorted(folder.glob(pattern)))

        # Exact timestamp names can also match the prefixed glob, and
        # case-insensitive filesystems can return the same file once for each
        # extension spelling. Preserve order while deduplicating by file ID.
        unique_candidates = []
        seen_files = set()
        for candidate in candidates:
            stat = candidate.stat()
            file_id = (stat.st_dev, stat.st_ino)
            if file_id not in seen_files:
                seen_files.add(file_id)
                unique_candidates.append(candidate)
        candidates = unique_candidates

        if not require_sync:
            # Avoid accidental matches from files with additional unrelated text after
            # the requested timestamp, while still allowing the optional _SYNC suffix.
            valid_suffixes = {
                f"_{file_date}_{file_time}",
                f"_{file_date}_{file_time}_SYNC",
            }
            candidates = [
                p for p in candidates
                if any(p.stem.endswith(suffix) for suffix in valid_suffixes)
            ]

        if len(candidates) == 0:
            sync_msg = " with _SYNC" if require_sync else " with or without _SYNC"
            print(
                f"[WARNING] No WAV found for {name} at {file_date}_{file_time}{sync_msg} in {folder}. Skipping recorder."
            )
            continue

        if len(candidates) > 1:
            raise ValueError(
                f"Expected one WAV for {name} at {file_date}_{file_time}, "
                f"found {len(candidates)}: {candidates}"
            )

        wav_path = candidates[0]
        selected_paths.append(wav_path)

        if fs is None:
            x, fs = librosa.load(wav_path, sr=None, mono=mono)
        else:
            x, _ = librosa.load(wav_path, sr=fs, mono=mono)

        loaded.append(np.asarray(x, dtype=np.float32))
        loaded_names.append(name)

    if len(loaded) == 0:
        raise FileNotFoundError(
            f"No matching AudioMoth recordings found for {file_date}_{file_time}."
        )

    lengths = np.array([len(x) for x in loaded])

    if len(set(lengths)) != 1:
        if trim_to_shortest:
            min_len = lengths.min()
            loaded = [x[:min_len] for x in loaded]
        else:
            raise ValueError(f"Different signal lengths: {dict(zip(loaded_names, lengths))}")

    if return_dict:
        signals_out = {name: sig for name, sig in zip(loaded_names, loaded)}
        paths_out = {name: path for name, path in zip(loaded_names, selected_paths)}
    else:
        signals_out = np.stack(loaded, axis=0)
        paths_out = selected_paths

    outputs = [signals_out, fs]

    if return_names:
        outputs.append(loaded_names)
    if return_paths:
        outputs.append(paths_out)


    return tuple(outputs)


def _normalize_site_id(site_id, prefix="site_"):
    """Normalize site identifiers to folder names like 'site_1'."""
    s = str(site_id).strip().lower()
    prefix = str(prefix).strip().lower()

    if s.startswith(prefix):
        return s

    m = re.search(r"(\d+)$", s)
    if m is None:
        raise ValueError(f"Could not extract site number from ID: {site_id!r}")

    return f"{prefix}{int(m.group(1))}"


def read_jardin_site_wavs_by_datetime(
    root_path,
    site_ids,
    recording_date,
    recording_time,
    *,
    sr=None,
    mono=True,
    trim_to_shortest=True,
    return_names=True,
    return_paths=False,
):
    """
    Read WAVs from the Jardin Ecologique site/audiomoth folder layout.

    Expected folder/file organization
    ---------------------------------
    root_path/
        site_1/
            audiomoth/
                YYYYMMDD/
                    248C7D075B1F1E11_YYYYMMDD_HHMMSS.WAV
        site_2/
            audiomoth/
                YYYYMMDD/
                    ...
    """
    root_path = Path(root_path)
    names = [_normalize_site_id(site_id) for site_id in site_ids]
    file_date, folder_date, file_time = _normalize_recording_date_and_time(
        recording_date,
        recording_time,
    )

    loaded = []
    selected_paths = []
    fs = None if sr is None else int(sr)

    for name in names:
        folder = root_path / name / "audiomoth" / folder_date
        if not folder.exists():
            raise FileNotFoundError(f"Missing date folder for {name}: {folder}")

        stem_pattern = f"*_{file_date}_{file_time}"
        candidates = []
        for ext in (".WAV", ".wav"):
            candidates.extend(sorted(folder.glob(stem_pattern + ext)))

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No WAV found for {name} at {file_date}_{file_time} in {folder}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Expected one WAV for {name} at {file_date}_{file_time}, "
                f"found {len(candidates)}: {candidates}"
            )

        wav_path = candidates[0]
        selected_paths.append(wav_path)

        if fs is None:
            x, fs = librosa.load(wav_path, sr=None, mono=mono)
        else:
            x, _ = librosa.load(wav_path, sr=fs, mono=mono)

        loaded.append(np.asarray(x, dtype=np.float32))

    lengths = np.array([len(x) for x in loaded])
    if len(set(lengths)) != 1:
        if trim_to_shortest:
            min_len = lengths.min()
            loaded = [x[:min_len] for x in loaded]
        else:
            raise ValueError(f"Different signal lengths: {dict(zip(names, lengths))}")

    outputs = [np.stack(loaded, axis=0), fs]
    if return_names:
        outputs.append(names)
    if return_paths:
        outputs.append(selected_paths)

    return tuple(outputs)


# ============================================================
# Raven annotation table readers
# ============================================================

def read_raven_selection_times(
    txt_path,
    selection_ids=None,
    *,
    return_dict=True,
    include_annotation=False,
    include_confidence=False,
):
    import pandas as pd

    """
    Read selection start/end times from a Raven selection table.

    Expected columns include:
        Selection, Begin Time (s), End Time (s)

    Parameters
    ----------
    txt_path : str or pathlib.Path
        Path to a Raven .selection.table.txt file.
    selection_ids : iterable or None
        Selection IDs to keep. If None, all selections are returned.
    return_dict : bool, default True
        If True, return {selection_id: (start_s, end_s)}.
        If include_annotation=True, return
        {selection_id: {"start": start_s, "end": end_s, ...}}.
    include_annotation : bool, default False
        If True, also include the Annotation column when available.
    include_confidence : bool, default False
        If True, include the case-insensitive ``confidence`` column. Raises
        ValueError when that column is unavailable.

    Returns
    -------
    annotations : dict or ndarray
        Selection timing information.
    """
    txt_path = Path(txt_path)

    if not txt_path.exists():
        raise FileNotFoundError(f"Missing Raven selection table: {txt_path}")

    df = pd.read_csv(txt_path, sep="\t")

    required = ["Selection", "Begin Time (s)", "End Time (s)"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns {missing_cols} in {txt_path}. "
            f"Available columns are: {list(df.columns)}"
        )

    keep_cols = required.copy()
    has_annotation = "Annotation" in df.columns
    if include_annotation and has_annotation:
        keep_cols.append("Annotation")
    confidence_column = next(
        (
            column
            for column in df.columns
            if str(column).strip().lower() == "confidence"
        ),
        None,
    )
    if include_confidence:
        if confidence_column is None:
            raise ValueError(
                f"No confidence column found in {txt_path}. "
                f"Available columns are: {list(df.columns)}"
            )
        keep_cols.append(confidence_column)

    df = df[keep_cols].copy()
    df["Selection"] = pd.to_numeric(df["Selection"], errors="coerce")
    df["Begin Time (s)"] = pd.to_numeric(df["Begin Time (s)"], errors="coerce")
    df["End Time (s)"] = pd.to_numeric(df["End Time (s)"], errors="coerce")
    required_numeric = ["Selection", "Begin Time (s)", "End Time (s)"]
    if include_confidence:
        df[confidence_column] = pd.to_numeric(
            df[confidence_column],
            errors="coerce",
        )
        required_numeric.append(confidence_column)
    df = df.dropna(subset=required_numeric)

    df["Selection"] = df["Selection"].astype(int)
    df["Begin Time (s)"] = df["Begin Time (s)"].astype(float)
    df["End Time (s)"] = df["End Time (s)"].astype(float)

    if selection_ids is not None:
        selection_ids = [int(x) for x in selection_ids]
        df = df[df["Selection"].isin(selection_ids)]

    if return_dict:
        out = {}
        for _, row in df.iterrows():
            selection_id = int(row["Selection"])
            start_s = float(row["Begin Time (s)"])
            end_s = float(row["End Time (s)"])

            if include_annotation or include_confidence:
                value = {
                    "start": start_s,
                    "end": end_s,
                }
                if include_annotation:
                    value["annotation"] = (
                        str(row["Annotation"]) if has_annotation else None
                    )
                if include_confidence:
                    value["confidence"] = float(row[confidence_column])
                out[selection_id] = value
            else:
                out[selection_id] = (start_s, end_s)

        return out

    output_columns = ["Selection", "Begin Time (s)", "End Time (s)"]
    if include_annotation and has_annotation:
        output_columns.append("Annotation")
    if include_confidence:
        output_columns.append(confidence_column)
    if len(output_columns) > 3:
        return df[output_columns].to_numpy()

    return df[["Selection", "Begin Time (s)", "End Time (s)"]].to_numpy(dtype=float)


def gps_groups_to_centered_plot_coords(*gps_groups, ref_group=0, center=np.array([80.0, 80.0]), context_margin_m=60.0):
    """
    Convert GPS groups to local meter coordinates for the algorithm/plots, but keep
    enough metadata to draw a correctly aligned satellite basemap behind them.

    Returns
    -------
    centered_groups : tuple[np.ndarray]
        Local coordinates shifted so the geometry is centered around `center`.
    map_info : dict
        Information needed by add_satellite_background_to_local_axes().
    """
    local_groups = gps_groups_to_local(*gps_groups, ref_group=ref_group)

    all_local = np.vstack(local_groups)
    local_min = all_local.min(axis=0)
    local_max = all_local.max(axis=0)
    local_center = 0.5 * (local_min + local_max)

    plot_origin_local = local_center - np.asarray(center, dtype=float)
    centered_groups = tuple(g - plot_origin_local for g in local_groups)

    all_centered = np.vstack(centered_groups)
    data_min = all_centered.min(axis=0)
    data_max = all_centered.max(axis=0)

    # Make the plotting domain square because the existing geometry plotting
    # helpers use roomDims/environmentSize and set axes to [0, environmentSize].
    data_center = 0.5 * (data_min + data_max)
    side = float(max(data_max - data_min) + 2.0 * context_margin_m)
    plot_min = data_center - side / 2.0
    plot_max = data_center + side / 2.0

    # Shift everything once more so the final local plotting domain is exactly
    # [0, environmentSize] x [0, environmentSize]. This avoids the white border
    # caused by the plotting helpers resetting the limits to roomDims.
    centered_groups = tuple(g - plot_min for g in centered_groups)
    plot_origin_local = plot_origin_local + plot_min
    plot_min = np.array([0.0, 0.0], dtype=float)
    plot_max = np.array([side, side], dtype=float)

    map_info = {
        "plot_origin_local": plot_origin_local,
        "plot_min": plot_min,
        "plot_max": plot_max,
        "environment_size": side,
    }

    return centered_groups, map_info


def _gps_lat_lon_arrays(gps_group):
    """
    This code assumes the same convention used elsewhere in this script:
    gps_group[:, 0] = latitude, gps_group[:, 1] = longitude.
    """
    gps_group = np.asarray(gps_group, dtype=float)
    lat = gps_group[:, 0]
    lon = gps_group[:, 1]
    return lat, lon


def add_satellite_background_to_local_axes(
    ax,
    gps_groups,
    local_groups_centered,
    map_info,
    zoom=18,
    source=xyz.Esri.WorldImagery, # xyz.OpenStreetMap.Mapnik
    alpha=1.0,
):
    """
    Add satellite imagery to an axis whose data coordinates are your centered
    local meter coordinates.

    The image is fetched in EPSG:3857 using GPS bounds, then displayed with an
    extent converted back into the same centered local coordinate system used by
    the TDOA pipeline.
    """
    gps_all = np.vstack(gps_groups)
    local_all = np.vstack(local_groups_centered)

    lat, lon = _gps_lat_lon_arrays(gps_all)
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_3857, y_3857 = to_3857.transform(lon, lat)
    merc_all = np.column_stack([x_3857, y_3857])

    # Fit an affine transform between centered local coordinates and Web Mercator.
    # This preserves your local coordinate convention while allowing correct map alignment.
    A = np.column_stack([local_all, np.ones(local_all.shape[0])])
    coef_x, *_ = np.linalg.lstsq(A, merc_all[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, merc_all[:, 1], rcond=None)

    def local_to_merc(local_xy):
        local_xy = np.asarray(local_xy, dtype=float)
        A_xy = np.column_stack([local_xy, np.ones(local_xy.shape[0])])
        return np.column_stack([A_xy @ coef_x, A_xy @ coef_y])

    # Inverse affine transform. This is important because contextily returns
    # imagery snapped to full web tiles, so the returned extent is usually larger
    # than the requested extent. Using the requested local extent directly can
    # create a visible position shift.
    H = np.array([
        [coef_x[0], coef_x[1], coef_x[2]],
        [coef_y[0], coef_y[1], coef_y[2]],
        [0.0,       0.0,       1.0],
    ], dtype=float)
    H_inv = np.linalg.inv(H)

    def merc_to_local(merc_xy):
        merc_xy = np.asarray(merc_xy, dtype=float)
        M_xy = np.column_stack([merc_xy, np.ones(merc_xy.shape[0])])
        local_h = M_xy @ H_inv.T
        return local_h[:, :2]

    plot_min = np.asarray(map_info["plot_min"], dtype=float)
    plot_max = np.asarray(map_info["plot_max"], dtype=float)
    local_corners = np.array([
        [plot_min[0], plot_min[1]],
        [plot_max[0], plot_min[1]],
        [plot_max[0], plot_max[1]],
        [plot_min[0], plot_max[1]],
    ])
    merc_corners = local_to_merc(local_corners)

    xmin, ymin = merc_corners.min(axis=0)
    xmax, ymax = merc_corners.max(axis=0)

    img, ext = cx.bounds2img(
        xmin,
        ymin,
        xmax,
        ymax,
        zoom=zoom,
        source=source,
        ll=False,
    )

    # contextily's returned extent is in Web Mercator and is snapped to tile
    # boundaries. Convert that actual returned extent back to local coordinates
    # instead of forcing it to the requested extent.
    ext_xmin, ext_xmax, ext_ymin, ext_ymax = ext
    ext_merc_corners = np.array([
        [ext_xmin, ext_ymin],
        [ext_xmax, ext_ymin],
        [ext_xmax, ext_ymax],
        [ext_xmin, ext_ymax],
    ])
    ext_local_corners = merc_to_local(ext_merc_corners)
    img_xmin, img_ymin = ext_local_corners.min(axis=0)
    img_xmax, img_ymax = ext_local_corners.max(axis=0)

    # Display the fetched image using local coordinates, not Mercator coordinates.
    ax.imshow(
        img,
        extent=[img_xmin, img_xmax, img_ymin, img_ymax],
        origin="upper",
        alpha=alpha,
        zorder=-10,
    )
    ax.set_xlim(plot_min[0], plot_max[0])
    ax.set_ylim(plot_min[1], plot_max[1])
    ax.set_aspect("equal", adjustable="box")

    return ax

def plot_valid_mask_on_reference_spectrogram(
    signal_ref,
    fs,
    valid_mask,
    band_ranges_hz,
    mask_times,
    nperseg=1024,
    noverlap=512,
    nfft=1024,
    alpha=0.35,
    fmin=0.0,
    fmax=None,
    db_floor=80.0,
    cmap="gray_r",
    ref_db=None,
    fixed_clim=True,
):

    f, t, S = spectrogram(
        signal_ref,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        mode="magnitude",
    )

    keep_f = f >= float(fmin)
    if fmax is not None:
        keep_f &= f <= fmax
    f = f[keep_f]
    S = S[keep_f, :]
    if f.size == 0:
        raise ValueError("No spectrogram bins remain between fmin and fmax")

    nF, _ = S.shape
    nB, nTm = valid_mask.shape

    mask_full = np.zeros((nF, nTm), dtype=float)

    for b, (f_low, f_high) in enumerate(band_ranges_hz[:nB]):
        freq_idx = (f >= f_low) & (f < f_high)
        mask_full[freq_idx, :] = valid_mask[b][None, :]

    fig, ax = plot_reference_spectrogram(
        signals=signal_ref[None, :],
        fs=fs,
        mic_idx=0,
        t_start=0.0,
        t_end=len(signal_ref) / fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        title="Valid mask for pair (0,1) over reference spectrogram",
        fmin=fmin,
        fmax=fmax,
        ref_db=ref_db,
        db_floor=db_floor,
        cmap=cmap,
        fixed_clim=fixed_clim,
    )

    overlay = np.ma.masked_where(mask_full <= 0, mask_full)

    ax.imshow(
        overlay,
        origin="lower",
        aspect="auto",
        extent=[mask_times[0], mask_times[-1], f[0], f[-1]],
        alpha=alpha,
        cmap="autumn",
        interpolation="nearest",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Valid mask over reference spectrogram")
    fig.tight_layout()

    return fig, ax

# =============================================================================
# Helper functions for GPS <-> local coordinate conversion and GPX export
# =============================================================================

def fit_local_to_gps_affine(local_xy, gps_latlon):
    """
    Fit a local affine transform from centered local plot coordinates to GPS.

    This avoids depending on the internal structure of map_info. It uses the
    known microphone positions as control points:
        local micPos[x,y] -> GPS micGPS[lat,lon]

    For the small field scale of this experiment, an affine lat/lon mapping is
    accurate enough for exporting detections to GIS.
    """
    local_xy = np.asarray(local_xy, dtype=float)
    gps_latlon = np.asarray(gps_latlon, dtype=float)

    if local_xy.ndim != 2 or local_xy.shape[1] != 2:
        raise ValueError("local_xy must have shape (N, 2)")
    if gps_latlon.ndim != 2 or gps_latlon.shape[1] != 2:
        raise ValueError("gps_latlon must have shape (N, 2), ordered as [lat, lon]")
    if local_xy.shape[0] != gps_latlon.shape[0]:
        raise ValueError("local_xy and gps_latlon must contain the same number of points")
    if local_xy.shape[0] < 3:
        raise ValueError("At least 3 control points are needed for a stable affine GPS transform")

    A = np.column_stack([local_xy[:, 0], local_xy[:, 1], np.ones(local_xy.shape[0])])
    coef_lat, *_ = np.linalg.lstsq(A, gps_latlon[:, 0], rcond=None)
    coef_lon, *_ = np.linalg.lstsq(A, gps_latlon[:, 1], rcond=None)

    return coef_lat, coef_lon


def local_xy_to_gps(local_xy, gps_affine):
    """Convert one or more local [x,y] positions to [lat, lon]."""
    local_xy = np.asarray(local_xy, dtype=float)
    was_1d = local_xy.ndim == 1
    if was_1d:
        local_xy = local_xy[None, :]

    coef_lat, coef_lon = gps_affine
    A = np.column_stack([local_xy[:, 0], local_xy[:, 1], np.ones(local_xy.shape[0])])
    lat = A @ coef_lat
    lon = A @ coef_lon
    out = np.column_stack([lat, lon])

    return out[0] if was_1d else out


def write_detections_gpx(rows, gpx_path):
    """Write localized detections as GPX waypoints for QGIS."""
    gpx_path = Path(gpx_path)
    gpx_path.parent.mkdir(parents=True, exist_ok=True)

    with gpx_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gpx version="1.1" creator="TDOA localization" xmlns="http://www.topografix.com/GPX/1/1">\n')

        for row in rows:
            if not row.get("localization_success", False):
                continue
            lat = row.get("latitude")
            lon = row.get("longitude")
            if lat is None or lon is None or np.isnan(lat) or np.isnan(lon):
                continue

            name = f"ann_{int(row['annotation_id']):03d}_src_{int(row['source_idx'])}"
            desc = (
                f"annotation_id={row['annotation_id']}; "
                f"start_s={row['start_s']}; end_s={row['end_s']}; "
                f"x_m={row['x_m']}; y_m={row['y_m']}; "
                f"n_constraints={row['n_constraints']}"
            )
            f.write(f'  <wpt lat="{float(lat):.8f}" lon="{float(lon):.8f}">\n')
            f.write(f"    <name>{name}</name>\n")
            f.write(f"    <desc>{desc}</desc>\n")
            f.write("  </wpt>\n")

        f.write("</gpx>\n")
