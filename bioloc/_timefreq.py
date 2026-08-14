"""Small core time-frequency helpers without plotting/geo dependencies."""

from __future__ import annotations

import numpy as np


def attach_timefreq_info_to_peaks(
    tau_obs,
    bands,
    frames,
    peak_taus,
    tol_tau,
    n_bands=None,
    n_frames=None,
):
    tau_obs = np.asarray(tau_obs)
    bands = np.asarray(bands)
    frames = np.asarray(frames)
    output = []
    for peak_index, delay in enumerate(np.asarray(peak_taus)):
        indices = np.flatnonzero(np.abs(tau_obs - delay) <= tol_tau)
        peak_bands = bands[indices].copy()
        peak_frames = frames[indices].copy()
        unique, counts = np.unique(peak_bands, return_counts=True)
        raw = {
            "peak_idx": peak_index,
            "tau": float(delay),
            "bands": peak_bands,
            "band_counts": {
                int(band): int(count) for band, count in zip(unique, counts)
            },
            "frames": peak_frames,
            "obs_idx": indices,
            "n_support": int(indices.size),
            "tf_cells": (
                np.stack([peak_bands, peak_frames], axis=1)
                if indices.size
                else np.zeros((0, 2), dtype=int)
            ),
        }
        if n_bands is not None and n_frames is not None:
            support = np.zeros((n_bands, n_frames), dtype=np.float32)
            for band, frame in raw["tf_cells"]:
                if 0 <= band < n_bands and 0 <= frame < n_frames:
                    support[band, frame] += 1.0
            raw["tf_support_map"] = support
        output.append(raw)
    return output


def hz_seconds_to_tf_size(freq_hz, time_sec, band_ranges_hz, hop, fs):
    if float(freq_hz) <= 0:
        frequency_cells = 0
    else:
        centers = [0.5 * (float(lo) + float(hi)) for lo, hi in band_ranges_hz]
        widths = [max(float(hi) - float(lo), 1e-12) for lo, hi in band_ranges_hz]
        if len(centers) >= 2:
            step = float(np.median(np.diff(np.sort(centers))))
        elif widths:
            step = widths[0]
        else:
            step = max(float(freq_hz), 1.0)
        frequency_cells = max(1, int(np.ceil(float(freq_hz) / max(step, 1e-12))))
    if float(time_sec) <= 0:
        time_cells = 0
    else:
        time_cells = max(
            1,
            int(np.ceil(float(time_sec) / max(float(hop) / float(fs), 1e-12))),
        )
    return frequency_cells, time_cells


def hz_seconds_to_stft_sigma(freq_hz, time_sec, fs, nfft, hop):
    frequency_sigma = max(0.0, float(freq_hz)) / (
        3.0 * (float(fs) / float(nfft))
    )
    time_sigma = max(0.0, float(time_sec)) / (
        3.0 * (float(hop) / float(fs))
    )
    return frequency_sigma, time_sigma
