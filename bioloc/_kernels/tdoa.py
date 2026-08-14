import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.fft import next_fast_len, rfft, irfft
from scipy.ndimage import gaussian_filter1d, binary_opening, binary_closing, binary_dilation
from scipy.signal import butter, sosfiltfilt, stft, istft
from scipy.signal.windows import hann, tukey
"""
Convention
----------------------
All TDOA / delay variables (tau) in this module are expressed in samples,
unless explicitly stated otherwise.


Pipeline:
  1) Extract local sub-band TDOA observations via GCC argmax.
  2) Aggregate those observations into a smoothed histogram on a tau grid.
  3) Apply matching pursuit on the histogram to obtain peak_taus (samples).
"""


def compute_frame_starts(n_samples, frame_size, hop, force_last_frame=False):
    """
    Compute no-padding analysis frame starts.

    Normal frames start at 0, hop, 2*hop, ... and must fully fit inside the
    signal. If force_last_frame=True, also include the last possible full frame
    starting at n_samples - frame_size when it is not already on the hop grid.
    """
    n_samples = int(n_samples)
    frame_size = int(frame_size)
    hop = int(hop)

    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    if hop <= 0:
        raise ValueError("hop must be positive")
    if n_samples < frame_size:
        return np.zeros(0, dtype=int)

    starts = list(range(0, n_samples - frame_size + 1, hop))

    if force_last_frame:
        last_start = n_samples - frame_size
        if len(starts) == 0 or starts[-1] != last_start:
            starts.append(last_start)

    return np.asarray(starts, dtype=int)


# ============================================================
# Frequency band construction
# ============================================================

def build_frequency_bands(fs, nfft, fmax, omega_size_hz, overlap_hz, fmin=0, ):
    """
    Build overlapping frequency-band index sets for sub-band processing.

    Parameters
    ----------
    fs : float
        Sampling rate.
    nfft : int
        FFT size (defines frequency bin spacing df = fs/nfft).
    fmax : float
        Maximum frequency considered [Hz] (upper cutoff).
    omega_size_hz : float
        Sub-band width [Hz].
    overlap_hz : float
        Sub-band overlap [Hz].

    Returns
    -------
    Omega_list : list of 1D ndarrays
        List of arrays of FFT-bin indices, one per sub-band.
    band_ranges_hz : list of tuples
        [(f_lo, f_hi), ...] frequency limits for each band.
    """
    F = nfft // 2 + 1
    df = fs / float(nfft)

    omega_bins = int(round(omega_size_hz / df))
    overlap_bins = int(round(overlap_hz / df))
    step_bins = omega_bins - overlap_bins
    if step_bins <= 0:
        raise ValueError("overlap_hz too large: step_bins <= 0.")

    min_bin = int(np.floor(fmin / df))

    max_bin = int(np.floor(fmax / df))
    max_bin = min(max_bin, F - 1)

    Omega_list = []
    start_bin = min_bin
    while start_bin <= max_bin:
        end_bin = min(start_bin + omega_bins, max_bin + 1)
        if end_bin > start_bin:
            Omega_list.append(np.arange(start_bin, end_bin, dtype=np.int32))
        start_bin += step_bins

    freqs = np.arange(F) * df
    band_ranges_hz = []

    for Omega_idx in Omega_list:
        if len(Omega_idx) == 0:
            band_ranges_hz.append((0.0, 0.0))
            continue

        f_lo = float(freqs[int(np.min(Omega_idx))])
        f_hi = float(freqs[int(np.max(Omega_idx))])
        band_ranges_hz.append((f_lo, f_hi))

    return Omega_list, band_ranges_hz

# ============================================================
# Histogram smoothing
# ============================================================

def smooth_tdoa_histogram_gaussian_fast(tau_obs, L, t_min, t_max, sigma_tau):
    """
    Smooth a set of discrete TDOA observations using Gaussian kernel evaluation
    on a fixed tau grid of length L.

    Notes
    -----
    This computes:
        y[k] = sum_n exp(-0.5 * ((bin_center[k] - tau_obs[n]) / sigma_tau)^2)

    Parameters
    ----------
    tau_obs : (N,) array-like
        TDOA observations in samples (typically integer lags from GCC).
    L : int
        Number of histogram bins on [t_min, t_max).
    t_min, t_max : float
        Tau range covered by the grid, in samples.
    sigma_tau : float
        Gaussian standard deviation in samples.

    Returns
    -------
    y : (L,) ndarray
        Smoothed histogram values.
    bin_centers : (L,) ndarray
        Tau grid points (bin centers) in samples.
    """
    tau_obs = np.asarray(tau_obs, dtype=float)
    bin_centers = np.linspace(t_min, t_max, L, endpoint=False)

    diff = bin_centers[:, None] - tau_obs[None, :]
    y = np.exp(-0.5 * (diff / sigma_tau) ** 2).sum(axis=1)

    return y, bin_centers


# ============================================================
# Matching pursuit
# ============================================================

def _blackman_atom(L, Q):
    """
    Build a Blackman-window atom of length L centered at the histogram midpoint.

    Parameters
    ----------
    L : int
        Histogram length.
    Q : int
        Atom width in bins. Forced to be odd for symmetry.
    """
    if Q % 2 == 0:
        Q += 1
    w = np.blackman(Q)

    atom = np.zeros(L, dtype=np.float32)
    mid = L // 2
    half = Q // 2
    start = max(0, mid - half)
    end = min(L, mid + half + 1)
    atom[start:end] = w
    return atom


def _shift_atom_linear(atom, shift):
    """
    Shift an atom by an integer amount without wrap-around (linear shift).

    Parameters
    ----------
    atom : (L,) ndarray
    shift : int
        Positive shift moves content to the right.

    Returns
    -------
    out : (L,) ndarray
        Shifted atom with zero padding.
    """
    L = atom.shape[0]
    out = np.zeros_like(atom)
    s = int(shift)

    if s >= 0:
        if s < L:
            out[s:] = atom[: L - s]
    else:
        s = -s
        if s < L:
            out[: L - s] = atom[s:]
    return out


def _energy(vec):
    """Squared L2 norm."""
    return float(np.dot(vec, vec))


# ============================================================
# Matching pursuit peak extraction
# ============================================================

# def matching_pursuit_tdoa_maxSources(
#         y,
#         L,
#         QN,
#         QW,
#         maxSources,
#         t_min,
#         t_max,
#         u_min=None,
#         verbose=False,
# ):
#     """
#     Matching pursuit over a linear 1D histogram defined on a tau grid.
#
#     Parameters
#     ----------
#     y : (L,) ndarray
#         Smoothed histogram values (nonnegative).
#     L : int
#         Histogram length (kept for compatibility; must match len(y)).
#     QN, QW : int
#         Atom widths in bins:
#           - QN: narrow atom for correlation / peak search
#           - QW: wide atom for subtraction (suppresses neighborhood after selecting peak)
#     maxSources : int
#         Maximum number of peaks to extract (iterations).
#     t_min, t_max : float
#         Tau range (in samples) of the histogram grid.
#     u_min : float or None
#         Kept only for API compatibility.
#         It is not used in this function.
#     verbose : bool
#         If True, returns debug traces and prints iteration info.
#
#     Returns
#     -------
#     peak_taus : (P,) ndarray
#         Estimated peak locations in samples (float grid locations).
#     betas : (P,) ndarray
#         MP coefficients.
#     residual : (L,) ndarray
#         Residual histogram after subtraction.
#     debug : dict or None
#         Debug traces if verbose else None.
#     """
#     y = np.asarray(y, dtype=float).copy()
#     if y.ndim != 1 or len(y) != L:
#         raise ValueError("y must be 1D of length L")
#
#     cN = _blackman_atom(L, QN)
#     cW = _blackman_atom(L, QW)
#     E_CN = _energy(cN)
#
#     span = float(t_max - t_min)
#     _ = u_min  # API compatibility only; no minimum-separation masking in this variant
#
#     peak_taus, betas = [], []
#     residual = y.copy()
#
#     traces = None
#     if verbose:
#         traces = {"a_list": [], "i_star": [], "beta": [], "y_snapshots": [residual.copy()]}
#
#     for j in range(int(maxSources)):
#         # Padding reduces edge bias when correlating near the boundaries
#         pad = int(QW)
#         res_padded = np.pad(residual, pad, mode="reflect")
#         a_full = np.correlate(res_padded, cN, mode="same")
#         a = a_full[pad:pad + L]
#
#         i_star = int(np.argmax(a))
#
#         # i_star -> tau on the same grid used in smoothing
#         tau_hat = t_min + i_star * span / L
#         beta_j = float(a[i_star] / (E_CN + 1e-12))
#
#         # Subtract a wide atom centered at i_star to avoid selecting the same peak again
#         cW_shift = _shift_atom_linear(cW, i_star - L // 2)
#         residual = np.maximum(residual - cW_shift, 0.0)
#
#         peak_taus.append(tau_hat)
#         betas.append(beta_j)
#
#         if verbose:
#             print(f"[j={j}] i*={i_star}, tau_hat={tau_hat:.6f} samp, beta={beta_j:.4f}")
#             traces["a_list"].append(a.copy())
#             traces["i_star"].append(i_star)
#             traces["beta"].append(beta_j)
#             traces["y_snapshots"].append(residual.copy())
#
#     return np.asarray(peak_taus), np.asarray(betas), residual, traces


def remove_baseline(y, sigma_base=200):
    """
    y: (L,) nonnegative histogram (often normalized)
    sigma_base: large -> captures only slow floor variations
    """
    y = np.asarray(y, float)
    base = gaussian_filter1d(y, sigma=sigma_base, mode="reflect")
    r = y - base
    r[r < 0] = 0.0
    return r, base


def matching_pursuit_tdoa(
        y,  # length-L TDOA histogram (smoothed), nonnegative
        L,  # number of bins (len(y))
        QN, QW,  # odd widths (in bins) for narrow & wide atoms
        t_min, t_max,  # TDOA range covered by the histogram
        verbose=False
):
    """
    Matching pursuit on a linear TDOA histogram over [t_min, t_max], using the
    same atom construction, candidate selection, and residual update logic as
    `matching_pursuit_tdoa_maxSources`, but with an automatic stopping rule.

    Parameters
    ----------
    y : array-like, shape (L,)
        Smoothed TDOA histogram.
    L : int
        Number of bins.
    QN, QW : int
        Atom widths (in bins) for narrow and wide atoms.
    u_min : float
        Kept only for API compatibility.
        It is not used in this function.
    t_min, t_max : float
        Min and max TDOA values represented by the histogram.
    verbose : bool
        Print debug info.

    Returns
    -------
    tdoas_hat : (P,) ndarray
        Estimated TDOA peaks (same units as t_min/t_max).
    betas : (P,) ndarray
        MP coefficients for each detected peak.
    residual : (L,) ndarray
        Final residual histogram.
    debug : dict or None
        Internal traces if verbose, else None.
    """
    y = np.asarray(y, dtype=float).copy()
    if y.ndim != 1 or len(y) != L:
        raise ValueError("y must be 1D of length L")

    # y, _ = remove_baseline(y, sigma_base=QN//2)

    cN = _blackman_atom(L, QN)
    cW = _blackman_atom(L, QW)
    E_CN = _energy(cN)
    qn_effective = int(QN) + (int(QN) % 2 == 0)
    cN_window = np.blackman(qn_effective)

    span = float(t_max - t_min)

    tdoas_hat, betas = [], []
    residual = y.copy()

    traces = {"a_list": [], "i_star": [], "beta": [], "y_snapshots": [residual.copy()]}

    def rms(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

    j = 1
    a1_max = None
    while True:
        # Same correlation logic as matching_pursuit_tdoa_maxSources
        pad = int(QW)
        res_padded = np.pad(residual, pad, mode="reflect")
        # cN is zero outside its short Blackman support. Correlating with only
        # that support is mathematically equivalent and avoids O(L^2) work.
        a_full = np.correlate(res_padded, cN_window, mode="same")
        a = a_full[pad:pad + L]
        traces["a_list"].append(a.copy())

        if not np.any(np.isfinite(a)):
            break

        i_star = int(np.argmax(a))
        traces["i_star"].append(i_star)

        tdoa_hat = t_min + i_star * span / L
        beta_j = float(a[i_star] / (E_CN + 1e-12))
        traces["beta"].append(beta_j)

        if verbose:
            print(f"[j={j}] i*={i_star}, TDOA={tdoa_hat:.6f}, a[i*]={a[i_star]:.4f}, beta={beta_j:.4f}")

        # Automatic stopping rule: compare local peak RMS against residual background.
        half_w = max(1, QW // 2)
        p0 = max(0, i_star - half_w)
        p1 = min(L, i_star + half_w + 1)

        peak_vals = residual[p0:p1]

        noise_mask = np.ones(L, dtype=bool)
        noise_mask[p0:p1] = False
        noise_vals = residual[noise_mask]

        peak_rms = rms(peak_vals)
        noise_rms = rms(noise_vals) if noise_vals.size > 0 else 1e-12
        snr_rms = peak_rms / (noise_rms + 1e-12)

        if verbose:
            print(f"  peak_rms={peak_rms:.4g}, noise_rms={noise_rms:.4g}, ratio={snr_rms:.2f}")

        if j == 1:
            a1_max = a[i_star]
            if snr_rms < 1.5:
                break
        #
        # if snr_rms < 3.0 and a[i_star] / (a1_max + 1e-12) < 0.2:
        #     break
        relative_height = a[i_star] / (a1_max + 1e-12)
        if j > 1:
            if relative_height < 0.1 or snr_rms < 1.5:
                break
        # if a[i_star] / (a1_max + 1e-12) < 0.2:
        #     break
        # Same residual update logic as matching_pursuit_tdoa_maxSources
        cW_shift = _shift_atom_linear(cW, i_star - L // 2)
        residual = np.maximum(residual - cW_shift, 0.0)

        traces["y_snapshots"].append(residual.copy())

        tdoas_hat.append(tdoa_hat)
        betas.append(beta_j)
        j += 1

    debug = traces if verbose else None
    return np.asarray(tdoas_hat), np.asarray(betas), residual, debug


# ============================================================
# Pairwise GCC
# ============================================================

def gcc_pairs(
        signals_dft: np.ndarray,
        pair_i: np.ndarray,
        pair_j: np.ndarray,
        *,
        tdoa_max: int = None,
        abs_val: bool = False,
        ifft: bool = True,
        n_dft_bins: int = None,
        beta: float = 1.0,
        weighting: str = "beta",
        coh_exponent: float = 1.0,
        eps: float = 1e-10,
        normalize_lag_overlap: bool = True,
        frame_len: int = None,
):
    """
    Pairwise GCC between microphone pairs (pair_i[k], pair_j[k]).

    Notes
    -----
    - If `ifft=True`, this returns time-domain GCC curves indexed by integer lags (samples).
    - The lag axis is centered so that lag=0 is at the center index before cropping.
    - If `tdoa_max` is provided, the output is cropped to lags:
        [-tdoa_max, ..., tdoa_max-1]  (length 2*tdoa_max)

    Parameters
    ----------
    signals_dft : (M, F) complex ndarray
        rFFT-domain signals per microphone.
    pair_i, pair_j : (P,) int arrays
        Microphone pair indices, P = number of pairs.
    tdoa_max : int or None
        If provided and ifft=True, crop to lags in [-tdoa_max, tdoa_max).
    abs_val : bool
        If True, take abs() of time-domain GCC.
    ifft : bool
        If True, return time-domain GCC via irfft.
        If False, return weighted cross-spectrum (frequency domain).
    n_dft_bins : int or None
        IFFT length. If None, inferred from rFFT length F as n = 2*(F-1).
    weighting : {"beta","phat","gcc","roth","scot","coherence"}
        Weighting scheme for GCC.
    normalize_lag_overlap : bool
        If True and frame_len is provided, divide by (frame_len - |lag|) to account for
        reduced overlap near large lags.
    frame_len : int or None
        Frame length (samples) used for overlap normalization.

    Returns
    -------
    gcc_out : ndarray
        If ifft=True: shape (P, Tlag), real.
        If ifft=False: shape (P, F), complex.
    lags : (Tlag,) ndarray
        Integer lag axis in samples. If cropped, lags are in [-tdoa_max, ..., tdoa_max-1].
    """
    X = signals_dft
    if X.ndim != 2:
        raise ValueError("signals_dft must have shape (M, F)")

    M, F = X.shape
    pair_i = np.asarray(pair_i, dtype=int)
    pair_j = np.asarray(pair_j, dtype=int)

    # Cross-spectrum for each pair: (P, F)
    cross = X[pair_i] * np.conj(X[pair_j])

    weighting = weighting.lower()

    # Weighting. This was used only for testing, all results in the paper are with phat weighting (beta=1).
    if weighting == "beta":
        denom = (np.abs(cross) ** beta) + eps
        G = cross / denom
    elif weighting == "phat":
        denom = np.abs(cross) + eps
        G = cross / denom
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    if not ifft:
        n = (F - 1) * 2
        max_shift = n // 2
        lags = np.arange(-max_shift, max_shift, dtype=int)
        return G, lags

    if n_dft_bins is None:
        n_dft_bins = (F - 1) * 2

    g = np.fft.irfft(G, n=n_dft_bins, axis=-1)

    if abs_val:
        g = np.abs(g)

    # Shift so lag=0 is at the center
    half = g.shape[-1] // 2
    g = np.concatenate((g[..., half:], g[..., :half]), axis=-1)

    # Full lag axis
    n = g.shape[-1]
    max_shift = n // 2
    lags = np.arange(-max_shift, max_shift, dtype=int)

    # Crop if requested
    if tdoa_max is not None:
        tdoa_max = int(tdoa_max)
        center = n // 2
        start = center - tdoa_max
        end = center + tdoa_max
        g = g[..., start:end]  # (P, 2*tdoa_max)
        lags = np.arange(-tdoa_max, tdoa_max, dtype=int)

        if normalize_lag_overlap and frame_len is not None:
            overlaps = frame_len - np.abs(lags)
            overlaps = np.maximum(overlaps, 1)
            g = g / overlaps[None, :]

    return g, lags


# ============================================================
# Sub-band GCC observation extraction
# ============================================================

def extract_subband_tdoas_gccpairs(
        signalsSTFT,
        W,
        iu,
        ju,
        maxTDOA,
        nfft,
        frameSize,
        B=None,
        beta=0.8,
        weighting="beta",
        normalize_lag_overlap=True,
        abs_val=False,
        valid_mask=None,  # (P, B, T) boolean mask
):
    """
    Extract local TDOA observations (tau) per pair from sub-band GCC maxima.

    For each time frame and each sub-band, compute GCC for all requested pairs and
    store the argmax lag.

    Parameters
    ----------
    signalsSTFT : (M, F, T) complex ndarray
        STFT (or any frequency-domain frame representation) per microphone.
    W : (n_bands, F) ndarray
        Band masks in frequency (one row per sub-band).
    iu, ju : (P,) int arrays
        Pair list: p-th pair is (iu[p], ju[p]).
    maxTDOA : int
        Maximum TDOA lag in samples (GCC is cropped to [-maxTDOA, ..., maxTDOA-1]).
    nfft : int
        IFFT length used inside GCC (passed to gcc_pairs as n_dft_bins).
    frameSize : int
        Frame length in samples (used for lag-overlap normalization).
    B : int or None
        If provided, process only the first B frames.
    beta, weighting, normalize_lag_overlap, abs_val
        Parameters forwarded to gcc_pairs.

    Returns
    -------
    obs_tau : list of length P
        obs_tau[p] is a list of raw lag observations (integer samples) for pair p.
    obs_band : list of length P
        obs_band[p] is the aligned band index for each observation in obs_tau[p].
    obs_frame : list of length P
        obs_frame[p] is the aligned frame index for each observation in obs_tau[p].
    lags : (2*maxTDOA,) ndarray
        Lag axis (samples) returned by gcc_pairs.
    """
    M, F, T = signalsSTFT.shape
    P = len(iu)
    n_bands = W.shape[0]

    T_use = T if B is None else min(int(B), T)

    obs_tau = [[] for _ in range(P)]
    obs_band = [[] for _ in range(P)]
    obs_frame = [[] for _ in range(P)]

    lags_last = None

    for t in range(T_use):
        Xt = signalsSTFT[:, :, t].astype(np.complex64)

        for b in range(n_bands):
            # Select only active pairs for this (band, frame)
            if valid_mask is None:
                active_idx = np.arange(P)
            else:
                active_idx = np.where(valid_mask[:, b, t])[0]

            if active_idx.size == 0:
                continue

            Xb = Xt * W[b][None, :]

            gcc, lags = gcc_pairs(
                Xb,
                iu[active_idx],
                ju[active_idx],
                tdoa_max=maxTDOA,
                beta=beta,
                ifft=True,
                n_dft_bins=nfft,
                weighting=weighting,
                frame_len=frameSize,
                normalize_lag_overlap=normalize_lag_overlap,
                abs_val=abs_val,
            )

            kmax = np.argmax(gcc, axis=-1)
            taus = lags[kmax]

            for local_idx, p in enumerate(active_idx):
                obs_tau[p].append(int(taus[local_idx]))
                obs_band[p].append(int(b))
                obs_frame[p].append(int(t))

            lags_last = lags

    if lags_last is None:
        lags_last = np.arange(-maxTDOA, maxTDOA + 1)

    return obs_tau, obs_band, obs_frame, lags_last


def build_pair_band_energy_mask(
        signals,
        fs,
        Omega_list,
        iu,
        ju,
        frameSize,
        hop,
        *,
        nfft=None,
        pair_mode="min",
        threshold_mode="percentile",
        percentile=50.0,
        rel_factor=1.5,
        eps=1e-12,
        force_last_frame=False,
        return_energies=False,
):
    """
    Build a boolean mask indicating which (pair, band, frame) zones have enough energy.

    This version computes the band energies directly from the time-domain signals
    using rectangular frequency bands from Omega_list. It does not require the
    caller to precompute an STFT or a W matrix.

    Parameters
    ----------
    signals : (M, N) ndarray
        Time-domain microphone signals.
    fs : float
        Sampling rate. Kept for API clarity; the bin indices in Omega_list are
        already defined by nfft.
    Omega_list : list of 1D ndarrays
        Frequency-bin indices for each rectangular sub-band, defined on the
        rFFT grid of length nfft.
    iu, ju : (P,) int ndarray
        Pair index arrays. Pair p is (iu[p], ju[p]).
    frameSize : int
        Analysis frame length in samples.
    hop : int
        Hop size in samples.
    nfft : int or None
        FFT size used to define Omega_list. If None, uses frameSize.
    pair_mode : {"min", "geom_mean", "mean", "max"}
        How to combine the two microphone energies of a pair:
          - "min"       : conservative, requires both mics to have energy
          - "geom_mean" : softer than min
          - "mean"      : average energy
          - "max"       : permissive, either mic can have energy
    threshold_mode : {"percentile", "median_rel"}
        How to threshold pair-band energies across time:
          - "percentile": keep frames above the given percentile for each (pair, band)
          - "median_rel": keep frames above rel_factor * median for each (pair, band)
    percentile : float
        Percentile used if threshold_mode="percentile".
    rel_factor : float
        Factor used if threshold_mode="median_rel".
    eps : float
        Small constant for numerical stability.
    force_last_frame : bool
        If True, include the last possible full frame using compute_frame_starts.
    return_energies : bool
        If True, also return:
          - band_energy_mic: (M, B, T)
          - pair_energy: (P, B, T)
          - thresholds: (P, B)

    Returns
    -------
    valid_mask : (P, B, T) bool ndarray
        True where the zone is energetic enough.
    band_energy_mic : (M, B, T) ndarray, optional
    pair_energy : (P, B, T) ndarray, optional
    thresholds : (P, B) ndarray, optional
    """
    _ = fs  # Omega_list already encodes the frequency bins for the chosen nfft.

    signals = np.asarray(signals, dtype=np.float32)
    iu = np.asarray(iu, dtype=int)
    ju = np.asarray(ju, dtype=int)

    if signals.ndim != 2:
        raise ValueError("signals must have shape (M, N)")
    if iu.shape != ju.shape:
        raise ValueError("iu and ju must have the same shape")

    if nfft is None:
        nfft = int(frameSize)
    else:
        nfft = int(nfft)

    frameSize = int(frameSize)
    hop = int(hop)

    M, N = signals.shape
    P = len(iu)
    B = len(Omega_list)

    frame_starts = compute_frame_starts(
        N,
        frameSize,
        hop,
        force_last_frame=force_last_frame,
    )
    T = len(frame_starts)

    band_energy_mic = np.zeros((M, B, T), dtype=np.float32)

    for t, s in enumerate(frame_starts):
        s = int(s)
        e = s + frameSize
        frame = signals[:, s:e]

        if frame.shape[1] != frameSize:
            continue

        X = np.fft.rfft(frame, n=nfft, axis=1)
        mag2 = np.abs(X).astype(np.float32) ** 2

        for b, omega_idx in enumerate(Omega_list):
            omega_idx = np.asarray(omega_idx, dtype=int)
            omega_idx = omega_idx[(omega_idx >= 0) & (omega_idx < mag2.shape[1])]
            if omega_idx.size == 0:
                continue

            # Rectangular band energy. No Hann/W weighting is applied here.
            band_energy_mic[:, b, t] = np.sum(mag2[:, omega_idx], axis=1)

    Ei = band_energy_mic[iu]  # (P,B,T)
    Ej = band_energy_mic[ju]  # (P,B,T)

    if pair_mode == "min":
        pair_energy = np.minimum(Ei, Ej)
    elif pair_mode == "geom_mean":
        pair_energy = np.sqrt(np.maximum(Ei * Ej, 0.0))
    elif pair_mode == "mean":
        pair_energy = 0.5 * (Ei + Ej)
    elif pair_mode == "max":
        pair_energy = np.maximum(Ei, Ej)
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")

    if threshold_mode == "percentile":
        thresholds = np.percentile(pair_energy, percentile, axis=2)  # (P,B)
    elif threshold_mode == "median_rel":
        thresholds = rel_factor * np.median(pair_energy, axis=2)
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    valid_mask = pair_energy >= (thresholds[:, :, None] + eps)

    if return_energies:
        return valid_mask, band_energy_mic, pair_energy, thresholds
    return valid_mask




def build_fullband_rfft_mask(freqs_full, f_lo, f_hi, taper_hz=0.0):
    """
    Build a real-valued band mask on a full-signal rFFT frequency axis.
    """
    mask = np.zeros_like(freqs_full, dtype=np.float32)

    if f_hi <= f_lo:
        return mask

    if taper_hz <= 0:
        mask[(freqs_full >= f_lo) & (freqs_full <= f_hi)] = 1.0
        return mask

    # Cosine ramps
    left0 = max(0.0, f_lo - taper_hz)
    left1 = f_lo
    right0 = f_hi
    right1 = f_hi + taper_hz

    passband = (freqs_full >= left1) & (freqs_full <= right0)
    mask[passband] = 1.0

    left = (freqs_full >= left0) & (freqs_full < left1)
    if np.any(left):
        x = (freqs_full[left] - left0) / max(left1 - left0, 1e-12)
        mask[left] = 0.5 - 0.5 * np.cos(np.pi * x)

    right = (freqs_full > right0) & (freqs_full <= right1)
    if np.any(right):
        x = (freqs_full[right] - right0) / max(right1 - right0, 1e-12)
        mask[right] = 0.5 + 0.5 * np.cos(np.pi * x)

    return mask


def design_full_signal_band_butterworth(fs, f_lo, f_hi, order=4):
    """Design the SOS filter used for one full-signal frequency band."""
    nyquist = 0.5 * float(fs)
    f_lo = max(0.0, float(f_lo))
    f_hi = min(float(f_hi), nyquist)

    if f_hi <= f_lo:
        return None
    if f_lo <= 0.0 and f_hi >= nyquist:
        return np.empty((0, 6), dtype=float)
    if f_lo <= 0.0:
        return butter(order, f_hi, btype="lowpass", fs=fs, output="sos")
    if f_hi >= nyquist:
        return butter(order, f_lo, btype="highpass", fs=fs, output="sos")
    return butter(
        order,
        [f_lo, f_hi],
        btype="bandpass",
        fs=fs,
        output="sos",
    )


def filter_full_signal_band_butterworth(
        signals,
        fs,
        f_lo,
        f_hi,
        order=4,
        sos=None,
):
    """
    Zero-phase Butterworth filtering for one frequency band.

    Bands touching DC or Nyquist are implemented as low-pass or high-pass
    filters, respectively. Interior bands use a band-pass filter.
    """
    if sos is None:
        sos = design_full_signal_band_butterworth(
            fs,
            f_lo,
            f_hi,
            order=order,
        )
    if sos is None:
        return None
    if np.asarray(sos).size == 0:
        return np.asarray(signals, dtype=np.float32).copy()

    return sosfiltfilt(sos, signals, axis=1).astype(np.float32)


def fast_xcorr_short_in_long_nonorm(
        x_ref,
        y_search,
        lag_min,
        maxlag,
        weights=None,
):
    """
    Sliding non-normalized cross-correlation with optional time weighting.

    Returns scores indexed by lag where lag means:
        lag = (start index of y window) - (start index of x window)

    ``weights`` are folded into x_ref, so this requires only the original
    correlation FFT. Unlike fast_xcorr_short_in_long, candidate-window
    energies are not normalized.

    IMPORTANT:
    This raw lag has the opposite sign of your pipeline convention
    tau_ij = tof_i - tof_j, so the caller should negate it.
    """
    x_ref = np.asarray(x_ref, dtype=np.float32)
    y_search = np.asarray(y_search, dtype=np.float32)

    L = len(x_ref)
    M = len(y_search)

    if M < L:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (L,):
            raise ValueError("weights must have the same length as x_ref")
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("weights must be non-negative and not all zero")
        x_ref = weights * x_ref

    # Reverse the weighted reference so convolution gives sliding correlation.
    xr = x_ref[::-1]

    nfft = 1
    while nfft < M + L - 1:
        nfft *= 2

    X = np.fft.rfft(xr, n=nfft)
    Y = np.fft.rfft(y_search, n=nfft)
    conv = np.fft.irfft(X * Y, n=nfft)

    # Valid windows: y[k:k+L], k = 0 ... M-L
    scores = conv[L-1 : L-1 + (M - L + 1)]

    lags = lag_min + np.arange(M - L + 1, dtype=int)

    valid = (lags >= -maxlag) & (lags <= maxlag)
    return scores[valid].astype(np.float32), lags[valid]

def fast_xcorr_short_in_long(
        x_ref,
        y_search,
        lag_min,
        maxlag,
        weights=None,
        eps=1e-12,
):
    """
    Sliding weighted normalized cross-correlation.

    Returns scores indexed by lag:
        lag = start(y_window) - start(x_ref)

    ``weights`` applies the same sample weighting to the reference and every
    candidate window. This is equivalent to windowing both signals by
    sqrt(weights), while retaining an efficient sliding implementation.
    """

    x_ref = np.asarray(x_ref, dtype=np.float32)
    y_search = np.asarray(y_search, dtype=np.float32)
    L = len(x_ref)
    M = len(y_search)
    if M < L:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)

    if weights is None:
        weights = np.ones(L, dtype=np.float32)
        rectangular = True
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (L,):
            raise ValueError("weights must have the same length as x_ref")
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("weights must be non-negative and not all zero")
        rectangular = np.all(weights == 1.0)

    # Weighted numerator: sum_n weights[n] * x[n] * y[k + n].
    xr = (weights * x_ref)[::-1]
    nfft = 1
    while nfft < M + L - 1:
        nfft <<= 1
    X = np.fft.rfft(xr, n=nfft)
    Y = np.fft.rfft(y_search, n=nfft)
    conv = np.fft.irfft(X * Y, n=nfft)
    scores = conv[L - 1: L - 1 + (M - L + 1)]

    x_norm = np.sqrt(np.sum(weights * x_ref ** 2))
    y2 = y_search ** 2
    if rectangular:
        csum = np.cumsum(y2)
        win_energy = csum[L - 1:].copy()
        if L < M:
            win_energy[1:] -= csum[:-L]
    else:
        W = np.fft.rfft(weights[::-1], n=nfft)
        Y2 = np.fft.rfft(y2, n=nfft)
        energy_conv = np.fft.irfft(W * Y2, n=nfft)
        win_energy = energy_conv[L - 1: L - 1 + (M - L + 1)]

    y_norm = np.sqrt(np.maximum(win_energy, 0.0))
    scores = scores / (x_norm * y_norm + eps)
    lags = lag_min + np.arange(M - L + 1, dtype=int)
    valid = (lags >= -maxlag) & (lags <= maxlag)
    return scores[valid].astype(np.float32), lags[valid]


def fast_gcc_short_in_long(
        x_ref,
        y_search,
        fs,
        f_lo,
        f_hi,
        lag_min,
        maxlag,
        taper_hz=0.0,
        alpha=1.0,
        weights=None,
        eps=1e-12,
):
    """
    GCC/partial-PHAT between a short reference and longer search signal.

    Returns scores indexed by lag where lag means:
        lag = (start index of y window) - (start index of x window)

    ``alpha=0`` gives raw GCC, ``alpha=1`` gives GCC-PHAT. ``weights`` are
    folded into x_ref only; this keeps GCC efficient but is not symmetric
    per-window weighting of y_search.
    """
    x_ref = np.asarray(x_ref, dtype=np.float32)
    y_search = np.asarray(y_search, dtype=np.float32)

    L = len(x_ref)
    M = len(y_search)
    if M < L:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (L,):
            raise ValueError("weights must have the same length as x_ref")
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("weights must be non-negative and not all zero")
        x_ref = weights * x_ref

    nfft = next_fast_len(L + M - 1)
    freqs_full = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = build_fullband_rfft_mask(
        freqs_full,
        f_lo,
        f_hi,
        taper_hz=taper_hz,
    )

    X = rfft(x_ref, n=nfft)
    Y = rfft(y_search, n=nfft)
    G = Y * np.conj(X)

    if alpha != 0.0:
        magnitude = np.abs(G)
        G = G / np.maximum(magnitude, eps) ** alpha

    G = G * mask
    gcc = irfft(G, n=nfft)

    n_positions = M - L + 1
    scores = gcc[:n_positions]

    lags = lag_min + np.arange(n_positions, dtype=int)
    valid = (lags >= -maxlag) & (lags <= maxlag)
    return scores[valid].astype(np.float32), lags[valid]


def extract_subband_tdoas_windowed_search(
        signals,
        fs,
        band_ranges_hz,
        iu,
        ju,
        pair_max_tdoa,
        frameSize,
        hop,
        *,
        B=None,
        valid_mask=None,  # (P, BANDS, T)
        taper_hz=0.0,
        filter_mode="fft_mask",
        filter_order=4,
        time_window="rectangular",
        time_window_alpha=0.25,
        force_last_frame=False,
        correlation_mode="norm_cc",
        gcc_alpha=1.0,
        band_filter_sos=None,
        workers=1,
):
    """
    Extract local TDOA observations using:
      - short analysis window on mic i
      - larger lag-search interval on mic j

    This decouples frameSize from the maximum pair delay.

    Parameters
    ----------
    signals : (M, N) ndarray
        Time-domain microphone signals.
    fs : float
    band_ranges_hz : list[(f_lo, f_hi)]
        Frequency ranges for each sub-band.
    iu, ju : (P,) int arrays
        Pair indices.
    pair_max_tdoa : (P,) int array
        Pair-dependent maximum feasible lag in samples.
    frameSize : int
        Short analysis window size.
    hop : int
        Hop size between frames.
    B : int or None
        Number of frames to use.
    valid_mask : (P, n_bands, T) bool ndarray or None
        Optional mask to skip low-energy zones.
    taper_hz : float
        Optional taper width when building full-signal band masks.
        Only used when filter_mode="fft_mask".
    filter_mode : {"fft_mask", "butterworth"}
        Sub-band frequency filtering method. "fft_mask" preserves the existing
        rectangular/tapered full-signal RFFT mask. "butterworth" applies a
        zero-phase IIR filter to avoid phase-delay bias.
    filter_order : int
        Butterworth order used when filter_mode="butterworth".
    time_window : {"rectangular", "hann", "tukey"}
        Time weighting used by the normalized correlation. Tukey is generally
        a good compromise between edge suppression and effective frame length.
    time_window_alpha : float
        Tukey taper fraction. Ignored by the other window types.
    force_last_frame : bool
        If True, include the final full frame starting at N - frameSize even if
        it is not aligned with the hop grid. This covers the end of the signal
        without padding or using a shorter frame.
    correlation_mode : {"cc", "norm_cc", "gcc"}
        Correlation score to use.
    gcc_alpha : float
        GCC spectral weighting exponent. 0 gives raw GCC, 1 gives GCC-PHAT.
    workers : int, default 1
        Number of frequency bands to process concurrently using threads.
        ``workers=1`` preserves the original serial implementation.

    Returns
    -------
    obs_tau : list of length P
    obs_band : list of length P
    obs_frame : list of length P
    lags : ndarray
        Global lag axis for downstream use.
    """
    signals = np.asarray(signals, dtype=np.float32)
    iu = np.asarray(iu, dtype=int)
    ju = np.asarray(ju, dtype=int)
    pair_max_tdoa = np.asarray(pair_max_tdoa, dtype=int)

    if signals.ndim != 2:
        raise ValueError("signals must have shape (M, N)")
    if iu.shape != ju.shape:
        raise ValueError("iu and ju must have the same shape")
    if pair_max_tdoa.shape[0] != iu.shape[0]:
        raise ValueError("pair_max_tdoa must have one entry per pair")
    if filter_mode not in {"fft_mask", "butterworth"}:
        raise ValueError(
            "filter_mode must be either 'fft_mask' or 'butterworth'"
        )
    if int(filter_order) != filter_order or filter_order <= 0:
        raise ValueError("filter_order must be a positive integer")
    if correlation_mode not in {"cc", "norm_cc", "gcc"}:
        raise ValueError(
            "correlation_mode must be 'cc', 'norm_cc', or 'gcc'"
        )
    gcc_alpha = float(gcc_alpha)
    if gcc_alpha < 0.0:
        raise ValueError("gcc_alpha must be non-negative")
    if time_window not in {"rectangular", "hann", "tukey"}:
        raise ValueError(
            "time_window must be 'rectangular', 'hann', or 'tukey'"
        )
    if not 0.0 <= float(time_window_alpha) <= 1.0:
        raise ValueError("time_window_alpha must be between 0 and 1")
    if band_filter_sos is not None:
        if filter_mode != "butterworth":
            raise ValueError(
                "band_filter_sos is only valid with filter_mode='butterworth'"
            )
        if len(band_filter_sos) != len(band_ranges_hz):
            raise ValueError(
                "band_filter_sos must contain one entry per frequency band"
            )
    if int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)

    M, N = signals.shape
    P = iu.size

    if N < frameSize:
        raise ValueError("frameSize is longer than the signals")

    frame_starts = compute_frame_starts(
        N,
        frameSize,
        hop,
        force_last_frame=force_last_frame,
    )

    n_frames_total = len(frame_starts)
    T_use = n_frames_total if B is None else min(int(B), n_frames_total)

    obs_tau = [[] for _ in range(P)]
    obs_band = [[] for _ in range(P)]
    obs_frame = [[] for _ in range(P)]

    if time_window == "rectangular":
        correlation_weights = None
    elif time_window == "hann":
        correlation_weights = hann(frameSize, sym=False).astype(np.float32)
    else:
        correlation_weights = tukey(
            frameSize,
            alpha=float(time_window_alpha),
            sym=False,
        ).astype(np.float32)

    if filter_mode == "fft_mask":
        X_full = np.fft.rfft(signals, axis=1)
        freqs_full = np.fft.rfftfreq(N, d=1.0 / fs)

    def process_band(b):
        f_lo, f_hi = band_ranges_hz[b]
        band_tau = [[] for _ in range(P)]
        band_frame = [[] for _ in range(P)]

        if valid_mask is not None and not np.any(valid_mask[:, b, :T_use]):
            return b, band_tau, band_frame

        if filter_mode == "fft_mask":
            mask = build_fullband_rfft_mask(
                freqs_full,
                f_lo,
                f_hi,
                taper_hz=taper_hz,
            )
            if not np.any(mask > 0):
                return b, band_tau, band_frame

            Xb = X_full * mask[None, :]
            xb = np.fft.irfft(Xb, n=N, axis=1).astype(np.float32)
        else:
            xb = filter_full_signal_band_butterworth(
                signals,
                fs,
                f_lo,
                f_hi,
                order=int(filter_order),
                sos=(
                    None if band_filter_sos is None
                    else band_filter_sos[b]
                ),
            )
            if xb is None:
                return b, band_tau, band_frame

        for t in range(T_use):
            s = int(frame_starts[t])
            e = s + frameSize

            if valid_mask is None:
                active_idx = np.arange(P, dtype=int)
            else:
                active_idx = np.where(valid_mask[:, b, t])[0]

            if active_idx.size == 0:
                continue

            for p in active_idx:
                i = int(iu[p])
                j = int(ju[p])
                maxlag = int(pair_max_tdoa[p])


                x_ref = xb[i, s:e]
                search_start = max(0, s - maxlag)
                search_end = min(N, e + maxlag)

                if search_end - search_start < frameSize:
                    continue

                y_search = xb[j, search_start:search_end]
                lag_min = search_start - s

                if correlation_mode == "norm_cc":
                    if correlation_weights is None:
                        x_ref_windowed = x_ref
                    else:
                        x_ref_windowed = x_ref * correlation_weights
                    scores, lags_local = fast_xcorr_short_in_long(
                        x_ref_windowed,
                        y_search,
                        lag_min=lag_min,
                        maxlag=maxlag,
                        weights=None,
                    )
                elif correlation_mode == "cc":
                    scores, lags_local = fast_xcorr_short_in_long_nonorm(
                        x_ref,
                        y_search,
                        lag_min=lag_min,
                        maxlag=maxlag,
                        weights=correlation_weights,
                    )
                else:
                    scores, lags_local = fast_gcc_short_in_long(
                        x_ref,
                        y_search,
                        fs=fs,
                        f_lo=f_lo,
                        f_hi=f_hi,
                        lag_min=lag_min,
                        maxlag=maxlag,
                        taper_hz=taper_hz,
                        alpha=gcc_alpha,
                        weights=correlation_weights,
                    )

                if scores.size == 0:
                    continue

                k = int(np.argmax(scores))
                tau_hat = -int(lags_local[k])

                band_tau[p].append(tau_hat)
                band_frame[p].append(int(t))

        return b, band_tau, band_frame

    band_indices = range(len(band_ranges_hz))
    if workers == 1 or len(band_ranges_hz) <= 1:
        band_results = map(process_band, band_indices)
    else:
        executor = ThreadPoolExecutor(
            max_workers=min(workers, len(band_ranges_hz)),
            thread_name_prefix="tdoa-band",
        )
        band_results = executor.map(process_band, band_indices)

    try:
        for b, band_tau, band_frame in band_results:
            for p in range(P):
                count = len(band_tau[p])
                if count == 0:
                    continue
                obs_tau[p].extend(band_tau[p])
                obs_band[p].extend([int(b)] * count)
                obs_frame[p].extend(band_frame[p])
    finally:
        if workers > 1 and len(band_ranges_hz) > 1:
            executor.shutdown(wait=True)

    return obs_tau, obs_band, obs_frame


# ============================================================
# Spectrogram masking and reconstruction
# ============================================================

def build_source_tf_masks(n_sources, n_bands, T, clusters, obs_band, obs_frame, obs_pair):
    """
    Build binary time-frequency masks per source from clustered observations.

    Parameters
    ----------
    n_sources : int
    n_bands : int
    T : int
    clusters : list of lists
        Each cluster contains indices of observations belonging to that source.
    obs_band, obs_frame, obs_pair : lists (per pair)
        Output from extraction pipeline (flattened indexing assumed externally).

    Returns
    -------
    masks : (n_sources, n_bands, T) ndarray
        Binary masks per source.
    """
    masks = np.zeros((n_sources, n_bands, T), dtype=np.float32)

    for s, cluster in enumerate(clusters):
        for idx in cluster:
            b = obs_band[idx]
            t = obs_frame[idx]
            masks[s, b, t] = 1.0

    return masks


def apply_masks_to_stft(signalsSTFT, W, source_masks):
    """
    Apply source masks to STFT to generate per-source spectrograms.

    Parameters
    ----------
    signalsSTFT : (M, F, T)
    W : (B, F)
    source_masks : (S, B, T)

    Returns
    -------
    source_STFTs : (S, M, F, T)
    """
    M, F, T = signalsSTFT.shape
    S, B, _ = source_masks.shape

    source_STFTs = np.zeros((S, M, F, T), dtype=np.complex64)

    for s in range(S):
        mask_full = np.zeros((F, T), dtype=np.float32)

        for b in range(B):
            mask_full += W[b][:, None] * source_masks[s, b][None, :]

        mask_full = np.clip(mask_full, 0.0, 1.0)

        for m in range(M):
            source_STFTs[s, m] = signalsSTFT[m] * mask_full

    return source_STFTs


def reconstruct_time_signals(source_STFTs, nfft, hop):
    """
    Reconstruct time-domain signals from masked STFTs using overlap-add.

    Parameters
    ----------
    source_STFTs : (S, M, F, T)
    nfft : int
    hop : int

    Returns
    -------
    signals_time : list of length S
        Each entry is (M, N) reconstructed signal
    """
    S, M, F, T = source_STFTs.shape
    N = (T - 1) * hop + nfft

    window = np.hanning(nfft).astype(np.float32)
    window_sq = window ** 2

    signals_out = []

    for s in range(S):
        sig = np.zeros((M, N), dtype=np.float32)

        for m in range(M):
            norm = np.zeros(N, dtype=np.float32)

            for t in range(T):
                frame = np.fft.irfft(source_STFTs[s, m, :, t], n=nfft)
                start = t * hop
                sig[m, start:start + nfft] += frame * window
                norm[start:start + nfft] += window_sq

            sig[m] /= np.maximum(norm, 1e-8)

        signals_out.append(sig.copy())

    return signals_out


# def smooth_source_maps(projected_maps, freq_sigma=1.0, time_sigma=1.0):
#     """
#     Smooth projected source maps with separable Gaussian filtering.
#
#     Parameters
#     ----------
#     projected_maps : list of (F, T) ndarrays
#         One projected source-support map per source.
#     freq_sigma, time_sigma : float
#         Gaussian std in frequency/time bins. If both are <= 0, maps are returned unchanged.
#
#     Returns
#     -------
#     smoothed_maps : list of (F, T) ndarrays
#     """
#     projected_maps = [np.asarray(m, dtype=np.float32) for m in projected_maps]
#     if len(projected_maps) == 0:
#         return []
#
#     if freq_sigma <= 0 and time_sigma <= 0:
#         return [np.maximum(m, 0.0) for m in projected_maps]
#
#     def _gaussian_kernel1d(sigma):
#         sigma = float(sigma)
#         if sigma <= 0:
#             return np.array([1.0], dtype=np.float32)
#         radius = max(1, int(np.ceil(3.0 * sigma)))
#         x = np.arange(-radius, radius + 1, dtype=np.float32)
#         k = np.exp(-0.5 * (x / sigma) ** 2)
#         k /= (np.sum(k) + 1e-12)
#         return k.astype(np.float32)
#
#     kf = _gaussian_kernel1d(freq_sigma)
#     kt = _gaussian_kernel1d(time_sigma)
#     pad_f = len(kf) // 2
#     pad_t = len(kt) // 2
#
#     smoothed_maps = []
#     for m in projected_maps:
#         x = np.maximum(m, 0.0).astype(np.float32, copy=False)
#
#         if len(kf) > 1:
#             xpad = np.pad(x, ((pad_f, pad_f), (0, 0)), mode="reflect")
#             x = np.apply_along_axis(lambda v: np.convolve(v, kf, mode="valid"), axis=0, arr=xpad)
#
#         if len(kt) > 1:
#             xpad = np.pad(x, ((0, 0), (pad_t, pad_t)), mode="reflect")
#             x = np.apply_along_axis(lambda v: np.convolve(v, kt, mode="valid"), axis=1, arr=xpad)
#
#         smoothed_maps.append(np.maximum(x, 0.0).astype(np.float32, copy=False))
#
#     return smoothed_maps

def smooth_source_maps(
    projected_maps,
    freq_sigma=1.0,
    time_sigma=1.0,
    *,
    core_gain=1.0,
    context_gain=0.35,
    background_floor=0.0,
    normalize_peak=True,
    context_mode="extend",
):
    """
    Add soft context around projected source maps while preserving the original
    cluster-assigned regions.

    This is different from simple smoothing. The original projected map is kept as
    the hard/core evidence, and a blurred copy is added only as nearby context:

        output = core_gain * original + context_gain * blurred(original)

    Optionally, `background_floor` can add a small nonzero value everywhere so that
    unsupported TF bins are not forced exactly to zero after mask normalization.

    Parameters
    ----------
    projected_maps : list of (F, T) ndarrays
        One projected source-support map per source.
    freq_sigma, time_sigma : float
        Gaussian std in frequency/time bins for the context blur.
        If both are <= 0, no context is added.
    core_gain : float
        Weight of the original cluster-assigned regions. Use 1.0 to keep them.
    context_gain : float
        Weight of the blurred context added around the original regions.
        Larger values make masks bleed more into nearby TF bins.
    background_floor : float
        Small fraction of the map peak added everywhere. For example 0.01 keeps
        a weak 1% floor instead of forcing unsupported regions to zero.
        Use 0.0 to keep unsupported regions exactly zero.
    normalize_peak : bool
        If True, each output map is normalized to max=1 when possible.
    context_mode : {"gaussian", "extend"}
        How to create the context map.
        - "gaussian": blurred context using Gaussian kernels.
        - "extend": hard extension/dilation by approximately freq_sigma and
          time_sigma bins, without Gaussian decay.

    Returns
    -------
    contextual_maps : list of (F, T) ndarrays
    """
    projected_maps = [np.asarray(m, dtype=np.float32) for m in projected_maps]
    if len(projected_maps) == 0:
        return []

    if core_gain < 0 or context_gain < 0:
        raise ValueError("core_gain and context_gain must be >= 0")
    if background_floor < 0:
        raise ValueError("background_floor must be >= 0")

    context_mode = str(context_mode).lower()
    if context_mode not in ("gaussian", "extend"):
        raise ValueError("context_mode must be 'gaussian' or 'extend'")

    def _gaussian_kernel1d(sigma):
        sigma = float(sigma)
        if sigma <= 0:
            return np.array([1.0], dtype=np.float32)
        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        # Energy-normalize the kernel. Peak-normalization makes the total blur
        # energy grow with sigma, so context_gain becomes sigma-dependent and
        # hard to interpret.
        k /= (np.sum(k) + 1e-12)
        return k.astype(np.float32)

    def _blur_map(x):
        kf = _gaussian_kernel1d(freq_sigma)
        kt = _gaussian_kernel1d(time_sigma)
        pad_f = len(kf) // 2
        pad_t = len(kt) // 2

        y = x.astype(np.float32, copy=True)

        if len(kf) > 1:
            ypad = np.pad(y, ((pad_f, pad_f), (0, 0)), mode="reflect")
            y = np.apply_along_axis(lambda v: np.convolve(v, kf, mode="valid"), axis=0, arr=ypad)

        if len(kt) > 1:
            ypad = np.pad(y, ((0, 0), (pad_t, pad_t)), mode="reflect")
            y = np.apply_along_axis(lambda v: np.convolve(v, kt, mode="valid"), axis=1, arr=ypad)

        return np.maximum(y, 0.0).astype(np.float32, copy=False)

    def _extend_map(x):
        """
        Hard-extend/dilate a support map by freq_sigma and time_sigma bins.

        Unlike _blur_map, this does not create Gaussian decay. Every bin inside
        the rectangular neighborhood of an active/core bin receives the local
        maximum support value found in that neighborhood.
        """
        rf = max(0, int(np.ceil(float(freq_sigma))))
        rt = max(0, int(np.ceil(float(time_sigma))))

        if rf == 0 and rt == 0:
            return x.astype(np.float32, copy=True)

        x = np.maximum(x, 0.0).astype(np.float32, copy=False)
        y = np.zeros_like(x, dtype=np.float32)

        F, T = x.shape
        active_f, active_t = np.nonzero(x > 0.0)

        for f, t in zip(active_f, active_t):
            f0 = max(0, f - rf)
            f1 = min(F, f + rf + 1)
            t0 = max(0, t - rt)
            t1 = min(T, t + rt + 1)
            y[f0:f1, t0:t1] = np.maximum(y[f0:f1, t0:t1], x[f, t])

        return y.astype(np.float32, copy=False)

    # Bins with original support from any source. The background floor should
    # only fill bins unsupported by all sources, not the core region of this or
    # another source.
    core_stack = np.stack([np.maximum(m, 0.0) for m in projected_maps], axis=0)
    any_source_support = np.any(core_stack > 0.0, axis=0)
    unsupported_by_all_sources = ~any_source_support
    contextual_maps = []
    for m in projected_maps:
        core = np.maximum(m, 0.0).astype(np.float32, copy=False)

        if freq_sigma <= 0 and time_sigma <= 0:
            context = np.zeros_like(core, dtype=np.float32)
        elif context_mode == "gaussian":
            context = _blur_map(core)
        elif context_mode == "extend":
            context = _extend_map(core)
        else:
            raise RuntimeError(f"Unexpected context_mode: {context_mode}")

        out = core_gain * core + context_gain * context

        peak = float(np.max(out)) if out.size else 0.0
        if background_floor > 0 and peak > 0:
            out = out.copy()
            out[unsupported_by_all_sources] += background_floor * peak

        if normalize_peak:
            peak = float(np.max(out)) if out.size else 0.0
            if peak > 0:
                out = out / peak

        contextual_maps.append(np.maximum(out, 0.0).astype(np.float32, copy=False))

    return contextual_maps


def morph_source_maps(
    projected_maps,
    *,
    threshold=0.0,
    open_size=(1, 1),
    close_size=(1, 1),
    dilate_size=(1, 1),
    post_area_close_size=(0, 0),
    min_area=0,
    area_connectivity=1,
    area_filter_position="after_close",
    core_gain=1.0,
    context_gain=0.35,
    background_floor=0.0,
    normalize_peak=True,
    return_binary=False,
):
    """
    Build contextual source maps using mathematical morphology instead of
    Gaussian smoothing.

    This is an alternative to `smooth_source_maps`. It treats each source map as
    a TF support image and applies binary morphological operations before adding
    context around the cleaned core support.

    Typical use:
        - opening removes tiny isolated TF islands
        - closing fills small holes and connects nearby support fragments
        - dilation expands the support region by a controlled number of bins
        - an optional second close can reconnect/smooth surviving components
          after the area filter

    Parameters
    ----------
    projected_maps : list of (B, T) or (F, T) ndarrays
        One source-support map per source.
    threshold : float
        Values above this threshold are considered active support.
    open_size : tuple(int, int)
        Structuring element size for binary opening in (freq_bins, time_bins).
        Use (0, 0) or (1, 1) to disable effectively. Active even dimensions
        are promoted to the next odd size so the operation is centered.
    close_size : tuple(int, int)
        Structuring element size for binary closing in (freq_bins, time_bins).
    dilate_size : tuple(int, int)
        Structuring element size for binary dilation in (freq_bins, time_bins).
        This creates context around the cleaned core support.
    post_area_close_size : tuple(int, int)
        Optional second closing after the connected-component area filter.
        Usually keep (0, 0), or use a small value such as (1, 2).
    min_area : int
        Connected-component area filter. Components with fewer than `min_area`
        active TF cells are removed. Use 0 or 1 to disable. This is often safer
        than a 2x2 opening because it can preserve thin but long source tracks.
    area_connectivity : int
        Connectivity used for connected components. Use 1 for 4-neighborhood
        connectivity and 2 for 8-neighborhood connectivity in the TF image.
    area_filter_position : {"before_open", "after_open", "after_close"}
        Where to apply connected-component area filtering in the morphology
        pipeline. The default "after_close" first connects nearby fragments and
        then removes components that are still too small.
    core_gain : float
        Weight assigned to the cleaned core support.
    context_gain : float
        Weight assigned to the dilated contextual region outside the core.
    background_floor : float
        Small fraction of the map peak added only to TF bins unsupported by all
        sources. Use 0.0 to keep unsupported bins exactly zero.
    normalize_peak : bool
        If True, normalize each output map to max=1 when possible.
    return_binary : bool
        If True, return binary maps after morphology instead of weighted maps.

    Returns
    -------
    morph_maps : list of ndarrays
        One morphology-processed map per source.
    """
    maps = [np.maximum(np.asarray(m, dtype=np.float32), 0.0) for m in projected_maps]
    if len(maps) == 0:
        return []

    if core_gain < 0 or context_gain < 0:
        raise ValueError("core_gain and context_gain must be >= 0")
    if background_floor < 0:
        raise ValueError("background_floor must be >= 0")

    min_area = int(min_area)
    area_connectivity = int(area_connectivity)
    area_filter_position = str(area_filter_position).lower()
    if area_connectivity not in (1, 2):
        raise ValueError("area_connectivity must be 1 or 2")
    if area_filter_position not in ("before_open", "after_open", "after_close"):
        raise ValueError(
            "area_filter_position must be one of: "
            "'before_open', 'after_open', 'after_close'"
        )

    def _selem(size):
        rf, rt = int(size[0]), int(size[1])
        if rf <= 1 and rt <= 1:
            return None
        rf = max(1, rf)
        rt = max(1, rt)
        # Even-sized scipy.ndimage footprints have no central cell and default
        # to an origin biased toward lower indices (down/left in our TF plots).
        rf += int(rf % 2 == 0)
        rt += int(rt % 2 == 0)
        return np.ones((rf, rt), dtype=bool)

    def _area_filter(binary):
        """Remove connected components smaller than min_area."""
        binary = np.asarray(binary, dtype=bool)
        if min_area <= 1:
            return binary

        from scipy.ndimage import label

        if area_connectivity == 1:
            structure = np.array(
                [[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]],
                dtype=bool,
            )
        else:
            structure = np.ones((3, 3), dtype=bool)

        labels, n_labels = label(binary, structure=structure)
        if n_labels == 0:
            return binary

        counts = np.bincount(labels.ravel())
        keep_labels = np.flatnonzero(counts >= min_area)
        keep_labels = keep_labels[keep_labels != 0]
        if keep_labels.size == 0:
            return np.zeros_like(binary, dtype=bool)

        return np.isin(labels, keep_labels)

    se_open = _selem(open_size)
    se_close = _selem(close_size)
    se_post_area_close = _selem(post_area_close_size)
    se_dilate = _selem(dilate_size)

    core_binaries = []
    context_binaries = []

    for m in maps:
        core = m > float(threshold)

        if area_filter_position == "before_open":
            core = _area_filter(core)

        if se_open is not None:
            core = binary_opening(core, structure=se_open)

        if area_filter_position == "after_open":
            core = _area_filter(core)

        if se_close is not None:
            core = binary_closing(core, structure=se_close)

        if area_filter_position == "after_close":
            core = _area_filter(core)

        if se_post_area_close is not None:
            core = binary_closing(core, structure=se_post_area_close)

        if se_dilate is not None:
            context = binary_dilation(core, structure=se_dilate)
        else:
            context = core.copy()

        core_binaries.append(core)
        context_binaries.append(context)

    any_core_support = np.any(np.stack(core_binaries, axis=0), axis=0)
    unsupported_by_all_sources = ~any_core_support

    morph_maps = []
    for core, context in zip(core_binaries, context_binaries):
        if return_binary:
            out = context.astype(np.float32)
        else:
            out = np.zeros(core.shape, dtype=np.float32)
            out[context] = float(context_gain)
            out[core] = float(core_gain)

            peak = float(np.max(out)) if out.size else 0.0
            if background_floor > 0 and peak > 0:
                out = out.copy()
                out[unsupported_by_all_sources] += background_floor * peak

            if normalize_peak:
                peak = float(np.max(out)) if out.size else 0.0
                if peak > 0:
                    out = out / peak

        morph_maps.append(np.maximum(out, 0.0).astype(np.float32, copy=False))

    return morph_maps

def build_soft_masks_from_projected_maps(
    projected_maps,
    eps=1e-8,
    power=1.0,
    *,
    unsupported_mode="zero",
    unsupported_value=0.0,
):
    """
    Convert projected source-support maps into normalized soft masks.

    Parameters
    ----------
    projected_maps : list of (F, T) ndarrays
        One nonnegative map per source, all on the same STFT grid.
    eps : float
        Numerical stabilizer.
    power : float
        Optional exponent applied before normalization.
        - power = 1.0 : standard soft masks
        - power < 1.0 : softer masks with more fade into nearby TF regions
        - power > 1.0 : sharper masks, closer to hard assignment
    unsupported_mode : {"zero", "uniform", "value"}
        How to treat TF bins where all source maps are zero:
        - "zero"    : all source masks remain zero there.
        - "uniform" : unsupported bins are split equally across sources.
        - "value"   : unsupported bins are set to `unsupported_value` for every source.
    unsupported_value : float
        Value used when unsupported_mode="value".

    Returns
    -------
    soft_masks : list of (F, T) ndarrays
        Soft masks that sum to ~1 across sources at each supported TF bin.
    """
    projected_maps = [np.maximum(np.asarray(m, dtype=np.float32), 0.0) for m in projected_maps]
    if len(projected_maps) == 0:
        return []

    if power <= 0:
        raise ValueError("power must be > 0")

    stack0 = np.stack(projected_maps, axis=0)
    unsupported = np.sum(stack0, axis=0) <= eps
    stack = np.maximum(stack0, 0.0)

    # stack = np.stack(projected_maps, axis=0)  # (S, F, T)
    if power != 1.0:
        # stack = np.power(stack + eps, power)
        stack = np.power(stack, power)


    denom = np.sum(stack, axis=0, keepdims=True)
    soft = stack / (denom + eps)

    soft[:, unsupported] = 0.0

    unsupported = denom[0] <= eps
    unsupported_mode = unsupported_mode.lower()

    if unsupported_mode == "zero":
        soft[:, unsupported] = 0.0
    elif unsupported_mode == "uniform":
        soft[:, unsupported] = 1.0 / float(soft.shape[0])
    elif unsupported_mode == "value":
        if unsupported_value < 0:
            raise ValueError("unsupported_value must be >= 0")
        soft[:, unsupported] = float(unsupported_value)
    else:
        raise ValueError(f"Unknown unsupported_mode: {unsupported_mode}")

    return [soft[s].astype(np.float32, copy=False) for s in range(soft.shape[0])]


# def separate_sources_with_soft_masks(
#         signal,
#         fs,
#         soft_masks,
#         *,
#         nperseg,
#         noverlap,
#         nfft,
# ):
#     """
#     Reconstruct one time-domain signal per source from a reference signal and soft TF masks.
#     """
#     from scipy.signal import stft, istft
#
#     signal = np.asarray(signal, dtype=np.float32)
#     if len(soft_masks) == 0:
#         return [], None
#
#     _, _, Zxx = stft(
#         signal,
#         fs=fs,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         nfft=nfft,
#         boundary=None,
#         padded=False,
#     )
#
#     separated_signals = []
#     for sid, mask in enumerate(soft_masks):
#         mask = np.asarray(mask, dtype=np.float32)
#         if mask.shape != Zxx.shape:
#             raise ValueError(
#                 f"Mask {sid} has shape {mask.shape}, but reference STFT has shape {Zxx.shape}"
#             )
#
#         Z_masked = Zxx * mask
#         _, x_rec = istft(
#             Z_masked,
#             fs=fs,
#             nperseg=nperseg,
#             noverlap=noverlap,
#             nfft=nfft,
#             input_onesided=True,
#             boundary=False,
#         )
#         separated_signals.append(np.asarray(x_rec, dtype=np.float32))
#
#     return separated_signals, Zxx



def separate_sources_with_soft_masks(
        signal,
        fs,
        soft_masks,
        *,
        nperseg,
        noverlap,
        nfft,
        mic_positions=None,
        source_positions=None,
        return_reference_mics=False,
):
    signal = np.asarray(signal, dtype=np.float32)

    closest_mic_mode = (
            signal.ndim == 2
            and mic_positions is not None
            and source_positions is not None
    )

    if closest_mic_mode:
        mic_positions = np.asarray(mic_positions, dtype=float)
        source_positions = np.asarray(source_positions, dtype=float)

        dists = np.linalg.norm(
            source_positions[:, None, :] - mic_positions[None, :, :],
            axis=-1,
        )
        reference_mics = np.argmin(dists, axis=1).astype(int).tolist()
    else:
        reference_mics = [0 for _ in soft_masks]

    separated_signals = []
    Zxx_refs = []
    stft_cache = {}

    for sid, mask in enumerate(soft_masks):
        ref_mic = reference_mics[sid]

        if signal.ndim == 1:
            x_ref = signal
            cache_key = "single"
        else:
            x_ref = signal[ref_mic]
            cache_key = ref_mic

        if cache_key not in stft_cache:
            _, _, Zxx_ref = stft(
                x_ref,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                boundary=None,
                padded=False,
            )
            stft_cache[cache_key] = Zxx_ref
        else:
            Zxx_ref = stft_cache[cache_key]

        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape != Zxx_ref.shape:
            raise ValueError(
                f"Mask {sid} has shape {mask.shape}, "
                f"but reference STFT for mic {ref_mic} has shape {Zxx_ref.shape}"
            )

        Z_masked = Zxx_ref * mask

        _, x_rec = istft(
            Z_masked,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            input_onesided=True,
            boundary=False,
        )

        separated_signals.append(np.asarray(x_rec, dtype=np.float32))
        Zxx_refs.append(Zxx_ref)

    if return_reference_mics:
        return separated_signals, Zxx_refs, reference_mics

    return separated_signals, Zxx_refs


# ============================================================
# Delay-and-sum source reconstruction from TDOA associations
# ============================================================

def _fractional_delay_fft(x, delay_samples):
    """
    Delay a real signal by a possibly fractional number of samples using the
    Fourier shift theorem on a zero-padded signal.

    Positive delay_samples shifts the signal later in time. Negative values
    advance it. Padding converts the FFT's circular shift into a linear shift
    over the returned analysis window, so delayed edge energy is discarded
    instead of wrapping around to the opposite edge.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    if n == 0:
        return x.copy()

    delay_samples = float(delay_samples)
    pad = int(np.ceil(abs(delay_samples))) + 64
    x_pad = np.pad(x, (pad, pad), mode="constant", constant_values=0.0)

    freqs = np.fft.rfftfreq(x_pad.shape[-1])
    X = np.fft.rfft(x_pad)
    phase = np.exp(-2j * np.pi * freqs * float(delay_samples))
    y_pad = np.fft.irfft(X * phase, n=x_pad.shape[-1])
    y = y_pad[pad:pad + n]
    return np.asarray(y, dtype=np.float32)


def _solve_relative_delays_from_source(source, n_mics, *, reference="mean"):
    """
    Estimate one relative delay per microphone from the pairwise TDOAs stored in
    one associated source.

    Convention assumed here matches the localization model used elsewhere:
        tau_ij = t_i - t_j

    For every source member with pair (i, j), solve:
        d_i - d_j = tau_ij

    The solution is only identifiable up to a constant offset. By default,
    reference='mean' imposes mean(d)=0 over the microphones involved in the
    source. If reference is an integer microphone index, that microphone delay is
    fixed to zero.
    """
    rows = []
    rhs = []
    used_mics = set()

    for mem in source.get("members", []):
        pair = mem.get("pair")
        if pair is None or len(pair) < 2 or "tau" not in mem:
            continue

        i, j = int(pair[0]), int(pair[1])
        tau = float(mem["tau"])

        row = np.zeros(int(n_mics), dtype=float)
        row[i] = 1.0
        row[j] = -1.0
        rows.append(row)
        rhs.append(tau)
        used_mics.add(i)
        used_mics.add(j)

    if len(rows) == 0:
        return None, []

    used_mics = sorted(used_mics)
    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=float)

    if reference == "mean":
        constraint = np.zeros(int(n_mics), dtype=float)
        constraint[used_mics] = 1.0 / float(len(used_mics))
        A = np.vstack([A, constraint])
        b = np.concatenate([b, [0.0]])
    elif reference is not None:
        ref = int(reference)
        constraint = np.zeros(int(n_mics), dtype=float)
        constraint[ref] = 1.0
        A = np.vstack([A, constraint])
        b = np.concatenate([b, [0.0]])

    delays, *_ = np.linalg.lstsq(A, b, rcond=None)
    return delays.astype(float), used_mics


def delay_and_sum_sources_from_tdoas(
        signals,
        sources,
        *,
        reference="mean",
        use_only_source_mics=True,
        normalize=True,
        return_debug=False,
):
    """
    Reconstruct one delay-and-sum signal per associated TDOA source.

    For each source, this function solves a consistent set of relative microphone
    delays from the source's pairwise TDOAs and then aligns the microphones by
    applying the opposite delays. The aligned signals are averaged/summed to
    produce one waveform per source.

    Parameters
    ----------
    signals : ndarray, shape (n_mics, n_samples)
        Time-domain microphone signals for the selected analysis segment.
    sources : list of dict
        Associated sources. Each source must contain members with fields:
        `pair` and `tau`, where `tau` is in samples.
    reference : "mean", int, list/tuple/ndarray, or None
        Constraint used to fix the arbitrary delay offset.
        - "mean" sets the mean delay of microphones involved in each source to zero.
        - an integer fixes that microphone delay to zero for every source.
        - a list/tuple/ndarray provides one reference mic per source; use this
          when source maps are aligned to per-source reference microphones.
        - None leaves the least-squares system unconstrained, which is not recommended.
    use_only_source_mics : bool
        If True, sum only microphones that appear in at least one TDOA member of
        that source. If False, all microphones are included; microphones without
        constraints keep whatever delay is implied by the least-squares solution,
        usually close to zero.
    normalize : bool
        If True, average by the number of summed microphones. If False, return
        the raw sum.
    return_debug : bool
        If True, also return per-source delays, used microphones, and alignment
        residuals.

    Returns
    -------
    reconstructed : list of ndarray
        One delay-and-sum waveform per source.
    debug : list of dict, optional
        Returned only if return_debug=True.
    """
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2:
        raise ValueError("signals must have shape (n_mics, n_samples)")

    n_mics, n_samples = signals.shape
    reconstructed = []
    debug = []

    if isinstance(reference, (list, tuple, np.ndarray)):
        reference_per_source = list(reference)
        if len(reference_per_source) != len(sources):
            raise ValueError(
                f"reference has length {len(reference_per_source)} but sources has length {len(sources)}"
            )
    else:
        reference_per_source = [reference for _ in sources]

    for src_idx, source in enumerate(sources):
        src_reference = reference_per_source[src_idx]
        if src_reference is not None and src_reference != "mean":
            src_reference = int(src_reference)

        delays, used_mics = _solve_relative_delays_from_source(
            source,
            n_mics,
            reference=src_reference,
        )

        if delays is None or len(used_mics) == 0:
            reconstructed.append(np.zeros(n_samples, dtype=np.float32))
            debug.append({
                "source_idx": int(src_idx),
                "reference": src_reference,
                "delays_samples": None,
                "used_mics": [],
                "pair_residuals_samples": [],
            })
            continue

        if use_only_source_mics:
            mics_to_sum = used_mics
        else:
            mics_to_sum = list(range(n_mics))

        aligned = []
        for m in mics_to_sum:
            # signals arrive delayed by d_m, so alignment applies -d_m.
            aligned.append(_fractional_delay_fft(signals[int(m)], -float(delays[int(m)])))

        y = np.sum(np.stack(aligned, axis=0), axis=0)
        if normalize and len(aligned) > 0:
            y = y / float(len(aligned))

        pair_residuals = []
        for mem in source.get("members", []):
            pair = mem.get("pair")
            if pair is None or len(pair) < 2 or "tau" not in mem:
                continue
            i, j = int(pair[0]), int(pair[1])
            tau = float(mem["tau"])
            pair_residuals.append(float((delays[i] - delays[j]) - tau))

        reconstructed.append(np.asarray(y, dtype=np.float32))
        debug.append({
            "source_idx": int(src_idx),
            "reference": src_reference,
            "delays_samples": delays.copy(),
            "used_mics": [int(m) for m in used_mics],
            "summed_mics": [int(m) for m in mics_to_sum],
            "pair_residuals_samples": pair_residuals,
        })

    if return_debug:
        return reconstructed, debug

    return reconstructed
def _maad_roi_features_from_signal(
        x,
        fs,
        nperseg,
        noverlap,
        dB_max=96,
        smooth_std=0.5,
):
    # Import MAAD only when ROI extraction actually runs. Eagerly importing it
    # before Perch v2's first XLA call can deadlock TensorFlow on Apple Silicon.
    from maad import sound
    from maad.util import power2dB

    Sxx_power, tn, fn, ext = sound.spectrogram(
        x=x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    Sxx_db = power2dB(Sxx_power) + dB_max

    Sxx_power_noNoise = sound.median_equalizer(
        Sxx=Sxx_power,
        display=False,
    )
    Sxx_db_noNoise = power2dB(Sxx_power_noNoise)

    Sxx_db_noNoise_smooth = sound.smooth(
        Sxx=Sxx_db_noNoise,
        std=smooth_std,
        display=False,
        savefig=None,
    )
    return Sxx_db_noNoise_smooth, tn, fn, ext, Sxx_db


def _maad_roi_mask_from_features(
        smoothed_features,
        *,
        bin_std=8,
        bin_per=0.5,
        min_roi=25,
        max_roi=None,
):
    from maad import rois

    im_mask = rois.create_mask(
        im=smoothed_features,
        mode_bin="relative",
        bin_std=bin_std,
        bin_per=bin_per,
        verbose=False,
        display=False,
    )

    im_rois, df_rois = rois.select_rois(
        im_bin=im_mask,
        min_roi=min_roi,
        max_roi=max_roi,
        display=False,
    )

    roi_mask = im_rois > 0
    return roi_mask, im_rois, df_rois


def _maad_roi_mask_from_signal(
        x,
        fs,
        nperseg,
        noverlap,
        dB_max=96,
        bin_std=8,
        bin_per=0.5,
        smooth_std=0.5,
        min_roi=25,
        max_roi=None,
):
    """Compute one scikit-maad ROI mask from a waveform."""
    features, tn, fn, ext, Sxx_db = _maad_roi_features_from_signal(
        x=x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        dB_max=dB_max,
        smooth_std=smooth_std,
    )
    roi_mask, im_rois, df_rois = _maad_roi_mask_from_features(
        features,
        bin_std=bin_std,
        bin_per=bin_per,
        min_roi=min_roi,
        max_roi=max_roi,
    )
    return roi_mask, tn, fn, ext, Sxx_db, im_rois, df_rois


def build_shared_maad_roi_masks(
        signals_win,
        fs,
        *,
        nperseg_maad=512,
        noverlap_maad=256,
        dB_max=96,
        smooth_std=0.5,
        pair_roi_kwargs=None,
        mic_roi_kwargs=None,
):
    """
    Build the two per-microphone ROI-mask variants from shared MAAD features.

    Spectrogram, median equalization, dB conversion, and smoothing are computed
    once per microphone. Only the inexpensive thresholding/ROI-selection stage
    is repeated for the pair-validity and microphone-selection configurations.
    """
    pair_roi_kwargs = dict(pair_roi_kwargs or {})
    mic_roi_kwargs = dict(mic_roi_kwargs or pair_roi_kwargs)
    pair_roi_masks = []
    mic_roi_masks = []

    for signal in np.asarray(signals_win):
        features, _, _, _, _ = _maad_roi_features_from_signal(
            x=signal,
            fs=fs,
            nperseg=nperseg_maad,
            noverlap=noverlap_maad,
            dB_max=dB_max,
            smooth_std=smooth_std,
        )
        pair_mask, _, _ = _maad_roi_mask_from_features(
            features,
            **pair_roi_kwargs,
        )
        if mic_roi_kwargs == pair_roi_kwargs:
            mic_mask = pair_mask.copy()
        else:
            mic_mask, _, _ = _maad_roi_mask_from_features(
                features,
                **mic_roi_kwargs,
            )
        pair_roi_masks.append(pair_mask)
        mic_roi_masks.append(mic_mask)

    return (
        np.asarray(pair_roi_masks, dtype=bool),
        np.asarray(mic_roi_masks, dtype=bool),
    )


def build_maad_roi_valid_mask_from_signals(
        signals_win,
        fs,
        Omega_list,
        iu,
        ju,
        nfft_alg,
        hop_alg,
        nperseg_maad=512,
        noverlap_maad=256,
        dB_max=96,
        bin_std=8,
        bin_per=0.5,
        smooth_std=0.5,
        min_roi=25,
        max_roi=None,
        pair_mode="max",
        force_last_frame=False,
        roi_masks=None,
):
    """
    Build valid_mask[pair, band, alg_frame] using scikit-maad ROIs.

    scikit-maad ROIs are computed on a shorter visualization-like STFT grid
    and then projected onto the algorithmic sub-band/time grid used by the
    TDOA extractor.

    If force_last_frame=True, the returned mask uses the same additional final
    full-frame grid as extract_subband_tdoas_windowed_search(...,
    force_last_frame=True).
    """
    n_pairs = len(iu)
    n_bands = len(Omega_list)
    frame_starts_alg = compute_frame_starts(
        signals_win.shape[1],
        nfft_alg,
        hop_alg,
        force_last_frame=force_last_frame,
    )
    n_frames_alg = len(frame_starts_alg)
    valid_mask = np.zeros((n_pairs, n_bands, n_frames_alg), dtype=bool)

    if roi_masks is None:
        roi_masks = []
        for m in range(signals_win.shape[0]):
            roi_mask_m, _, _, _, _, _, _ = _maad_roi_mask_from_signal(
                x=signals_win[m],
                fs=fs,
                nperseg=nperseg_maad,
                noverlap=noverlap_maad,
                dB_max=dB_max,
                bin_std=bin_std,
                bin_per=bin_per,
                smooth_std=smooth_std,
                min_roi=min_roi,
                max_roi=max_roi,
            )
            roi_masks.append(roi_mask_m)
    roi_masks = np.asarray(roi_masks, dtype=bool)

    # Map each algorithmic frame to all overlapping MAAD/STFT frames.
    # Previously this used only the MAAD frame nearest to the algorithm-frame
    # center. That can reject a valid algorithm frame when the ROI is present
    # inside the frame but not exactly at its center. This now mirrors the
    # frequency projection below: a band/frame is valid if ANY MAAD TF cell
    # overlapping that algorithmic TF region is active.
    maad_hop = nperseg_maad - noverlap_maad
    maad_n_frames = roi_masks.shape[2]
    maad_frame_starts = np.arange(maad_n_frames, dtype=float) * maad_hop
    maad_frame_ends = maad_frame_starts + float(nperseg_maad)

    frame_maad_indices = []
    for s_alg in frame_starts_alg:
        alg_start = float(s_alg)
        alg_end = alg_start + float(nfft_alg)

        overlap = (maad_frame_starts < alg_end) & (maad_frame_ends > alg_start)
        idx = np.flatnonzero(overlap)

        if idx.size == 0:
            # Fallback to nearest MAAD frame center. This should rarely happen,
            # but it keeps the function robust for very short or edge cases.
            alg_center = alg_start + 0.5 * float(nfft_alg)
            maad_centers = maad_frame_starts + 0.5 * float(nperseg_maad)
            idx = np.array([int(np.argmin(np.abs(maad_centers - alg_center)))], dtype=int)

        frame_maad_indices.append(idx.astype(int))

    # Map algorithmic frequency bins to maad frequency bins.
    freq_res_alg = fs / float(nfft_alg)
    freq_res_maad = fs / float(nperseg_maad)

    for p, (i, j) in enumerate(zip(iu, ju)):
        if pair_mode == "min":
            pair_roi_mask = roi_masks[int(i)] & roi_masks[int(j)]
        elif pair_mode == "max":
            pair_roi_mask = roi_masks[int(i)] | roi_masks[int(j)]
        elif pair_mode == "reference_i":
            pair_roi_mask = roi_masks[int(i)]
        elif pair_mode == "reference_j":
            pair_roi_mask = roi_masks[int(j)]
        else:
            raise ValueError(f"Unknown pair_mode: {pair_mode}")

        for b, omega_idx in enumerate(Omega_list):
            freqs_band = np.asarray(omega_idx, dtype=float) * freq_res_alg
            maad_freq_idx = np.rint(freqs_band / freq_res_maad).astype(int)
            maad_freq_idx = np.clip(maad_freq_idx, 0, pair_roi_mask.shape[0] - 1)
            maad_freq_idx = np.unique(maad_freq_idx)

            # For each algorithmic frame, check if any of the overlapping MAAD time frames are active.
            for t_alg, maad_time_idx in enumerate(frame_maad_indices):
                valid_mask[p, b, t_alg] = np.any(
                    pair_roi_mask[np.ix_(maad_freq_idx, maad_time_idx)]
                )

    return valid_mask


def build_maad_roi_mask_per_mic(
        signals_win,
        fs,
        Omega_list,
        nfft_alg,
        hop_alg,
        nperseg_maad=512,
        noverlap_maad=256,
        dB_max=96,
        bin_std=8,
        bin_per=0.5,
        smooth_std=0.5,
        min_roi=25,
        max_roi=None,
        force_last_frame=False,
        roi_masks=None,
):
    """
    Build mic_valid_mask[mic, band, alg_frame] using scikit-maad ROIs.

    This is the per-microphone version of build_maad_roi_valid_mask_from_signals().
    It is useful when the pipeline needs to choose which real microphone best
    contains a given source, rather than only deciding whether a pair is valid.
    """
    n_mics = signals_win.shape[0]
    n_bands = len(Omega_list)
    frame_starts_alg = compute_frame_starts(
        signals_win.shape[1],
        nfft_alg,
        hop_alg,
        force_last_frame=force_last_frame,
    )
    n_frames_alg = len(frame_starts_alg)
    mic_valid_mask = np.zeros((n_mics, n_bands, n_frames_alg), dtype=bool)

    if roi_masks is None:
        roi_masks = []
        for m in range(n_mics):
            roi_mask_m, _, _, _, _, _, _ = _maad_roi_mask_from_signal(
                x=signals_win[m],
                fs=fs,
                nperseg=nperseg_maad,
                noverlap=noverlap_maad,
                dB_max=dB_max,
                bin_std=bin_std,
                bin_per=bin_per,
                smooth_std=smooth_std,
                min_roi=min_roi,
                max_roi=max_roi,
            )
            roi_masks.append(roi_mask_m)
    roi_masks = np.asarray(roi_masks, dtype=bool)

    maad_hop = nperseg_maad - noverlap_maad
    maad_n_frames = roi_masks.shape[2]
    maad_frame_starts = np.arange(maad_n_frames, dtype=float) * maad_hop
    maad_frame_ends = maad_frame_starts + float(nperseg_maad)

    frame_maad_indices = []
    for s_alg in frame_starts_alg:
        alg_start = float(s_alg)
        alg_end = alg_start + float(nfft_alg)

        overlap = (maad_frame_starts < alg_end) & (maad_frame_ends > alg_start)
        idx = np.flatnonzero(overlap)

        if idx.size == 0:
            alg_center = alg_start + 0.5 * float(nfft_alg)
            maad_centers = maad_frame_starts + 0.5 * float(nperseg_maad)
            idx = np.array([int(np.argmin(np.abs(maad_centers - alg_center)))], dtype=int)

        frame_maad_indices.append(idx.astype(int))

    freq_res_alg = fs / float(nfft_alg)
    freq_res_maad = fs / float(nperseg_maad)

    for b, omega_idx in enumerate(Omega_list):
        freqs_band = np.asarray(omega_idx, dtype=float) * freq_res_alg
        maad_freq_idx = np.rint(freqs_band / freq_res_maad).astype(int)
        maad_freq_idx = np.clip(maad_freq_idx, 0, roi_masks.shape[1] - 1)
        maad_freq_idx = np.unique(maad_freq_idx)

        for t_alg, maad_time_idx in enumerate(frame_maad_indices):
            mic_valid_mask[:, b, t_alg] = np.any(
                roi_masks[:, maad_freq_idx[:, None], maad_time_idx[None, :]],
                axis=(1, 2),
            )

    return mic_valid_mask, roi_masks


def _project_highres_roi_mask_to_stft(
        roi_mask,
        fs,
        target_shape,
        *,
        target_nperseg,
        target_noverlap,
        target_nfft,
        roi_nperseg,
        roi_noverlap,
        target_boundary_zeros=False,
):
    """Project a binary MAAD ROI mask onto a reconstruction STFT grid."""
    roi_mask = np.asarray(roi_mask, dtype=bool)
    if roi_mask.ndim != 2:
        raise ValueError(
            f"High-resolution ROI mask must be 2D, got {roi_mask.shape}"
        )

    target_f, target_t = map(int, target_shape)
    target_hop = int(target_nperseg - target_noverlap)
    roi_hop = int(roi_nperseg - roi_noverlap)
    if min(target_hop, roi_hop, target_nfft, roi_nperseg) <= 0:
        raise ValueError("ROI and reconstruction STFT sizes must be positive")

    target_freqs = np.arange(target_f, dtype=float) * fs / float(target_nfft)
    roi_freqs = (
        np.arange(roi_mask.shape[0], dtype=float)
        * fs
        / float(roi_nperseg)
    )
    roi_freq_idx = np.rint(
        target_freqs / max(fs / float(roi_nperseg), 1e-12)
    ).astype(int)
    roi_freq_idx = np.clip(roi_freq_idx, 0, roi_freqs.size - 1)

    roi_starts = np.arange(roi_mask.shape[1], dtype=float) * roi_hop
    roi_ends = roi_starts + float(roi_nperseg)
    if target_boundary_zeros:
        target_starts = (
            np.arange(target_t, dtype=float) * target_hop
            - 0.5 * float(target_nperseg)
        )
    else:
        target_starts = np.arange(target_t, dtype=float) * target_hop
    target_ends = target_starts + float(target_nperseg)

    projected = np.zeros((target_f, target_t), dtype=np.float32)
    for target_idx, (start, end) in enumerate(
        zip(target_starts, target_ends)
    ):
        overlapping = np.flatnonzero(
            (roi_starts < end) & (roi_ends > start)
        )
        if overlapping.size == 0:
            continue
        projected[:, target_idx] = np.any(
            roi_mask[np.ix_(roi_freq_idx, overlapping)],
            axis=1,
        )
    return projected


def suppress_other_source_regions_from_das(
    das_signals,
    soft_masks,
    fs,
    nperseg,
    noverlap,
    nfft,
    threshold=0.25,
    mode="other_sources",
    projected_maps=None,
    valid_mask=None,
    Omega_list=None,
    source_delays=None,
    mask_reference_mic=None,
    target_mics=None,
    mask_hop=None,
    mask_nperseg=None,
    mask_nfft=None,
    taper_freq_bins=0.0,
    taper_time_frames=0.0,
    mic_roi_masks_high_res=None,
    roi_nperseg=1024,
    roi_noverlap=512,
):
    """
    Suppress regions of DAS signals using source-support maps.

    Supported modes
    ---------------
    mode="other_sources"
        Original behavior. Keep everything except TF regions strongly supported
        by other sources.

    mode="valid_mask"
        Keep the DAS signal intact outside the global valid/ROI regions. Inside
        valid/ROI regions, keep only regions belonging to the current source and
        suppress valid regions that do not belong to that source.

        In words:
            outside valid_mask -> keep
            inside valid_mask and current source region -> keep
            inside valid_mask and not current source region -> suppress

    Notes
    -----
    The source maps and valid masks used by this pipeline are often on the
    algorithm sub-band grid (bands, frames), not on the reconstruction STFT
    grid. This function projects keep masks by physical frequency and frame
    overlap when mask_nfft/mask_nperseg/mask_hop describe the algorithm grid.
    Reconstruction bins outside the frequency range covered by Omega_list are
    forced to zero.

    taper_freq_bins and taper_time_frames are Gaussian standard deviations on
    the reconstruction STFT grid. The CLI Hz/sec controls are converted to
    these sigma values by treating the requested Hz/sec values as a 3-sigma
    fade radius. The same taper softens the current-source support outward and
    the other-source zero regions outward while preserving the hard zero core.

    If source_delays, mask_reference_mic, target_mics, and mask_hop are provided,
    source masks are shifted inside this function before suppression. This is
    important for mode="other_sources": when cleaning output source sid from
    target mic target_mics[sid], every source mask must be expressed on that same
    target-mic timeline, not on each source's own closest mic timeline.
    """

    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.signal import stft, istft

    taper_freq_bins = float(taper_freq_bins)
    taper_time_frames = float(taper_time_frames)
    if taper_freq_bins < 0 or taper_time_frames < 0:
        raise ValueError("Mask taper sizes must be non-negative")

    def _merge_valid_mask_pairs(valid_mask_in):
        """Convert valid_mask from (pairs,bands,frames) or (bands,frames) to (bands,frames)."""
        valid_arr = np.asarray(valid_mask_in)
        if valid_arr.ndim == 3:
            # valid_mask[pair, band, frame] -> global valid[band, frame]
            return np.any(valid_arr > 0, axis=0)
        if valid_arr.ndim == 2:
            return valid_arr > 0
        raise ValueError(
            f"valid_mask must have shape (pairs,bands,frames) or (bands,frames), got {valid_arr.shape}"
        )

    def _match_time_axis(mask_2d, target_T):
        """Crop or pad the time axis of a 2D mask to target_T frames."""
        mask_2d = np.asarray(mask_2d, dtype=np.float32)
        target_T = int(target_T)
        if mask_2d.shape[1] == target_T:
            return mask_2d
        if mask_2d.shape[1] > target_T:
            return mask_2d[:, :target_T]
        pad = target_T - mask_2d.shape[1]
        return np.pad(mask_2d, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

    def _match_band_time_shape(mask_2d, target_shape, name):
        """Match time length and verify the sub-band dimension."""
        mask_2d = _match_time_axis(mask_2d, target_shape[1])
        if mask_2d.shape[0] != target_shape[0]:
            raise ValueError(
                f"{name} has {mask_2d.shape[0]} bands but target mask has "
                f"{target_shape[0]} bands"
            )
        return mask_2d

    def _expand_band_mask_to_stft(
        mask_bt,
        target_shape,
        *,
        combine="min",
        uncovered_value=0.0,
    ):
        """
        Project a band/time keep mask (B,T) to an STFT-bin/time mask (F,T).

        Time values are sampled from the algorithm frame whose physical center
        is nearest to each reconstruction-frame center. Using the minimum over
        every overlapping frame would substantially enlarge zero regions when
        the algorithm windows are longer than the reconstruction windows.
        """
        mask_bt = np.asarray(mask_bt, dtype=np.float32)
        target_F, target_T = int(target_shape[0]), int(target_shape[1])

        if mask_bt.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {mask_bt.shape}")
        if combine not in ("min", "max"):
            raise ValueError("combine must be 'min' or 'max'")

        if Omega_list is None:
            raise ValueError(
                "Omega_list must be provided when masks are on the sub-band grid "
                f"with shape {mask_bt.shape} and STFT target shape {(target_F, target_T)}"
            )

        if mask_bt.shape[0] != len(Omega_list):
            raise ValueError(
                f"Mask has {mask_bt.shape[0]} bands but Omega_list has {len(Omega_list)} bands"
            )

        source_nfft = int(mask_nfft if mask_nfft is not None else nfft)
        source_nperseg = int(
            mask_nperseg if mask_nperseg is not None else nperseg
        )
        source_hop = int(
            mask_hop if mask_hop is not None else nperseg - noverlap
        )
        target_hop = int(nperseg - noverlap)
        if min(source_nfft, source_nperseg, source_hop, target_hop) <= 0:
            raise ValueError("Mask and reconstruction STFT sizes/hops must be positive")

        source_freqs = np.arange(source_nfft // 2 + 1) * fs / float(source_nfft)
        target_freqs = np.arange(target_F) * fs / float(nfft)
        mask_ft = np.full(
            (target_F, target_T),
            float(uncovered_value),
            dtype=np.float32,
        )
        covered_ft = np.zeros((target_F, target_T), dtype=bool)
        target_centers = (
            np.arange(target_T, dtype=np.float64) * target_hop
            + 0.5 * nperseg
        )
        source_frame_idx = np.rint(
            (target_centers - 0.5 * source_nperseg) / float(source_hop)
        ).astype(int)
        source_frame_idx = np.clip(
            source_frame_idx,
            0,
            mask_bt.shape[1] - 1,
        )

        for b, omega_idx in enumerate(Omega_list):
            omega_idx = np.asarray(omega_idx, dtype=int)
            omega_idx = omega_idx[
                (omega_idx >= 0) & (omega_idx < source_freqs.size)
            ]
            if omega_idx.size == 0:
                continue
            f_lo = source_freqs[omega_idx.min()]
            f_hi = source_freqs[omega_idx.max()]
            target_bins = np.flatnonzero(
                (target_freqs >= f_lo) & (target_freqs <= f_hi)
            )
            if target_bins.size == 0:
                continue

            projected_time = mask_bt[b, source_frame_idx]
            projected = projected_time[None, :]
            target_view = mask_ft[target_bins, :]
            covered_view = covered_ft[target_bins, :]
            if combine == "min":
                target_view[covered_view] = np.minimum(
                    target_view[covered_view],
                    np.broadcast_to(projected, target_view.shape)[covered_view],
                )
                target_view[~covered_view] = np.broadcast_to(
                    projected,
                    target_view.shape,
                )[~covered_view]
            else:
                target_view[covered_view] = np.maximum(
                    target_view[covered_view],
                    np.broadcast_to(projected, target_view.shape)[covered_view],
                )
                target_view[~covered_view] = np.broadcast_to(
                    projected,
                    target_view.shape,
                )[~covered_view]
            covered_view[:, :] = True
            mask_ft[target_bins, :] = target_view
            covered_ft[target_bins, :] = covered_view

        return mask_ft

    def _project_current_source_support(source_region_bt, target_shape):
        """Project and optionally taper only the current source support."""
        source_region_bt = np.asarray(source_region_bt, dtype=np.float32)
        support_ft = _expand_band_mask_to_stft(
            source_region_bt,
            target_shape,
            combine="max",
            uncovered_value=0.0,
        )
        if taper_freq_bins > 0 or taper_time_frames > 0:
            tapered_border = gaussian_filter(
                support_ft,
                sigma=(taper_freq_bins, taper_time_frames),
                mode="nearest",
                truncate=3.0,
            )
            # Preserve the assigned source core at full gain. The Gaussian is
            # used only to add a gradual border outside that core.
            support_ft = np.maximum(support_ft, tapered_border)
        return np.clip(support_ft, 0.0, 1.0).astype(
            np.float32,
            copy=False,
        )

    def _apply_other_source_taper(base_keep, other_region_bt, target_shape):
        """Keep other-source cores at zero and fade smoothly out of them."""
        base_keep = np.asarray(base_keep, dtype=np.float32)
        other_core = _expand_band_mask_to_stft(
            np.asarray(other_region_bt, dtype=np.float32),
            target_shape,
            combine="max",
            uncovered_value=0.0,
        )
        if taper_freq_bins > 0 or taper_time_frames > 0:
            other_fade = gaussian_filter(
                other_core,
                sigma=(taper_freq_bins, taper_time_frames),
                mode="nearest",
                truncate=3.0,
            )
            base_keep = base_keep * (1.0 - np.clip(other_fade, 0.0, 1.0))
        base_keep[other_core >= 0.5] = 0.0
        return np.clip(base_keep, 0.0, 1.0).astype(np.float32, copy=False)

    def _analysis_frequency_coverage(target_shape):
        """Mask of reconstruction bins covered by the analyzed Omega_list bands."""
        coverage_bt = np.ones((len(Omega_list), 1), dtype=np.float32)
        return _expand_band_mask_to_stft(
            coverage_bt,
            target_shape,
            combine="max",
            uncovered_value=0.0,
        )

    def _shift_time_zero_fill(mask_2d, shift_frames):
        """Shift a 2D mask along time with zero fill. Positive means later."""
        mask_2d = np.asarray(mask_2d, dtype=np.float32)
        out = np.zeros_like(mask_2d, dtype=np.float32)
        shift_frames = int(shift_frames)

        if mask_2d.ndim != 2 or mask_2d.shape[1] == 0:
            return out
        if shift_frames == 0:
            return mask_2d.copy()
        if abs(shift_frames) >= mask_2d.shape[1]:
            return out

        if shift_frames > 0:
            out[:, shift_frames:] = mask_2d[:, :-shift_frames]
        else:
            k = -shift_frames
            out[:, :-k] = mask_2d[:, k:]

        return out

    def _shift_source_masks_to_target_mic(masks, target_mic):
        """
        Shift all source masks from mask_reference_mic to target_mic.

        source_delays must have shape (n_sources, n_mics), with the same delay
        convention used elsewhere: tau_ij = d_i - d_j. A mask referenced to mic r
        is shifted to mic q by d_q - d_r.
        """
        if source_delays is None:
            return masks
        if mask_reference_mic is None or mask_hop is None:
            raise ValueError(
                "mask_reference_mic and mask_hop must be provided when source_delays is provided"
            )

        delays = np.asarray(source_delays, dtype=float)
        if delays.ndim != 2:
            raise ValueError("source_delays must have shape (n_sources, n_mics)")
        if delays.shape[0] != len(masks):
            raise ValueError(
                f"source_delays has {delays.shape[0]} sources but there are {len(masks)} masks"
            )

        if isinstance(mask_reference_mic, (int, np.integer)):
            ref_mics = [int(mask_reference_mic)] * len(masks)
        else:
            ref_mics = list(mask_reference_mic)
            if len(ref_mics) != len(masks):
                raise ValueError(
                    f"mask_reference_mic has length {len(ref_mics)} but there are {len(masks)} masks"
                )

        target_mic = int(target_mic)
        shifted = []

        for k, mask_k in enumerate(masks):
            ref_mic_k = ref_mics[k]
            if ref_mic_k is None:
                shifted.append(np.asarray(mask_k, dtype=np.float32).copy())
                continue

            ref_mic_k = int(ref_mic_k)
            shift_samples = float(delays[k, target_mic] - delays[k, ref_mic_k])
            shift_frames = int(np.rint(shift_samples / float(mask_hop)))
            shifted.append(_shift_time_zero_fill(mask_k, shift_frames))

        return shifted

    cleaned = []

    if len(das_signals) == 0:
        return cleaned

    soft_masks = [np.asarray(m, dtype=np.float32) for m in soft_masks]

    if projected_maps is None:
        projected_maps = soft_masks
    else:
        projected_maps = [np.asarray(m, dtype=np.float32) for m in projected_maps]

    if len(soft_masks) != len(das_signals):
        raise ValueError(
            f"Number of soft masks ({len(soft_masks)}) must match number of DAS signals ({len(das_signals)})"
        )

    if len(projected_maps) != len(das_signals):
        raise ValueError(
            f"Number of projected maps ({len(projected_maps)}) must match number of DAS signals ({len(das_signals)})"
        )

    mode = str(mode).lower()

    if source_delays is not None:
        if target_mics is None:
            raise ValueError("target_mics must be provided when source_delays is provided")
        target_mics = [int(m) for m in target_mics]
        if len(target_mics) != len(das_signals):
            raise ValueError(
                f"target_mics has length {len(target_mics)} but das_signals has length {len(das_signals)}"
            )

    valid_bt = None
    if mode == "valid_mask":
        if valid_mask is None:
            raise ValueError("valid_mask must be provided when mode='valid_mask'")
        valid_bt = _merge_valid_mask_pairs(valid_mask).astype(np.float32)
    if mode == "mixed":
        if valid_mask is None:
            raise ValueError("valid_mask must be provided when mode='mixed'")
        valid_bt = _merge_valid_mask_pairs(valid_mask).astype(np.float32)

    shifted_mask_cache = {}
    roi_gate_cache = {}

    for sid, x in enumerate(das_signals):
        _, _, Zxx = stft(
            x,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary=None,
            padded=False,
        )

        if source_delays is not None:
            target_mic = target_mics[sid]
            if target_mic not in shifted_mask_cache:
                shifted_mask_cache[target_mic] = (
                    _shift_source_masks_to_target_mic(soft_masks, target_mic),
                    _shift_source_masks_to_target_mic(projected_maps, target_mic),
                )
            masks_this_target, projected_this_target = shifted_mask_cache[target_mic]
        else:
            masks_this_target = soft_masks
            projected_this_target = projected_maps

        current_roi_gate = None
        if mic_roi_masks_high_res is not None:
            if target_mics is None:
                raise ValueError(
                    "target_mics are required with mic_roi_masks_high_res"
                )
            target_mic = int(target_mics[sid])
            cache_key = (target_mic, Zxx.shape)
            if cache_key not in roi_gate_cache:
                roi_masks_arr = np.asarray(
                    mic_roi_masks_high_res,
                    dtype=bool,
                )
                if roi_masks_arr.ndim != 3:
                    raise ValueError(
                        "mic_roi_masks_high_res must have shape "
                        "(n_mics, freq_bins, frames)"
                    )
                if not 0 <= target_mic < roi_masks_arr.shape[0]:
                    raise ValueError(
                        f"ROI mask is unavailable for microphone {target_mic}"
                    )
                roi_gate_cache[cache_key] = (
                    _project_highres_roi_mask_to_stft(
                        roi_masks_arr[target_mic],
                        fs,
                        Zxx.shape,
                        target_nperseg=nperseg,
                        target_noverlap=noverlap,
                        target_nfft=nfft,
                        roi_nperseg=roi_nperseg,
                        roi_noverlap=roi_noverlap,
                        target_boundary_zeros=False,
                    )
                )
            current_roi_gate = roi_gate_cache[cache_key]

        if mode == "other_sources":
            # Original behavior: suppress regions strongly supported by other sources.
            other_maps = [projected_this_target[k] for k in range(len(projected_this_target)) if k != sid]
            if len(other_maps) == 0:
                keep_mask = np.ones(Zxx.shape, dtype=np.float32)
            else:
                other_stack = np.stack(other_maps, axis=0)
                other_mask_bt = np.max(np.clip(other_stack, 0.0, 1.0), axis=0)

                current_region_bt = masks_this_target[sid] >= threshold
                other_region_bt = other_mask_bt >= threshold
                base_keep_bt = np.ones_like(other_mask_bt, dtype=np.float32)
                base_keep_bt[other_region_bt] = 0.0
                base_keep = _expand_band_mask_to_stft(
                    base_keep_bt,
                    Zxx.shape,
                )
                base_keep = _apply_other_source_taper(
                    base_keep,
                    other_region_bt,
                    Zxx.shape,
                )
                current_support = _project_current_source_support(
                    current_region_bt,
                    Zxx.shape,
                )
                if current_roi_gate is not None:
                    current_support *= current_roi_gate
                keep_mask = np.maximum(base_keep, current_support)

        elif mode == "valid_mask":
            # New behavior requested:
            # outside valid mask -> keep intact
            # inside valid mask -> keep only current source region
            source_region_bt = np.asarray(masks_this_target[sid], dtype=np.float32) > 0
            valid_bool_bt = valid_bt > 0

            if source_region_bt.shape != valid_bool_bt.shape:
                source_region_bt = _match_time_axis(source_region_bt.astype(np.float32), valid_bool_bt.shape[1]) > 0
                if source_region_bt.shape[0] != valid_bool_bt.shape[0]:
                    raise ValueError(
                        f"source map shape {masks_this_target[sid].shape} and merged valid mask shape "
                        f"{valid_bool_bt.shape} have incompatible band dimensions"
                    )

            base_keep_bt = np.ones_like(valid_bool_bt, dtype=np.float32)
            base_keep_bt[valid_bool_bt] = 0.0
            base_keep = _expand_band_mask_to_stft(base_keep_bt, Zxx.shape)
            current_support = _project_current_source_support(
                source_region_bt,
                Zxx.shape,
            )
            if current_roi_gate is not None:
                current_support *= current_roi_gate
            keep_mask = np.maximum(base_keep, current_support)

        elif mode == "mixed":
            # Keep the current source, suppress regions exclusively assigned to
            # other sources, and attenuate valid but unassigned regions.
            valid_bool_bt = valid_bt > 0
            current_map_bt = _match_band_time_shape(
                masks_this_target[sid],
                valid_bool_bt.shape,
                "current source mask",
            )
            current_region_bt = current_map_bt >= threshold

            other_maps = [
                _match_band_time_shape(
                    projected_this_target[k],
                    valid_bool_bt.shape,
                    f"projected source mask {k}",
                )
                for k in range(len(projected_this_target))
                if k != sid
            ]
            if other_maps:
                other_mask_bt = np.max(
                    np.clip(np.stack(other_maps, axis=0), 0.0, 1.0),
                    axis=0,
                )
                other_region_bt = other_mask_bt >= threshold
            else:
                other_region_bt = np.zeros_like(valid_bool_bt)

            base_keep_bt = np.full(
                valid_bool_bt.shape,
                0.1,
                dtype=np.float32,
            )
            # base_keep_bt[valid_bool_bt & (~other_region_bt) & (~current_region_bt)] = 0.1
            # base_keep_bt[(~current_region_bt) & (~valid_bool_bt)] = 0.1
            base_keep_bt[(~current_region_bt)] = 0.1
            base_keep_bt[other_region_bt] = 0.0
            base_keep = _expand_band_mask_to_stft(
                base_keep_bt,
                Zxx.shape,
            )
            base_keep = _apply_other_source_taper(
                base_keep,
                other_region_bt,
                Zxx.shape,
            )
            current_support = _project_current_source_support(
                current_region_bt,
                Zxx.shape,
            )
            if current_roi_gate is not None:
                current_support *= current_roi_gate
            keep_mask = np.maximum(base_keep, current_support)


        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Use 'other_sources' or 'valid_mask'."
            )

        keep_mask *= _analysis_frequency_coverage(Zxx.shape)

        if keep_mask.shape != Zxx.shape:
            raise ValueError(
                f"keep_mask has shape {keep_mask.shape}, but DAS STFT has shape {Zxx.shape}"
            )

        _, x_clean = istft(
            Zxx * keep_mask,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            input_onesided=True,
            boundary=False,
        )

        cleaned.append(x_clean.astype(np.float32))

    return cleaned


def soft_suppress_other_source_regions_from_das(
    das_signals,
    soft_masks,
    fs,
    nperseg,
    noverlap,
    nfft,
    threshold=None,
    mode=None,
    projected_maps=None,
    valid_mask=None,
    Omega_list=None,
    source_delays=None,
    mask_reference_mic=None,
    target_mics=None,
    mask_hop=None,
    mask_nperseg=None,
    mask_nfft=None,
    min_gain=0.18,
    unassigned_gain=0.85,
    competition_power=1.5,
    mask_smooth_freq_bins=1.0,
    mask_smooth_time_frames=1.0,
    taper_freq_bins=None,
    taper_time_frames=None,
    outside_valid_gain=1.0,
    preserve_current=True,
    eps=1e-6,
):
    """
    Softly separate source DAS signals using current-source and other-source masks.

    This is a smoother alternative to suppress_other_source_regions_from_das().
    It avoids binary zeroing: inside valid/ROI regions, each TF bin receives a
    soft gain based on how strongly the current source wins over the strongest
    other source. Regions with no confident assignment are only mildly reduced.

    The default behavior is intentionally classifier-friendly:
        - current source support stays near full gain,
        - competing-source regions are attenuated to min_gain, not zeroed,
        - unassigned valid regions keep unassigned_gain,
        - the final STFT-domain gain is Gaussian-smoothed.

    source_delays/mask_reference_mic/target_mics follow the same alignment
    convention as suppress_other_source_regions_from_das().

    threshold, mode, taper_freq_bins, and taper_time_frames are accepted for
    near drop-in replacement of the hard-mask function. The first two are not
    used; taper values are used as smoothing sizes when explicit
    mask_smooth_* values are not provided.
    """

    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.signal import stft, istft

    min_gain = float(min_gain)
    unassigned_gain = float(unassigned_gain)
    competition_power = float(competition_power)
    if taper_freq_bins is not None and mask_smooth_freq_bins == 1.0:
        mask_smooth_freq_bins = taper_freq_bins
    if taper_time_frames is not None and mask_smooth_time_frames == 1.0:
        mask_smooth_time_frames = taper_time_frames
    mask_smooth_freq_bins = float(mask_smooth_freq_bins)
    mask_smooth_time_frames = float(mask_smooth_time_frames)
    outside_valid_gain = float(outside_valid_gain)
    eps = float(eps)

    if not (0.0 <= min_gain <= 1.0):
        raise ValueError("min_gain must be in [0, 1]")
    if not (0.0 <= unassigned_gain <= 1.0):
        raise ValueError("unassigned_gain must be in [0, 1]")
    if not (0.0 <= outside_valid_gain <= 1.0):
        raise ValueError("outside_valid_gain must be in [0, 1]")
    if competition_power <= 0:
        raise ValueError("competition_power must be positive")
    if mask_smooth_freq_bins < 0 or mask_smooth_time_frames < 0:
        raise ValueError("Mask smoothing sizes must be non-negative")

    def _merge_valid_mask_pairs(valid_mask_in):
        valid_arr = np.asarray(valid_mask_in)
        if valid_arr.ndim == 3:
            return np.any(valid_arr > 0, axis=0)
        if valid_arr.ndim == 2:
            return valid_arr > 0
        raise ValueError(
            f"valid_mask must have shape (pairs,bands,frames) or (bands,frames), got {valid_arr.shape}"
        )

    def _match_time_axis(mask_2d, target_T):
        mask_2d = np.asarray(mask_2d, dtype=np.float32)
        target_T = int(target_T)
        if mask_2d.shape[1] == target_T:
            return mask_2d
        if mask_2d.shape[1] > target_T:
            return mask_2d[:, :target_T]
        pad = target_T - mask_2d.shape[1]
        return np.pad(mask_2d, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

    def _match_band_time_shape(mask_2d, target_shape, name):
        mask_2d = _match_time_axis(mask_2d, target_shape[1])
        if mask_2d.shape[0] != target_shape[0]:
            raise ValueError(
                f"{name} has {mask_2d.shape[0]} bands but target mask has "
                f"{target_shape[0]} bands"
            )
        return np.clip(mask_2d, 0.0, 1.0).astype(np.float32, copy=False)

    def _expand_band_mask_to_stft(mask_bt, target_shape, *, uncovered_value=0.0):
        mask_bt = np.asarray(mask_bt, dtype=np.float32)
        target_F, target_T = int(target_shape[0]), int(target_shape[1])

        if mask_bt.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {mask_bt.shape}")
        if Omega_list is None:
            raise ValueError(
                "Omega_list must be provided when masks are on the sub-band grid "
                f"with shape {mask_bt.shape} and STFT target shape {(target_F, target_T)}"
            )
        if mask_bt.shape[0] != len(Omega_list):
            raise ValueError(
                f"Mask has {mask_bt.shape[0]} bands but Omega_list has {len(Omega_list)} bands"
            )

        source_nfft = int(mask_nfft if mask_nfft is not None else nfft)
        source_nperseg = int(mask_nperseg if mask_nperseg is not None else nperseg)
        source_hop = int(mask_hop if mask_hop is not None else nperseg - noverlap)
        target_hop = int(nperseg - noverlap)
        if min(source_nfft, source_nperseg, source_hop, target_hop) <= 0:
            raise ValueError("Mask and reconstruction STFT sizes/hops must be positive")

        source_freqs = np.arange(source_nfft // 2 + 1) * fs / float(source_nfft)
        target_freqs = np.arange(target_F) * fs / float(nfft)
        mask_ft = np.full((target_F, target_T), float(uncovered_value), dtype=np.float32)
        covered_ft = np.zeros((target_F, target_T), dtype=bool)

        target_centers = (
            np.arange(target_T, dtype=np.float64) * target_hop
            + 0.5 * nperseg
        )
        source_frame_idx = np.rint(
            (target_centers - 0.5 * source_nperseg) / float(source_hop)
        ).astype(int)
        source_frame_idx = np.clip(source_frame_idx, 0, mask_bt.shape[1] - 1)

        for b, omega_idx in enumerate(Omega_list):
            omega_idx = np.asarray(omega_idx, dtype=int)
            omega_idx = omega_idx[
                (omega_idx >= 0) & (omega_idx < source_freqs.size)
            ]
            if omega_idx.size == 0:
                continue

            f_lo = source_freqs[omega_idx.min()]
            f_hi = source_freqs[omega_idx.max()]
            target_bins = np.flatnonzero(
                (target_freqs >= f_lo) & (target_freqs <= f_hi)
            )
            if target_bins.size == 0:
                continue

            projected = mask_bt[b, source_frame_idx][None, :]
            projected = np.broadcast_to(projected, (target_bins.size, target_T))
            target_view = mask_ft[target_bins, :]
            covered_view = covered_ft[target_bins, :]
            target_view[covered_view] = np.maximum(
                target_view[covered_view],
                projected[covered_view],
            )
            target_view[~covered_view] = projected[~covered_view]
            covered_view[:, :] = True
            mask_ft[target_bins, :] = target_view
            covered_ft[target_bins, :] = covered_view

        return mask_ft

    def _analysis_frequency_coverage(target_shape):
        coverage_bt = np.ones((len(Omega_list), 1), dtype=np.float32)
        return _expand_band_mask_to_stft(
            coverage_bt,
            target_shape,
            uncovered_value=0.0,
        )

    def _shift_time_zero_fill(mask_2d, shift_frames):
        mask_2d = np.asarray(mask_2d, dtype=np.float32)
        out = np.zeros_like(mask_2d, dtype=np.float32)
        shift_frames = int(shift_frames)

        if mask_2d.ndim != 2 or mask_2d.shape[1] == 0:
            return out
        if shift_frames == 0:
            return mask_2d.copy()
        if abs(shift_frames) >= mask_2d.shape[1]:
            return out
        if shift_frames > 0:
            out[:, shift_frames:] = mask_2d[:, :-shift_frames]
        else:
            k = -shift_frames
            out[:, :-k] = mask_2d[:, k:]
        return out

    def _shift_source_masks_to_target_mic(masks, target_mic):
        if source_delays is None:
            return masks
        if mask_reference_mic is None or mask_hop is None:
            raise ValueError(
                "mask_reference_mic and mask_hop must be provided when source_delays is provided"
            )

        delays = np.asarray(source_delays, dtype=float)
        if delays.ndim != 2:
            raise ValueError("source_delays must have shape (n_sources, n_mics)")
        if delays.shape[0] != len(masks):
            raise ValueError(
                f"source_delays has {delays.shape[0]} sources but there are {len(masks)} masks"
            )

        if isinstance(mask_reference_mic, (int, np.integer)):
            ref_mics = [int(mask_reference_mic)] * len(masks)
        else:
            ref_mics = list(mask_reference_mic)
            if len(ref_mics) != len(masks):
                raise ValueError(
                    f"mask_reference_mic has length {len(ref_mics)} but there are {len(masks)} masks"
                )

        target_mic = int(target_mic)
        shifted = []
        for k, mask_k in enumerate(masks):
            ref_mic_k = ref_mics[k]
            if ref_mic_k is None:
                shifted.append(np.asarray(mask_k, dtype=np.float32).copy())
                continue
            shift_samples = float(delays[k, target_mic] - delays[k, int(ref_mic_k)])
            shift_frames = int(np.rint(shift_samples / float(mask_hop)))
            shifted.append(_shift_time_zero_fill(mask_k, shift_frames))
        return shifted

    def _soft_gain_from_masks(current_bt, other_bt, valid_bt):
        current_bt = np.clip(current_bt, 0.0, 1.0).astype(np.float32, copy=False)
        other_bt = np.clip(other_bt, 0.0, 1.0).astype(np.float32, copy=False)
        valid_bool = valid_bt > 0

        c = current_bt ** competition_power
        o = other_bt ** competition_power
        ratio = c / (c + o + eps)

        assigned = (current_bt + other_bt) > eps
        gain_bt = np.full(valid_bool.shape, outside_valid_gain, dtype=np.float32)
        gain_valid = min_gain + (1.0 - min_gain) * ratio
        gain_valid[~assigned] = unassigned_gain
        if preserve_current:
            gain_valid = np.maximum(gain_valid, current_bt)
        gain_bt[valid_bool] = gain_valid[valid_bool]
        return np.clip(gain_bt, min_gain, 1.0).astype(np.float32, copy=False)

    cleaned = []
    if len(das_signals) == 0:
        return cleaned

    soft_masks = [np.asarray(m, dtype=np.float32) for m in soft_masks]
    if projected_maps is None:
        projected_maps = soft_masks
    else:
        projected_maps = [np.asarray(m, dtype=np.float32) for m in projected_maps]

    if len(soft_masks) != len(das_signals):
        raise ValueError(
            f"Number of soft masks ({len(soft_masks)}) must match number of DAS signals ({len(das_signals)})"
        )
    if len(projected_maps) != len(das_signals):
        raise ValueError(
            f"Number of projected maps ({len(projected_maps)}) must match number of DAS signals ({len(das_signals)})"
        )

    if source_delays is not None:
        if target_mics is None:
            raise ValueError("target_mics must be provided when source_delays is provided")
        target_mics = [int(m) for m in target_mics]
        if len(target_mics) != len(das_signals):
            raise ValueError(
                f"target_mics has length {len(target_mics)} but das_signals has length {len(das_signals)}"
            )

    if valid_mask is None:
        valid_bt = np.ones_like(soft_masks[0], dtype=np.float32)
    else:
        valid_bt = _merge_valid_mask_pairs(valid_mask).astype(np.float32)

    shifted_mask_cache = {}

    for sid, x in enumerate(das_signals):
        _, _, Zxx = stft(
            x,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary=None,
            padded=False,
        )

        if source_delays is not None:
            target_mic = target_mics[sid]
            if target_mic not in shifted_mask_cache:
                shifted_mask_cache[target_mic] = (
                    _shift_source_masks_to_target_mic(soft_masks, target_mic),
                    _shift_source_masks_to_target_mic(projected_maps, target_mic),
                )
            masks_this_target, projected_this_target = shifted_mask_cache[target_mic]
        else:
            masks_this_target = soft_masks
            projected_this_target = projected_maps

        valid_this_bt = _match_band_time_shape(valid_bt, masks_this_target[sid].shape, "valid mask")
        current_bt = _match_band_time_shape(
            masks_this_target[sid],
            valid_this_bt.shape,
            "current source mask",
        )
        other_maps = [
            _match_band_time_shape(
                projected_this_target[k],
                valid_this_bt.shape,
                f"projected source mask {k}",
            )
            for k in range(len(projected_this_target))
            if k != sid
        ]
        if other_maps:
            other_bt = np.max(np.stack(other_maps, axis=0), axis=0)
        else:
            other_bt = np.zeros_like(current_bt, dtype=np.float32)

        gain_bt = _soft_gain_from_masks(current_bt, other_bt, valid_this_bt)
        keep_mask = _expand_band_mask_to_stft(
            gain_bt,
            Zxx.shape,
            uncovered_value=0.0,
        )

        if mask_smooth_freq_bins > 0 or mask_smooth_time_frames > 0:
            keep_mask = gaussian_filter(
                keep_mask,
                sigma=(mask_smooth_freq_bins, mask_smooth_time_frames),
                mode="nearest",
                truncate=3.0,
            )
            keep_mask = np.clip(keep_mask, min_gain, 1.0)

        keep_mask *= _analysis_frequency_coverage(Zxx.shape)

        if keep_mask.shape != Zxx.shape:
            raise ValueError(
                f"keep_mask has shape {keep_mask.shape}, but DAS STFT has shape {Zxx.shape}"
            )

        _, x_clean = istft(
            Zxx * keep_mask,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            input_onesided=True,
            boundary=False,
        )

        cleaned.append(x_clean.astype(np.float32))

    return cleaned


def _estimate_inside_mask_noise_db(
    original_stft,
    source_weight,
    *,
    min_core_weight=0.8,
    fallback_percentile=30.0,
    max_samples=100_000,
    eps=1e-12,
):
    """Estimate the background-noise peak in log-power inside a source mask."""
    from scipy.ndimage import uniform_filter

    # A raw complex-Gaussian STFT has a strongly skewed exponential power
    # distribution. Local averaging makes one stationary noise population
    # compact enough that a two-component fit does not manufacture a second
    # peak merely from that skew.
    power = uniform_filter(
        np.abs(np.asarray(original_stft)) ** 2,
        size=(3, 3),
        mode="nearest",
    )
    weight = np.asarray(source_weight, dtype=float)
    if power.shape != weight.shape:
        raise ValueError("original_stft and source_weight must have equal shapes")

    peak_weight = float(np.max(weight)) if weight.size else 0.0
    if peak_weight <= eps:
        return None

    # Stay in the mask core and away from the Gaussian crossfade. For masks
    # whose maximum is below 0.8 (for example, competing soft masks), use 90%
    # of their own maximum instead.
    core_threshold = min(float(min_core_weight), 0.9 * peak_weight)
    core = weight >= max(core_threshold, eps)
    values_db = 10.0 * np.log10(np.maximum(power[core], eps))
    values_db = values_db[np.isfinite(values_db)]
    if values_db.size < 16:
        return None

    if values_db.size > int(max_samples):
        indices = np.linspace(
            0,
            values_db.size - 1,
            int(max_samples),
            dtype=int,
        )
        values_db = values_db[indices]

    q01, q99 = np.percentile(values_db, (1.0, 99.0))
    fitted_values = np.clip(values_db, q01, q99)
    means = np.asarray(
        np.percentile(fitted_values, (30.0, 75.0)),
        dtype=float,
    )
    overall_variance = max(float(np.var(fitted_values)), 4.0)
    variances = np.full(2, overall_variance, dtype=float)
    mixture_weights = np.asarray([0.65, 0.35], dtype=float)

    responsibilities = None
    for _ in range(60):
        log_prob = np.empty((fitted_values.size, 2), dtype=float)
        for component in range(2):
            variance = max(float(variances[component]), 1.0)
            log_prob[:, component] = (
                np.log(max(float(mixture_weights[component]), eps))
                - 0.5 * np.log(2.0 * np.pi * variance)
                - 0.5
                * (fitted_values - means[component]) ** 2
                / variance
            )
        log_prob -= np.max(log_prob, axis=1, keepdims=True)
        responsibilities = np.exp(log_prob)
        responsibilities /= np.maximum(
            np.sum(responsibilities, axis=1, keepdims=True),
            eps,
        )
        component_mass = np.sum(responsibilities, axis=0)
        if np.any(component_mass <= eps):
            responsibilities = None
            break

        new_means = (
            np.sum(responsibilities * fitted_values[:, None], axis=0)
            / component_mass
        )
        new_variances = np.sum(
            responsibilities
            * (fitted_values[:, None] - new_means[None, :]) ** 2,
            axis=0,
        ) / component_mass
        new_variances = np.maximum(new_variances, 1.0)
        new_weights = component_mass / float(fitted_values.size)

        if np.max(np.abs(new_means - means)) < 1e-3:
            means = new_means
            variances = new_variances
            mixture_weights = new_weights
            break
        means = new_means
        variances = new_variances
        mixture_weights = new_weights

    noise_component = int(np.argmin(means))
    signal_component = 1 - noise_component
    separated = (
        responsibilities is not None
        and means[signal_component] - means[noise_component] >= 3.0
        and mixture_weights[noise_component] >= 0.05
        and mixture_weights[signal_component] >= 0.05
    )
    if not separated:
        return {
            "noise_db": float(
                np.percentile(values_db, float(fallback_percentile))
            ),
            "method": f"percentile_{float(fallback_percentile):g}",
            "n_bins": int(values_db.size),
        }

    noise_probability = responsibilities[:, noise_component]
    order = np.argsort(fitted_values)
    sorted_values = fitted_values[order]
    sorted_weights = noise_probability[order]
    cumulative = np.cumsum(sorted_weights)
    midpoint = 0.5 * float(cumulative[-1])
    weighted_median_idx = min(
        int(np.searchsorted(cumulative, midpoint)),
        sorted_values.size - 1,
    )
    return {
        "noise_db": float(sorted_values[weighted_median_idx]),
        "method": "two_component_gmm",
        "n_bins": int(values_db.size),
        "noise_peak_db": float(means[noise_component]),
        "signal_peak_db": float(means[signal_component]),
    }


def fill_outside_source_mask_with_white_noise(
    source_signals,
    source_masks,
    fs,
    nperseg,
    noverlap,
    nfft,
    *,
    Omega_list,
    source_delays=None,
    mask_reference_mic=None,
    target_mics=None,
    mask_hop=None,
    mask_nperseg=None,
    mask_nfft=None,
    mask_threshold=0.0,
    border_smooth_freq_bins=2.0,
    border_smooth_time_frames=1.0,
    noise_level=1.0,
    target_snr_db=None,
    replacement_mode="white_noise",
    white_noise_level_mode="target_snr",
    mic_roi_masks_high_res=None,
    roi_nperseg=1024,
    roi_noverlap=512,
    random_state=0,
    eps=1e-12,
    fallback_source_masks=None,
    fallback_source_mask_allowed=None,
    return_diagnostics=False,
):
    """
    Preserve each source mask and replace everything outside it.

    The source mask is aligned to the signal's target microphone, projected
    from the algorithm's band/time grid to the reconstruction STFT, and
    Gaussian-smoothed at its border.

    ``replacement_mode="white_noise"`` uses white noise with an equal-power
    crossfade. Its level can either follow ``target_snr_db`` or match the
    background-noise population measured inside the source mask.
    ``"scaled_original"`` keeps the original signal
    outside the mask at ``noise_level`` gain and interpolates that gain to 1
    inside the mask. When ``target_snr_db`` is provided, it replaces
    ``noise_level`` and sets the per-source ratio between preserved
    source-component power and the replacement component. For
    ``scaled_original`` this is technically a source-to-residual ratio rather
    than an independent-noise SNR.

    Parameters
    ----------
    source_signals : sequence of 1D arrays
        One real-microphone or reconstructed signal per source.
    source_masks : sequence of 2D arrays
        One source-only mask per signal, shaped (bands, frames).
    border_smooth_freq_bins, border_smooth_time_frames : float
        Gaussian sigma on the reconstruction STFT grid. Zero keeps the original
        mask edge.
    noise_level : float
        White-noise level or outside-mask original-signal gain, depending on
        ``replacement_mode``.
    target_snr_db : float or None
        Desired source-to-injected-noise power ratio in dB for each output.
        In ``scaled_original`` mode, this means source-to-residual-original
        power ratio. If None, use ``noise_level``. It is ignored when white
        noise uses ``white_noise_level_mode="inside_noise_floor"``.
    replacement_mode : {"white_noise", "scaled_original"}
        Content used outside the source mask.
    white_noise_level_mode : {"target_snr", "inside_noise_floor"}
        In white-noise mode, either retain the previous target-SNR/noise-level
        behavior or estimate the lower-energy population of log-STFT powers
        inside the mask and match the generated white noise to it.
    mic_roi_masks_high_res : ndarray or None
        Binary MAAD masks shaped (n_mics, freq_bins, frames). When provided,
        the mask from each source's target microphone hard-gates its projected
        source support before replacement.
    fallback_source_masks : sequence of 2D arrays or None
        Optional ungated masks. In ``inside_noise_floor`` mode, a source uses
        its corresponding fallback mask when its primary mask contains fewer
        than the minimum number of core bins required for noise estimation.
        The fallback mask then drives both calibration and reconstruction.
    fallback_source_mask_allowed : sequence of bool or None
        Optional per-source permission for fallback. This prevents a naturally
        sparse mask with no rejected TF contribution from being mislabeled as
        a contamination fallback.
    return_diagnostics : bool
        If True, return ``(outputs, diagnostics)``. Diagnostics report whether
        each source fell back to its original mask.
    random_state : int or None
        Reproducible base seed. Use None for different noise on every call.
    """

    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.signal import istft, stft

    source_signals = list(source_signals)
    source_masks = [
        np.asarray(mask, dtype=np.float32)
        for mask in source_masks
    ]
    if fallback_source_masks is not None:
        fallback_source_masks = [
            np.asarray(mask, dtype=np.float32)
            for mask in fallback_source_masks
        ]
        if len(fallback_source_masks) != len(source_masks):
            raise ValueError(
                "fallback_source_masks must contain one mask per source"
            )
    if fallback_source_mask_allowed is None:
        fallback_source_mask_allowed = [
            fallback_source_masks is not None
        ] * len(source_masks)
    else:
        fallback_source_mask_allowed = [
            bool(value) for value in fallback_source_mask_allowed
        ]
        if len(fallback_source_mask_allowed) != len(source_masks):
            raise ValueError(
                "fallback_source_mask_allowed must contain one value per source"
            )
    if len(source_signals) != len(source_masks):
        raise ValueError(
            f"Number of source signals ({len(source_signals)}) must match "
            f"number of source masks ({len(source_masks)})"
        )
    if len(source_signals) == 0:
        return ([], []) if return_diagnostics else []
    if Omega_list is None:
        raise ValueError("Omega_list must be provided")
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")
    if replacement_mode not in {"white_noise", "scaled_original"}:
        raise ValueError(
            "replacement_mode must be 'white_noise' or 'scaled_original'"
        )
    if white_noise_level_mode not in {"target_snr", "inside_noise_floor"}:
        raise ValueError(
            "white_noise_level_mode must be 'target_snr' or "
            "'inside_noise_floor'"
        )
    if target_snr_db is not None:
        target_snr_db = float(target_snr_db)
        if not np.isfinite(target_snr_db):
            raise ValueError("target_snr_db must be finite")
    if mask_threshold < 0 or mask_threshold > 1:
        raise ValueError("mask_threshold must be in [0, 1]")
    if border_smooth_freq_bins < 0 or border_smooth_time_frames < 0:
        raise ValueError("Border smoothing sizes must be non-negative")
    if mic_roi_masks_high_res is not None:
        mic_roi_masks_high_res = np.asarray(
            mic_roi_masks_high_res,
            dtype=bool,
        )
        if mic_roi_masks_high_res.ndim != 3:
            raise ValueError(
                "mic_roi_masks_high_res must have shape "
                "(n_mics, freq_bins, frames)"
            )
        if target_mics is None:
            raise ValueError(
                "target_mics are required with mic_roi_masks_high_res"
            )

    reconstruction_hop = int(nperseg - noverlap)
    source_hop = int(mask_hop if mask_hop is not None else reconstruction_hop)
    source_nperseg = int(
        mask_nperseg if mask_nperseg is not None else nperseg
    )
    source_nfft = int(mask_nfft if mask_nfft is not None else nfft)
    if min(
        reconstruction_hop,
        source_hop,
        source_nperseg,
        source_nfft,
    ) <= 0:
        raise ValueError("STFT sizes and hops must be positive")

    if source_delays is not None:
        delays = np.asarray(source_delays, dtype=float)
        if delays.ndim != 2 or delays.shape[0] != len(source_masks):
            raise ValueError(
                "source_delays must have shape (n_sources, n_mics)"
            )
        if target_mics is None or mask_reference_mic is None:
            raise ValueError(
                "target_mics and mask_reference_mic are required when "
                "source_delays is provided"
            )
        target_mics = [int(mic) for mic in target_mics]
        if len(target_mics) != len(source_masks):
            raise ValueError(
                "target_mics must contain one microphone per source"
            )
        if isinstance(mask_reference_mic, (int, np.integer)):
            reference_mics = [int(mask_reference_mic)] * len(source_masks)
        else:
            reference_mics = list(mask_reference_mic)
            if len(reference_mics) != len(source_masks):
                raise ValueError(
                    "mask_reference_mic must contain one microphone per source"
                )
    else:
        delays = None
        reference_mics = None

    source_freqs = (
        np.arange(source_nfft // 2 + 1, dtype=float)
        * fs
        / float(source_nfft)
    )

    def _shift_time_zero_fill(mask, shift_frames):
        shifted = np.zeros_like(mask, dtype=np.float32)
        shift_frames = int(shift_frames)
        if shift_frames == 0:
            return mask.copy()
        if abs(shift_frames) >= mask.shape[1]:
            return shifted
        if shift_frames > 0:
            shifted[:, shift_frames:] = mask[:, :-shift_frames]
        else:
            shift = -shift_frames
            shifted[:, :-shift] = mask[:, shift:]
        return shifted

    def _aligned_mask(source_idx, masks):
        mask = masks[source_idx]
        if mask.ndim != 2 or mask.shape[1] == 0:
            raise ValueError(
                f"Source mask {source_idx} must be a non-empty 2D array"
            )
        if mask.shape[0] != len(Omega_list):
            raise ValueError(
                f"Source mask {source_idx} has {mask.shape[0]} bands but "
                f"Omega_list has {len(Omega_list)}"
            )
        if delays is None:
            return mask

        reference_mic = reference_mics[source_idx]
        if reference_mic is None:
            return mask
        target_mic = target_mics[source_idx]
        shift_samples = float(
            delays[source_idx, target_mic]
            - delays[source_idx, int(reference_mic)]
        )
        shift_frames = int(np.rint(shift_samples / float(source_hop)))
        return _shift_time_zero_fill(mask, shift_frames)

    def _project_mask(mask_bt, target_shape):
        target_freq_bins, target_frames = target_shape
        target_freqs = (
            np.arange(target_freq_bins, dtype=float)
            * fs
            / float(nfft)
        )
        target_centers = (
            np.arange(target_frames, dtype=float) * reconstruction_hop
        )
        source_frame_idx = np.rint(
            (target_centers - 0.5 * source_nperseg)
            / float(source_hop)
        ).astype(int)
        source_frame_idx = np.clip(
            source_frame_idx,
            0,
            mask_bt.shape[1] - 1,
        )

        projected = np.zeros(target_shape, dtype=np.float32)
        for band_idx, omega_idx in enumerate(Omega_list):
            omega_idx = np.asarray(omega_idx, dtype=int)
            omega_idx = omega_idx[
                (omega_idx >= 0) & (omega_idx < source_freqs.size)
            ]
            if omega_idx.size == 0:
                continue
            low_frequency = source_freqs[omega_idx.min()]
            high_frequency = source_freqs[omega_idx.max()]
            target_bins = np.flatnonzero(
                (target_freqs >= low_frequency)
                & (target_freqs <= high_frequency)
            )
            if target_bins.size == 0:
                continue
            band_values = mask_bt[band_idx, source_frame_idx][None, :]
            projected[target_bins, :] = np.maximum(
                projected[target_bins, :],
                band_values,
            )
        return np.clip(projected, 0.0, 1.0)

    outputs = []
    diagnostics = []
    for source_idx, signal in enumerate(source_signals):
        source_diagnostic = {
            "source_idx": int(source_idx),
            "mask_fallback_used": False,
            "mask_contamination_possible": False,
            "mask_fallback_reason": None,
        }
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        if signal.size == 0:
            outputs.append(signal.copy())
            diagnostics.append(source_diagnostic)
            continue

        _, _, original_stft = stft(
            signal,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary="zeros",
            padded=True,
        )

        def _source_weight_from_masks(masks):
            mask_bt = np.clip(
                _aligned_mask(source_idx, masks),
                0.0,
                1.0,
            )
            if mask_threshold > 0:
                mask_bt = np.where(
                    mask_bt >= mask_threshold,
                    mask_bt,
                    0.0,
                )
            weight = _project_mask(mask_bt, original_stft.shape)
            if (
                border_smooth_freq_bins > 0
                or border_smooth_time_frames > 0
            ):
                border = gaussian_filter(
                    weight,
                    sigma=(
                        float(border_smooth_freq_bins),
                        float(border_smooth_time_frames),
                    ),
                    mode="nearest",
                    truncate=3.0,
                )
                weight = np.maximum(weight, border)
                weight = np.clip(weight, 0.0, 1.0)

            if mic_roi_masks_high_res is not None:
                target_mic = int(target_mics[source_idx])
                if not 0 <= target_mic < mic_roi_masks_high_res.shape[0]:
                    raise ValueError(
                        f"ROI mask is unavailable for microphone {target_mic}"
                    )
                roi_gate = _project_highres_roi_mask_to_stft(
                    mic_roi_masks_high_res[target_mic],
                    fs,
                    original_stft.shape,
                    target_nperseg=nperseg,
                    target_noverlap=noverlap,
                    target_nfft=nfft,
                    roi_nperseg=roi_nperseg,
                    roi_noverlap=roi_noverlap,
                    target_boundary_zeros=True,
                )
                weight *= roi_gate
            return weight

        source_weight = _source_weight_from_masks(source_masks)

        if replacement_mode == "scaled_original":
            source_component = source_weight * original_stft
            residual_component = (1.0 - source_weight) * original_stft
            if target_snr_db is None:
                residual_scale = float(noise_level)
            else:
                source_power = float(
                    np.sum(np.abs(source_component) ** 2)
                )
                unscaled_residual_power = float(
                    np.sum(np.abs(residual_component) ** 2)
                )
                target_linear = float(
                    np.exp(
                        np.clip(
                            target_snr_db * np.log(10.0) / 10.0,
                            -700.0,
                            700.0,
                        )
                    )
                )
                residual_scale = np.sqrt(
                    max(source_power, 0.0)
                    / max(
                        target_linear * unscaled_residual_power,
                        eps,
                    )
                )
            output_stft = (
                source_component
                + residual_scale * residual_component
            )
        else:
            seed = (
                None
                if random_state is None
                else int(random_state) + source_idx
            )
            rng = np.random.default_rng(seed)
            white_noise = rng.standard_normal(signal.size).astype(np.float32)
            _, _, noise_stft = stft(
                white_noise,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                boundary="zeros",
                padded=True,
            )

            if white_noise_level_mode == "inside_noise_floor":
                estimate = _estimate_inside_mask_noise_db(
                    original_stft,
                    source_weight,
                    eps=eps,
                )
                if estimate is None:
                    if (
                        fallback_source_masks is None
                        or not fallback_source_mask_allowed[source_idx]
                    ):
                        raise ValueError(
                            "Cannot estimate the inside-mask noise floor for "
                            f"source {source_idx}: the mask has too few core bins"
                        )
                    fallback_weight = _source_weight_from_masks(
                        fallback_source_masks
                    )
                    fallback_estimate = _estimate_inside_mask_noise_db(
                        original_stft,
                        fallback_weight,
                        eps=eps,
                    )
                    if fallback_estimate is None:
                        raise ValueError(
                            "Cannot estimate the inside-mask noise floor for "
                            f"source {source_idx}: both the rejected and "
                            "original masks have too few core bins"
                        )
                    source_weight = fallback_weight
                    estimate = fallback_estimate
                    source_diagnostic.update({
                        "mask_fallback_used": True,
                        "mask_contamination_possible": True,
                        "mask_fallback_reason": (
                            "ambiguous_pair_rejection_left_too_few_core_bins"
                        ),
                    })

                noise_weight = np.sqrt(
                    np.maximum(0.0, 1.0 - source_weight ** 2)
                )
                replacement_region = noise_weight >= 0.9
                if np.count_nonzero(replacement_region) < 32:
                    replacement_region = noise_weight > 0.0
                from scipy.ndimage import uniform_filter

                generated_noise_power = uniform_filter(
                    np.abs(noise_stft) ** 2,
                    size=(3, 3),
                    mode="nearest",
                )
                reference_noise_db = 10.0 * np.log10(
                    np.maximum(
                        generated_noise_power[replacement_region],
                        eps,
                    )
                )
                reference_noise_db = reference_noise_db[
                    np.isfinite(reference_noise_db)
                ]
                if reference_noise_db.size < 32:
                    raise ValueError(
                        "Cannot calibrate generated white noise for source "
                        f"{source_idx}: too few outside-mask bins"
                    )
                generated_median_db = float(np.median(reference_noise_db))
                noise_scale = float(
                    10.0
                    ** (
                        (
                            float(estimate["noise_db"])
                            - generated_median_db
                        )
                        / 20.0
                    )
                )
            else:
                noise_weight = np.sqrt(
                    np.maximum(0.0, 1.0 - source_weight ** 2)
                )

            if (
                white_noise_level_mode != "inside_noise_floor"
                and target_snr_db is None
            ):
                transition_weight = (
                    4.0 * source_weight * (1.0 - source_weight)
                )
                if float(np.sum(transition_weight)) <= eps:
                    transition_weight = np.maximum(
                        source_weight,
                        1.0 - source_weight,
                    )
                original_power = np.sum(
                    transition_weight * np.abs(original_stft) ** 2
                ) / max(float(np.sum(transition_weight)), eps)
                noise_power = np.sum(
                    transition_weight * np.abs(noise_stft) ** 2
                ) / max(float(np.sum(transition_weight)), eps)
                noise_scale = float(noise_level) * np.sqrt(
                    max(float(original_power), 0.0)
                    / max(float(noise_power), eps)
                )
            elif white_noise_level_mode != "inside_noise_floor":
                source_component = source_weight * original_stft
                noise_component = noise_weight * noise_stft
                source_power = float(
                    np.sum(np.abs(source_component) ** 2)
                )
                unscaled_noise_power = float(
                    np.sum(np.abs(noise_component) ** 2)
                )
                target_linear = float(
                    np.exp(
                        np.clip(
                            target_snr_db * np.log(10.0) / 10.0,
                            -700.0,
                            700.0,
                        )
                    )
                )
                noise_scale = np.sqrt(
                    max(source_power, 0.0)
                    / max(target_linear * unscaled_noise_power, eps)
                )
            noise_stft *= noise_scale

            if white_noise_level_mode == "inside_noise_floor":
                source_power = float(
                    np.sum(np.abs(source_weight * original_stft) ** 2)
                )
                injected_noise_power = float(
                    np.sum(np.abs(noise_weight * noise_stft) ** 2)
                )
                resulting_snr_db = 10.0 * np.log10(
                    max(source_power, eps)
                    / max(injected_noise_power, eps)
                )
                target_note = (
                    "; target_snr_db ignored in this mode"
                    if target_snr_db is not None
                    else ""
                )
                peak_note = ""
                if estimate["method"] == "two_component_gmm":
                    peak_note = (
                        f", peaks={estimate['noise_peak_db']:.1f}/"
                        f"{estimate['signal_peak_db']:.1f} dB"
                    )
                # print(
                #     f"[MASK NOISE] source {source_idx}: "
                #     f"inside floor={estimate['noise_db']:.1f} dB "
                #     f"({estimate['method']}, n={estimate['n_bins']}"
                #     f"{peak_note}), resulting SNR={resulting_snr_db:.1f} dB"
                #     f"{target_note}"
                # )

            output_stft = (
                source_weight * original_stft
                + noise_weight * noise_stft
            )
        _, output = istft(
            output_stft,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            input_onesided=True,
            boundary=True,
        )
        output = np.asarray(output, dtype=np.float32)
        if output.size < signal.size:
            output = np.pad(output, (0, signal.size - output.size))
        outputs.append(output[:signal.size])
        diagnostics.append(source_diagnostic)

    if return_diagnostics:
        return outputs, diagnostics
    return outputs


def omega_list_to_hz_ranges(Omega_list, fs, nfft):
    """
    Convert Omega_list (FFT-bin bands) into frequency ranges in Hz.

    Returns
    -------
    band_ranges_hz : list of tuples
        [(f_lo, f_hi), ...] for each band.
    """
    freqs = np.arange(nfft // 2 + 1) * (fs / float(nfft))
    band_ranges_hz = []

    for Omega_idx in Omega_list:
        Omega_idx = np.asarray(Omega_idx, dtype=int)
        if Omega_idx.size == 0:
            band_ranges_hz.append((0.0, 0.0))
            continue
        f_lo = float(freqs[Omega_idx.min()])
        f_hi = float(freqs[Omega_idx.max()])
        band_ranges_hz.append((f_lo, f_hi))

    return band_ranges_hz
