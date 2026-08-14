import numpy as np
from itertools import product

"""
Convention
----------------------
All TDOA / delay variables (tau) in this module are expressed in samples.

This module covers:
  - Feature construction from sub-band support (band-count feature vectors)
  - Association of TDOA peaks 
"""


# ============================================================
# Feature construction
# ============================================================

def attach_band_info_to_peaks_fast(tau_obs, bands, peak_taus, tol_tau):
    """
    Attach band-support information to each detected peak tau.

    For each peak tau_hat, this collects nearby observations (within tol_tau)
    and builds a band-support histogram (feature vector):
        band_counts[b] = number of observations assigned to band b.

    Parameters
    ----------
    tau_obs : (N,) array-like
        Local TDOA observations in samples.
    bands : (N,) array-like of int
        Band index for each observation in tau_obs.
    peak_taus : (P,) array-like
        Peak locations in samples (from matching pursuit).
    tol_tau : float
        Assignment tolerance in samples (W in the paper):
            an observation tau is assigned to tau_hat if |tau - tau_hat| <= tol_tau.

    Returns
    -------
    peaks : list of dict
        One dict per peak, with:
          - "tau" : float (samples)
          - "bands" : list[int] unique band indices supporting the peak
          - "band_counts" : dict {band_id: count}
    """
    tau_obs = np.asarray(tau_obs, dtype=float)
    bands = np.asarray(bands, dtype=int)
    peak_taus = np.asarray(peak_taus, dtype=float)

    if tau_obs.size != bands.size:
        raise ValueError("tau_obs and bands must have the same length.")

    peaks = []
    if peak_taus.size == 0:
        return peaks

    # If there are no observations, return empty support for each peak
    if tau_obs.size == 0:
        for tau_hat in peak_taus:
            peaks.append({"tau": float(tau_hat), "bands": [], "band_counts": {}})
        return peaks

    for tau_hat in peak_taus:
        close = np.abs(tau_obs - tau_hat) <= tol_tau
        b = bands[close]

        if b.size == 0:
            peaks.append({"tau": float(tau_hat), "bands": [], "band_counts": {}})
            continue

        uniq, cnt = np.unique(b, return_counts=True)
        peaks.append(
            {
                "tau": float(tau_hat),  # samples
                "bands": uniq.tolist(),
                "band_counts": dict(zip(uniq.tolist(), cnt.tolist())),
            }
        )

    return peaks


def peak_band_vector(peak, n_bands):
    """
    Convert peak["band_counts"] into a dense band-support vector.

    Parameters
    ----------
    peak : dict
        Must contain 'band_counts' (dict {band_id: count}).
    n_bands : int
        Total number of bands (feature dimension).

    Returns
    -------
    vec : (n_bands,) ndarray
        Dense band-support vector (float).
    """
    vec = np.zeros(n_bands, dtype=float)
    for b, cnt in peak.get("band_counts", {}).items():
        b = int(b)
        if 0 <= b < n_bands:
            vec[b] = float(cnt)
    return vec


def peak_timefreq_vector(
    peak,
    n_bands,
    n_frames=None,
    frames_per_bin=1,
    frame_start=0,
):
    """
    Convert a peak's time-frequency support into a dense flattened vector.

    The output is the row-major flattening of a matrix with shape
    ``(n_bands, n_time_bins)``. Consecutive analysis frames can be pooled into
    one time bin using ``frames_per_bin``.

    Parameters
    ----------
    peak : dict
        Peak containing ``tf_cells`` as ``(band, frame)`` rows. If ``tf_cells``
        is absent, ``tf_support_map`` is used instead.
    n_bands : int
        Number of frequency bands.
    n_frames : int or None
        Number of analysis frames starting at ``frame_start``. Use the same
        value for every peak that will be compared. If None, infer it from the
        available peak support.
    frames_per_bin : int
        Number of consecutive analysis frames pooled into each time bin.
    frame_start : int
        First global analysis frame represented in the vector.

    Returns
    -------
    vec : (n_bands * n_time_bins,) ndarray
        Flattened time-frequency count vector.
    """
    n_bands = int(n_bands)
    frames_per_bin = int(frames_per_bin)
    frame_start = int(frame_start)

    if n_bands <= 0:
        raise ValueError("n_bands must be positive.")
    if frames_per_bin <= 0:
        raise ValueError("frames_per_bin must be positive.")

    tf_cells = peak.get("tf_cells")
    support_map = peak.get("tf_support_map")

    if n_frames is None:
        if tf_cells is not None:
            cells = np.asarray(tf_cells)
            if cells.size == 0:
                n_frames = 0
            else:
                cells = cells.reshape(-1, 2)
                n_frames = max(0, int(np.max(cells[:, 1])) - frame_start + 1)
        elif support_map is not None:
            support_map = np.asarray(support_map)
            if support_map.ndim != 2:
                raise ValueError("tf_support_map must be a 2D array.")
            n_frames = max(0, support_map.shape[1] - frame_start)
        else:
            n_frames = 0
    else:
        n_frames = int(n_frames)
        if n_frames < 0:
            raise ValueError("n_frames must be non-negative.")

    n_time_bins = int(np.ceil(n_frames / frames_per_bin))
    timefreq = np.zeros((n_bands, n_time_bins), dtype=float)
    if n_time_bins == 0:
        return timefreq.ravel()

    if tf_cells is not None:
        cells = np.asarray(tf_cells)
        if cells.size == 0:
            return timefreq.ravel()
        cells = cells.reshape(-1, 2)
        bands = cells[:, 0].astype(int)
        relative_frames = cells[:, 1].astype(int) - frame_start
        time_bins = relative_frames // frames_per_bin
        valid = (
            (bands >= 0)
            & (bands < n_bands)
            & (relative_frames >= 0)
            & (relative_frames < n_frames)
            & (time_bins >= 0)
            & (time_bins < n_time_bins)
        )
        np.add.at(timefreq, (bands[valid], time_bins[valid]), 1.0)
        return timefreq.ravel()

    if support_map is not None:
        support_map = np.asarray(support_map, dtype=float)
        if support_map.ndim != 2:
            raise ValueError("tf_support_map must be a 2D array.")

        band_stop = min(n_bands, support_map.shape[0])
        frame_stop = min(frame_start + n_frames, support_map.shape[1])
        if frame_start < 0:
            raise ValueError("frame_start must be non-negative when using tf_support_map.")

        for frame in range(frame_start, frame_stop):
            time_bin = (frame - frame_start) // frames_per_bin
            timefreq[:band_stop, time_bin] += support_map[:band_stop, frame]

    return timefreq.ravel()


def peak_time_vector(
    peak,
    n_frames,
    frames_per_bin=1,
    frame_start=0,
):
    """
    Convert a peak's support into a time-only activity vector.

    Support from all frequency bands is summed within each pooled time bin.
    """
    n_frames = int(n_frames)
    frames_per_bin = int(frames_per_bin)
    frame_start = int(frame_start)
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative.")
    if frames_per_bin <= 0:
        raise ValueError("frames_per_bin must be positive.")

    n_time_bins = int(np.ceil(n_frames / frames_per_bin))
    time_vector = np.zeros(n_time_bins, dtype=float)
    if n_time_bins == 0:
        return time_vector

    tf_cells = peak.get("tf_cells")
    if tf_cells is not None:
        cells = np.asarray(tf_cells)
        if cells.size == 0:
            return time_vector
        cells = cells.reshape(-1, 2)
        relative_frames = cells[:, 1].astype(int) - frame_start
        time_bins = relative_frames // frames_per_bin
        valid = (
            (relative_frames >= 0)
            & (relative_frames < n_frames)
            & (time_bins >= 0)
            & (time_bins < n_time_bins)
        )
        np.add.at(time_vector, time_bins[valid], 1.0)
        return time_vector

    support_map = peak.get("tf_support_map")
    if support_map is not None:
        support_map = np.asarray(support_map, dtype=float)
        if support_map.ndim != 2:
            raise ValueError("tf_support_map must be a 2D array.")
        if frame_start < 0:
            raise ValueError("frame_start must be non-negative.")

        frame_stop = min(frame_start + n_frames, support_map.shape[1])
        for frame in range(frame_start, frame_stop):
            time_bin = (frame - frame_start) // frames_per_bin
            time_vector[time_bin] += np.sum(support_map[:, frame])

    return time_vector


def cosine_similarity(vec_a, vec_b):
    """Return cosine similarity, or zero if either vector has zero norm."""
    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)
    if vec_a.shape != vec_b.shape:
        raise ValueError("Vectors must have the same shape.")
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denominator)


def time_cosine_similarity_shifted(vec_a, vec_b, max_shift_bins=0):
    """
    Return the best cosine similarity over zero-filled time shifts.

    The denominator uses the original vector norms so support shifted outside
    the analysis interval is penalized rather than artificially renormalized.
    """
    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)
    max_shift_bins = int(max_shift_bins)
    if vec_a.shape != vec_b.shape:
        raise ValueError("Time vectors must have the same shape.")
    if vec_a.ndim != 1:
        raise ValueError("Time vectors must be one-dimensional.")
    if max_shift_bins < 0:
        raise ValueError("max_shift_bins must be non-negative.")
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denominator == 0.0:
        return 0.0, 0

    best_score = -np.inf
    best_shift = 0
    n_bins = vec_a.size
    for shift in range(-max_shift_bins, max_shift_bins + 1):
        shifted_b = np.zeros_like(vec_b)
        if shift == 0:
            shifted_b = vec_b
        elif 0 < shift < n_bins:
            shifted_b[shift:] = vec_b[:-shift]
        elif -n_bins < shift < 0:
            shifted_b[:shift] = vec_b[-shift:]

        score = float(np.dot(vec_a, shifted_b) / denominator)
        if score > best_score:
            best_score = score
            best_shift = shift

    return float(best_score), int(best_shift)


def timefreq_cosine_similarity_shifted(
    vec_a,
    vec_b,
    n_bands,
    max_shift_bins=0,
):
    """
    Return the best cosine similarity over zero-filled time shifts.

    Positive shifts move ``vec_b`` later in time; negative shifts move it
    earlier. Both vectors must represent the same time-frequency grid.
    """
    n_bands = int(n_bands)
    max_shift_bins = int(max_shift_bins)
    if n_bands <= 0:
        raise ValueError("n_bands must be positive.")
    if max_shift_bins < 0:
        raise ValueError("max_shift_bins must be non-negative.")

    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)
    if vec_a.shape != vec_b.shape:
        raise ValueError("Time-frequency vectors must have the same shape.")
    if vec_a.size % n_bands != 0:
        raise ValueError("Vector length must be divisible by n_bands.")

    map_a = vec_a.reshape(n_bands, -1)
    map_b = vec_b.reshape(n_bands, -1)
    norm_a = np.linalg.norm(map_a)
    if norm_a == 0.0 or np.linalg.norm(map_b) == 0.0:
        return 0.0, 0

    best_score = -np.inf
    best_shift = 0
    n_time_bins = map_a.shape[1]

    for shift in range(-max_shift_bins, max_shift_bins + 1):
        shifted_b = np.zeros_like(map_b)
        if shift == 0:
            shifted_b = map_b
        elif 0 < shift < n_time_bins:
            shifted_b[:, shift:] = map_b[:, :-shift]
        elif -n_time_bins < shift < 0:
            shifted_b[:, :shift] = map_b[:, -shift:]

        denominator = norm_a * np.linalg.norm(shifted_b)
        score = 0.0 if denominator == 0.0 else float(
            np.sum(map_a * shifted_b) / denominator
        )
        if score > best_score:
            best_score = score
            best_shift = shift

    return float(best_score), int(best_shift)


# ============================================================
# Association
# ============================================================

def associate_tdoa_peaks_by_band_cosine_constrained(
    tdoas_hat_pairs,
    n_bands,
    cosine_thresh=0.5,
    max_sources=None,
    min_tdoas_per_source=2,
):
    """
    Associate (mic_pair, peak) candidates across microphone pairs using cosine similarity
    between band-support feature vectors, under a physical constraint:
        a source can induce at most one TDOA peak per microphone pair.

    Overview
    --------
    1) Create one node per peak (per microphone pair).
    2) Compute cosine similarity between feature vectors for all cross-pair node pairs.
    3) Keep only similarities >= cosine_thresh, and only across different mic pairs.
    4) Greedily merge clusters (from strongest to weakest link) while enforcing:
         - no cluster may contain two peaks from the same mic pair.
    5) Return clusters as source hypotheses.

    Parameters
    ----------
    tdoas_hat_pairs : dict
        (i,j) -> dict with key "peaks" (list of dicts).
        Each peak dict is expected to contain:
          - "tau" (samples)
          - "band_counts" (dict) or be compatible with peak_band_vector
    n_bands : int
        Feature dimension (total number of bands).
    cosine_thresh : float
        Minimum cosine similarity to consider two peaks compatible (beta in the paper).
    max_sources : int or None
        If set, keep only the largest `max_sources` clusters (S in the paper).
    min_tdoas_per_source : int
        Discard clusters with fewer than this many members (Discards groups form by only one TDOA).

    Returns
    -------
    sources : list of dict
        Each dict corresponds to one source hypothesis and contains:
          - "members": list of dict with keys:
              "pair", "peak_idx", "tau", "bands", "band_counts", "vec"
    """
    # -------------------------
    # 1) Build nodes: one per peak
    # -------------------------
    nodes = []
    for (i, j), info in tdoas_hat_pairs.items():
        for p_idx, peak in enumerate(info.get("peaks", [])):
            vec = peak_band_vector(peak, n_bands)
            nodes.append(((i, j), p_idx, peak, vec))

    N = len(nodes)
    if N == 0:
        return []

    vecs = np.stack([n[3] for n in nodes], axis=0).astype(float)
    norms = np.linalg.norm(vecs, axis=1) + 1e-12
    pair_of = [nodes[u][0] for u in range(N)]  # mic pair label for node u

    def cosine(u, v):
        return float(np.dot(vecs[u], vecs[v]) / (norms[u] * norms[v]))

    # -------------------------
    # 2) Prospective pairs (above threshold, cross-pair only)
    # -------------------------
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if pair_of[u] == pair_of[v]:
                continue  # cannot associate two peaks from the same mic pair
            s = cosine(u, v)
            if s >= cosine_thresh:
                edges.append((s, u, v))

    # If no edges survive, each node becomes its own cluster
    if len(edges) == 0:
        comps = [[u] for u in range(N)]
    else:
        # -------------------------
        # 3) Constrained greedy merging
        # -------------------------
        edges.sort(reverse=True, key=lambda x: x[0])

        parent = np.arange(N, dtype=int)
        size = np.ones(N, dtype=int)
        used_pairs = [set([pair_of[u]]) for u in range(N)]

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False

            # Physical constraint: one peak per mic pair inside each cluster
            if used_pairs[ra] & used_pairs[rb]:
                return False

            # Union by size
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            used_pairs[ra] |= used_pairs[rb]
            return True

        for _, u, v in edges:
            union(u, v)

        buckets = {}
        for u in range(N):
            r = find(u)
            buckets.setdefault(r, []).append(u)
        comps = list(buckets.values())

    # -------------------------
    # 4) Build output sources
    # -------------------------
    def make_source(indices):
        members = []
        for idx in indices:
            pair, p_idx, peak, vec = nodes[idx]
            members.append(
                {
                    "pair": pair,
                    "peak_idx": p_idx,
                    "tau": float(peak["tau"]),  # samples
                    "bands": peak.get("bands", []),
                    "band_counts": peak.get("band_counts", {}),
                    "vec": vec,
                }
            )
        return {"members": members}

    sources = [make_source(c) for c in comps]

    # Keep only largest sources if requested
    if max_sources is not None and len(sources) > max_sources:
        sources = sorted(sources, key=lambda s: len(s["members"]), reverse=True)[:max_sources]

    # Enforce minimum number of supporting pairs
    sources = [s for s in sources if len(s["members"]) >= min_tdoas_per_source]

    return sources


def associate_tdoa_peaks_by_timefreq_cosine_constrained(
    tdoas_hat_pairs,
    n_bands,
    n_frames,
    cosine_thresh=0.5,
    max_shift_bins=0,
    frames_per_bin=1,
    frame_start=0,
    alpha=0.5,
    max_sources=None,
    min_tdoas_per_source=2,
):
    """
    Associate peaks using separate frequency and time similarities.

    The edge score is ``alpha * frequency_cosine + (1 - alpha) *
    shifted_time_cosine``. Frequency-only association remains implemented by
    ``associate_tdoa_peaks_by_band_cosine_constrained`` and is not changed.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    nodes = []
    for pair, info in tdoas_hat_pairs.items():
        for peak_idx, peak in enumerate(info.get("peaks", [])):
            frequency_vec = peak_band_vector(peak, n_bands)
            time_vec = peak_time_vector(
                peak,
                n_frames=n_frames,
                frames_per_bin=frames_per_bin,
                frame_start=frame_start,
            )
            nodes.append(
                (tuple(pair), peak_idx, peak, frequency_vec, time_vec)
            )

    n_nodes = len(nodes)
    if n_nodes == 0:
        return []

    pair_of = [node[0] for node in nodes]
    edges = []
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            if pair_of[u] == pair_of[v]:
                continue
            frequency_score = cosine_similarity(nodes[u][3], nodes[v][3])
            time_score, best_shift = time_cosine_similarity_shifted(
                nodes[u][4],
                nodes[v][4],
                max_shift_bins=max_shift_bins,
            )
            score = alpha * frequency_score + (1.0 - alpha) * time_score
            if score >= cosine_thresh:
                edges.append(
                    (score, u, v, best_shift, frequency_score, time_score)
                )

    parent = np.arange(n_nodes, dtype=int)
    size = np.ones(n_nodes, dtype=int)
    used_pairs = [{pair_of[u]} for u in range(n_nodes)]

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if used_pairs[root_left] & used_pairs[root_right]:
            return
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]
        used_pairs[root_left] |= used_pairs[root_right]

    edges.sort(reverse=True, key=lambda edge: edge[0])
    for _, u, v, _, _, _ in edges:
        union(u, v)

    components = {}
    for node in range(n_nodes):
        components.setdefault(find(node), []).append(node)

    sources = []
    for indices in components.values():
        members = []
        for idx in indices:
            pair, peak_idx, peak, frequency_vec, time_vec = nodes[idx]
            members.append(
                {
                    "pair": pair,
                    "peak_idx": peak_idx,
                    "tau": float(peak["tau"]),
                    "bands": peak.get("bands", []),
                    "band_counts": peak.get("band_counts", {}),
                    "vec": frequency_vec,
                    "frequency_vec": frequency_vec,
                    "time_vec": time_vec,
                }
            )
        sources.append({"members": members})

    if max_sources is not None and len(sources) > max_sources:
        sources = sorted(
            sources,
            key=lambda source: len(source["members"]),
            reverse=True,
        )[:max_sources]

    return [
        source
        for source in sources
        if len(source["members"]) >= min_tdoas_per_source
    ]


def associate_tdoa_peaks_by_timefreq_candidates(
    tdoas_hat_pairs,
    n_bands,
    n_frames,
    cosine_thresh=0.5,
    inpair_cosine_thresh=0.9,
    max_shift_bins=0,
    frames_per_bin=1,
    frame_start=0,
    alpha=0.5,
    max_sources=None,
    min_tdoas_per_source=2,
    max_exact_combinations=10000,
):
    """
    Associate logical same-pair candidate groups without using geometry.

    Highly similar peaks from one microphone-pair orientation are retained as
    alternatives. Logical groups are associated across pair orientations, then
    one peak per group is selected by maximum frequency/time agreement with the
    other selected peaks. This is useful when reciprocal pairs such as (0, 1)
    and (1, 0) are used solely to associate two-microphone observations.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    inpair_cosine_thresh = float(inpair_cosine_thresh)
    if not 0.0 <= inpair_cosine_thresh <= 1.0:
        raise ValueError("inpair_cosine_thresh must be between 0 and 1.")
    max_exact_combinations = int(max_exact_combinations)
    if max_exact_combinations < 1:
        raise ValueError("max_exact_combinations must be positive.")

    peak_nodes = []
    for pair, info in tdoas_hat_pairs.items():
        for peak_idx, peak in enumerate(info.get("peaks", [])):
            frequency_vec = peak_band_vector(peak, n_bands)
            time_vec = peak_time_vector(
                peak,
                n_frames=n_frames,
                frames_per_bin=frames_per_bin,
                frame_start=frame_start,
            )
            peak_nodes.append(
                (tuple(pair), peak_idx, peak, frequency_vec, time_vec)
            )

    if not peak_nodes:
        return []

    def peak_similarity(left_idx, right_idx):
        frequency_score = cosine_similarity(
            peak_nodes[left_idx][3],
            peak_nodes[right_idx][3],
        )
        time_score, best_shift = time_cosine_similarity_shifted(
            peak_nodes[left_idx][4],
            peak_nodes[right_idx][4],
            max_shift_bins=max_shift_bins,
        )
        score = alpha * frequency_score + (1.0 - alpha) * time_score
        return score, frequency_score, time_score, best_shift

    def make_member(peak_node_idx):
        pair, peak_idx, peak, frequency_vec, time_vec = peak_nodes[peak_node_idx]
        return {
            "pair": pair,
            "peak_idx": peak_idx,
            "tau": float(peak["tau"]),
            "bands": peak.get("bands", []),
            "band_counts": peak.get("band_counts", {}),
            "vec": frequency_vec,
            "frequency_vec": frequency_vec,
            "time_vec": time_vec,
        }

    peak_indices_by_pair = {}
    for peak_node_idx, node in enumerate(peak_nodes):
        peak_indices_by_pair.setdefault(node[0], []).append(peak_node_idx)

    logical_nodes = []
    inpair_edges = []
    for pair, pair_indices in sorted(peak_indices_by_pair.items()):
        local_parent = {idx: idx for idx in pair_indices}

        def local_find(idx):
            while local_parent[idx] != idx:
                local_parent[idx] = local_parent[local_parent[idx]]
                idx = local_parent[idx]
            return idx

        def local_union(left_idx, right_idx):
            root_left = local_find(left_idx)
            root_right = local_find(right_idx)
            if root_left != root_right:
                local_parent[root_right] = root_left

        for pos, left_idx in enumerate(pair_indices):
            for right_idx in pair_indices[pos + 1:]:
                score, frequency_score, time_score, best_shift = (
                    peak_similarity(left_idx, right_idx)
                )
                admitted = score >= inpair_cosine_thresh
                inpair_edges.append(
                    {
                        "pair": pair,
                        "left_peak_idx": int(peak_nodes[left_idx][1]),
                        "right_peak_idx": int(peak_nodes[right_idx][1]),
                        "score": float(score),
                        "frequency_score": float(frequency_score),
                        "time_score": float(time_score),
                        "best_shift": int(best_shift),
                        "grouped": bool(admitted),
                    }
                )
                if admitted:
                    local_union(left_idx, right_idx)

        groups = {}
        for peak_node_idx in pair_indices:
            groups.setdefault(local_find(peak_node_idx), []).append(peak_node_idx)

        for candidate_indices in groups.values():
            logical_nodes.append(
                {
                    "pair": pair,
                    "candidate_indices": candidate_indices,
                    "frequency_vec": np.max(
                        np.stack(
                            [peak_nodes[idx][3] for idx in candidate_indices]
                        ),
                        axis=0,
                    ),
                    "time_vec": np.max(
                        np.stack(
                            [peak_nodes[idx][4] for idx in candidate_indices]
                        ),
                        axis=0,
                    ),
                }
            )

    n_logical_nodes = len(logical_nodes)
    parent = np.arange(n_logical_nodes, dtype=int)
    size = np.ones(n_logical_nodes, dtype=int)
    used_pairs = [{node["pair"]} for node in logical_nodes]

    def find(node_idx):
        while parent[node_idx] != node_idx:
            parent[node_idx] = parent[parent[node_idx]]
            node_idx = parent[node_idx]
        return node_idx

    def union(left_idx, right_idx):
        root_left = find(left_idx)
        root_right = find(right_idx)
        if root_left == root_right:
            return False
        if used_pairs[root_left] & used_pairs[root_right]:
            return False
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]
        used_pairs[root_left] |= used_pairs[root_right]
        return True

    cross_edges = []
    for left_idx in range(n_logical_nodes):
        for right_idx in range(left_idx + 1, n_logical_nodes):
            if logical_nodes[left_idx]["pair"] == logical_nodes[right_idx]["pair"]:
                continue
            frequency_score = cosine_similarity(
                logical_nodes[left_idx]["frequency_vec"],
                logical_nodes[right_idx]["frequency_vec"],
            )
            time_score, best_shift = time_cosine_similarity_shifted(
                logical_nodes[left_idx]["time_vec"],
                logical_nodes[right_idx]["time_vec"],
                max_shift_bins=max_shift_bins,
            )
            score = alpha * frequency_score + (1.0 - alpha) * time_score
            if score >= cosine_thresh:
                cross_edges.append(
                    (
                        score,
                        left_idx,
                        right_idx,
                        best_shift,
                        frequency_score,
                        time_score,
                    )
                )

    cross_edges.sort(reverse=True, key=lambda edge: edge[0])
    for _, left_idx, right_idx, _, _, _ in cross_edges:
        union(left_idx, right_idx)

    components = {}
    for logical_idx in range(n_logical_nodes):
        components.setdefault(find(logical_idx), []).append(logical_idx)

    def evaluate_choice(choice):
        pair_scores = []
        for left_pos, left_idx in enumerate(choice):
            for right_idx in choice[left_pos + 1:]:
                pair_scores.append(peak_similarity(left_idx, right_idx)[0])
        agreement = float(np.mean(pair_scores)) if pair_scores else 0.0
        support = float(np.mean([
            np.sum(peak_nodes[idx][3]) + np.sum(peak_nodes[idx][4])
            for idx in choice
        ]))
        return agreement, support

    def select_coordinate_descent(candidate_groups):
        ranked_groups = [
            sorted(
                group,
                key=lambda idx: (
                    np.sum(peak_nodes[idx][3])
                    + np.sum(peak_nodes[idx][4])
                ),
                reverse=True,
            )
            for group in candidate_groups
        ]
        seeds = [tuple(group[0] for group in ranked_groups)]
        for group_idx, group in enumerate(ranked_groups):
            for candidate in group[1:]:
                seed = list(seeds[0])
                seed[group_idx] = candidate
                seeds.append(tuple(seed))
                if len(seeds) >= max_exact_combinations:
                    break
            if len(seeds) >= max_exact_combinations:
                break

        best_choice = None
        best_score = None
        n_evaluated = 0
        for seed in seeds:
            choice = list(seed)
            improved = True
            while improved:
                improved = False
                current_score = evaluate_choice(choice)
                n_evaluated += 1
                for group_idx, group in enumerate(ranked_groups):
                    local_choice = choice[group_idx]
                    local_score = current_score
                    for candidate in group:
                        trial = list(choice)
                        trial[group_idx] = candidate
                        trial_score = evaluate_choice(trial)
                        n_evaluated += 1
                        if trial_score > local_score:
                            local_choice = candidate
                            local_score = trial_score
                    if local_choice != choice[group_idx]:
                        choice[group_idx] = local_choice
                        current_score = local_score
                        improved = True

            score = evaluate_choice(choice)
            n_evaluated += 1
            if best_score is None or score > best_score:
                best_choice = tuple(choice)
                best_score = score

        return best_choice, best_score, n_evaluated

    sources = []
    for component in components.values():
        if len(component) < int(min_tdoas_per_source):
            continue

        candidate_groups = [
            logical_nodes[logical_idx]["candidate_indices"]
            for logical_idx in component
        ]
        combination_count = int(
            np.prod([len(group) for group in candidate_groups], dtype=object)
        )

        if combination_count <= max_exact_combinations:
            choices = product(*candidate_groups)
            search_method = "exact"
            best_choice = None
            best_score = None
            n_evaluated = 0
            for choice in choices:
                score = evaluate_choice(choice)
                n_evaluated += 1
                if best_score is None or score > best_score:
                    best_choice = tuple(choice)
                    best_score = score
        else:
            best_choice, best_score, n_evaluated = (
                select_coordinate_descent(candidate_groups)
            )
            search_method = "coordinate_descent"

        if best_choice is None:
            continue

        selected_source = {
            "members": [make_member(idx) for idx in best_choice]
        }
        selected_indices = set(best_choice)
        grouped_indices = [
            peak_idx
            for logical_idx in component
            for peak_idx in logical_nodes[logical_idx]["candidate_indices"]
        ]
        selected_source["candidate_selection"] = {
            "search_method": search_method,
            "n_combinations": combination_count,
            "n_evaluated": n_evaluated,
            "agreement_score": float(best_score[0]),
            "support_score": float(best_score[1]),
            "rejected_members": [
                make_member(idx)
                for idx in grouped_indices
                if idx not in selected_indices
            ],
            "inpair_cosine_thresh": inpair_cosine_thresh,
            "candidate_groups": [
                {
                    "pair": logical_nodes[logical_idx]["pair"],
                    "peak_indices": [
                        int(peak_nodes[idx][1])
                        for idx in logical_nodes[logical_idx][
                            "candidate_indices"
                        ]
                    ],
                }
                for logical_idx in component
            ],
            "inpair_edges": [
                edge
                for edge in inpair_edges
                if edge["pair"] in {
                    logical_nodes[logical_idx]["pair"]
                    for logical_idx in component
                }
            ],
        }
        sources.append(selected_source)

    if max_sources is not None and len(sources) > max_sources:
        sources = sorted(
            sources,
            key=lambda source: len(source["members"]),
            reverse=True,
        )[:max_sources]

    return sources


def associate_tdoa_peaks_by_timefreq_candidates_geometric(
    tdoas_hat_pairs,
    n_bands,
    n_frames,
    mic_positions,
    fs,
    c,
    env_bounds,
    cosine_thresh=0.5,
    inpair_cosine_thresh=0.9,
    max_shift_bins=0,
    frames_per_bin=1,
    frame_start=0,
    alpha=0.5,
    max_sources=None,
    min_tdoas_per_source=2,
    max_exact_combinations=10000,
    grid_spacing=None,
    n_starts=5,
):
    """
    Group same-pair alternatives, associate groups, then select TDOAs geometrically.

    Peaks from the same microphone pair are first grouped when their weighted
    frequency/time similarity exceeds ``inpair_cosine_thresh``. Each logical
    group keeps all original TDOAs but represents their acoustic support using
    a bin-wise maximum. The original strongest-edge-first, one-node-per-pair
    greedy constraint is then applied to these logical nodes. Finally, one
    original TDOA per selected group is chosen by minimum global geometric RMS.
    """
    from .posEstimator import fit_source_and_geometric_residuals

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    inpair_cosine_thresh = float(inpair_cosine_thresh)
    if not 0.0 <= inpair_cosine_thresh <= 1.0:
        raise ValueError("inpair_cosine_thresh must be between 0 and 1.")
    max_exact_combinations = int(max_exact_combinations)
    if max_exact_combinations < 1:
        raise ValueError("max_exact_combinations must be positive.")

    peak_nodes = []
    for pair, info in tdoas_hat_pairs.items():
        for peak_idx, peak in enumerate(info.get("peaks", [])):
            frequency_vec = peak_band_vector(peak, n_bands)
            time_vec = peak_time_vector(
                peak,
                n_frames=n_frames,
                frames_per_bin=frames_per_bin,
                frame_start=frame_start,
            )
            peak_nodes.append(
                (tuple(pair), peak_idx, peak, frequency_vec, time_vec)
            )

    n_peak_nodes = len(peak_nodes)
    if n_peak_nodes == 0:
        return []

    def peak_similarity(left_idx, right_idx):
        frequency_score = cosine_similarity(
            peak_nodes[left_idx][3],
            peak_nodes[right_idx][3],
        )
        time_score, best_shift = time_cosine_similarity_shifted(
            peak_nodes[left_idx][4],
            peak_nodes[right_idx][4],
            max_shift_bins=max_shift_bins,
        )
        score = alpha * frequency_score + (1.0 - alpha) * time_score
        return score, frequency_score, time_score, best_shift

    def make_member(peak_node_idx):
        pair, peak_idx, peak, frequency_vec, time_vec = peak_nodes[peak_node_idx]
        return {
            "pair": pair,
            "peak_idx": peak_idx,
            "tau": float(peak["tau"]),
            "bands": peak.get("bands", []),
            "band_counts": peak.get("band_counts", {}),
            "vec": frequency_vec,
            "frequency_vec": frequency_vec,
            "time_vec": time_vec,
        }

    # Group highly similar peaks inside each microphone pair.
    peak_indices_by_pair = {}
    for peak_node_idx, node in enumerate(peak_nodes):
        peak_indices_by_pair.setdefault(node[0], []).append(peak_node_idx)

    logical_nodes = []
    inpair_edges = []
    for pair, pair_indices in sorted(peak_indices_by_pair.items()):
        local_parent = {idx: idx for idx in pair_indices}

        def local_find(idx):
            while local_parent[idx] != idx:
                local_parent[idx] = local_parent[local_parent[idx]]
                idx = local_parent[idx]
            return idx

        def local_union(left_idx, right_idx):
            root_left = local_find(left_idx)
            root_right = local_find(right_idx)
            if root_left != root_right:
                local_parent[root_right] = root_left

        for pos, left_idx in enumerate(pair_indices):
            for right_idx in pair_indices[pos + 1:]:
                score, frequency_score, time_score, best_shift = (
                    peak_similarity(left_idx, right_idx)
                )
                admitted = score >= inpair_cosine_thresh
                inpair_edges.append(
                    {
                        "pair": pair,
                        "left_peak_idx": int(peak_nodes[left_idx][1]),
                        "right_peak_idx": int(peak_nodes[right_idx][1]),
                        "score": float(score),
                        "frequency_score": float(frequency_score),
                        "time_score": float(time_score),
                        "best_shift": int(best_shift),
                        "grouped": bool(admitted),
                    }
                )
                if admitted:
                    local_union(left_idx, right_idx)

        groups = {}
        for peak_node_idx in pair_indices:
            groups.setdefault(local_find(peak_node_idx), []).append(
                peak_node_idx
            )

        for candidate_indices in groups.values():
            frequency_vec = np.max(
                np.stack(
                    [peak_nodes[idx][3] for idx in candidate_indices],
                    axis=0,
                ),
                axis=0,
            )
            time_vec = np.max(
                np.stack(
                    [peak_nodes[idx][4] for idx in candidate_indices],
                    axis=0,
                ),
                axis=0,
            )
            logical_nodes.append(
                {
                    "pair": pair,
                    "candidate_indices": candidate_indices,
                    "frequency_vec": frequency_vec,
                    "time_vec": time_vec,
                }
            )

    n_logical_nodes = len(logical_nodes)
    parent = np.arange(n_logical_nodes, dtype=int)
    size = np.ones(n_logical_nodes, dtype=int)
    used_pairs = [{node["pair"]} for node in logical_nodes]

    def find(node_idx):
        while parent[node_idx] != node_idx:
            parent[node_idx] = parent[parent[node_idx]]
            node_idx = parent[node_idx]
        return node_idx

    def union(left_idx, right_idx):
        root_left = find(left_idx)
        root_right = find(right_idx)
        if root_left == root_right:
            return False
        if used_pairs[root_left] & used_pairs[root_right]:
            return False
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]
        used_pairs[root_left] |= used_pairs[root_right]
        return True

    # Original constrained greedy clustering, now over logical group nodes.
    cross_edges = []
    for left_idx in range(n_logical_nodes):
        for right_idx in range(left_idx + 1, n_logical_nodes):
            if logical_nodes[left_idx]["pair"] == logical_nodes[right_idx]["pair"]:
                continue
            frequency_score = cosine_similarity(
                logical_nodes[left_idx]["frequency_vec"],
                logical_nodes[right_idx]["frequency_vec"],
            )
            time_score, best_shift = time_cosine_similarity_shifted(
                logical_nodes[left_idx]["time_vec"],
                logical_nodes[right_idx]["time_vec"],
                max_shift_bins=max_shift_bins,
            )
            score = alpha * frequency_score + (1.0 - alpha) * time_score
            if score >= cosine_thresh:
                cross_edges.append(
                    (
                        score,
                        left_idx,
                        right_idx,
                        best_shift,
                        frequency_score,
                        time_score,
                    )
                )

    cross_edges.sort(reverse=True, key=lambda edge: edge[0])
    for _, left_idx, right_idx, _, _, _ in cross_edges:
        union(left_idx, right_idx)

    components = {}
    for logical_idx in range(n_logical_nodes):
        components.setdefault(find(logical_idx), []).append(logical_idx)

    def evaluate_choice(choice, cache):
        key = tuple(choice)
        if key in cache:
            return cache[key]

        source = {"members": [make_member(idx) for idx in choice]}
        estimate, residuals = fit_source_and_geometric_residuals(
            source,
            mic_positions=mic_positions,
            fs=fs,
            c=c,
            env_bounds=env_bounds,
            grid_spacing=grid_spacing,
            n_starts=n_starts,
        )
        residuals = np.asarray(residuals, dtype=float)
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        maximum = float(np.max(residuals))
        result = ((rms, maximum), source, estimate, residuals)
        cache[key] = result
        return result

    def select_coordinate_descent(candidate_groups, cache):
        ranked_groups = [
            sorted(
                group,
                key=lambda peak_idx: np.sum(
                    peak_nodes[peak_idx][3]
                ) + np.sum(peak_nodes[peak_idx][4]),
                reverse=True,
            )
            for group in candidate_groups
        ]
        seeds = [tuple(group[0] for group in ranked_groups)]
        for group_idx, group in enumerate(ranked_groups):
            for candidate in group[1:]:
                seed = list(seeds[0])
                seed[group_idx] = candidate
                seeds.append(tuple(seed))
                if len(seeds) >= max_exact_combinations:
                    break
            if len(seeds) >= max_exact_combinations:
                break

        best = None
        for seed in seeds:
            choice = list(seed)
            improved = True
            while improved:
                improved = False
                current = evaluate_choice(choice, cache)
                for group_idx, group in enumerate(ranked_groups):
                    local_best = current
                    local_choice = choice[group_idx]
                    for candidate in group:
                        trial = list(choice)
                        trial[group_idx] = candidate
                        evaluated = evaluate_choice(trial, cache)
                        if evaluated[0] < local_best[0]:
                            local_best = evaluated
                            local_choice = candidate
                    if local_choice != choice[group_idx]:
                        choice[group_idx] = local_choice
                        current = local_best
                        improved = True
            evaluated = evaluate_choice(choice, cache)
            if best is None or evaluated[0] < best[0]:
                best = evaluated
        return best

    sources = []
    for component in components.values():
        if len(component) < int(min_tdoas_per_source):
            continue

        candidate_groups = [
            logical_nodes[logical_idx]["candidate_indices"]
            for logical_idx in component
        ]
        combination_count = int(
            np.prod([len(group) for group in candidate_groups], dtype=object)
        )
        cache = {}

        if combination_count <= max_exact_combinations:
            best = None
            for choice in product(*candidate_groups):
                evaluated = evaluate_choice(choice, cache)
                if best is None or evaluated[0] < best[0]:
                    best = evaluated
            search_method = "exact"
        else:
            best = select_coordinate_descent(candidate_groups, cache)
            search_method = "coordinate_descent"

        if best is None:
            continue

        _, selected_source, estimate, residuals = best
        selected_peak_indices = {
            (
                tuple(member["pair"]),
                int(member["peak_idx"]),
            )
            for member in selected_source["members"]
        }
        grouped_peak_indices = [
            peak_idx
            for logical_idx in component
            for peak_idx in logical_nodes[logical_idx]["candidate_indices"]
        ]
        rejected = [
            make_member(peak_idx)
            for peak_idx in grouped_peak_indices
            if (
                peak_nodes[peak_idx][0],
                int(peak_nodes[peak_idx][1]),
            ) not in selected_peak_indices
        ]
        selected_source["candidate_selection"] = {
            "search_method": search_method,
            "n_combinations": combination_count,
            "n_evaluated": len(cache),
            "estimate": estimate,
            "rms_residual_samples": float(
                np.sqrt(np.mean(np.asarray(residuals) ** 2))
            ),
            "max_residual_samples": float(np.max(residuals)),
            "rejected_members": rejected,
            "inpair_cosine_thresh": inpair_cosine_thresh,
            "candidate_groups": [
                {
                    "pair": logical_nodes[logical_idx]["pair"],
                    "peak_indices": [
                        int(peak_nodes[peak_idx][1])
                        for peak_idx in logical_nodes[logical_idx][
                            "candidate_indices"
                        ]
                    ],
                }
                for logical_idx in component
            ],
            "inpair_edges": [
                edge
                for edge in inpair_edges
                if edge["pair"] in {
                    logical_nodes[logical_idx]["pair"]
                    for logical_idx in component
                }
            ],
        }
        sources.append(selected_source)

    if max_sources is not None and len(sources) > max_sources:
        sources = sorted(
            sources,
            key=lambda source: len(source["members"]),
            reverse=True,
        )[:max_sources]

    return sources
