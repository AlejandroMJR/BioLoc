import numpy as np

# === Helper functions for shifting source masks between microphones ===

def solve_relative_delays_from_source(source, n_mics, reference="mean"):
    """
    Solve relative microphone delays from the pairwise TDOAs in one source.

    Convention:
        tau_ij = d_i - d_j

    where d_m is the arrival delay at microphone m, in samples. The absolute
    offset is arbitrary, so we constrain either the mean delay of used mics to
    zero or one reference mic to zero.
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


def shift_tf_map_time_zero_fill(M, shift_frames):
    """
    Shift a 2D TF map along the time axis with zero fill.

    Positive shift_frames moves the map later. Negative shift_frames moves it
    earlier. This intentionally does not wrap around.
    """
    M = np.asarray(M, dtype=np.float32)
    out = np.zeros_like(M, dtype=np.float32)
    shift_frames = int(shift_frames)

    if M.ndim != 2 or M.shape[1] == 0:
        return out
    if shift_frames == 0:
        return M.copy()
    if abs(shift_frames) >= M.shape[1]:
        return out

    if shift_frames > 0:
        out[:, shift_frames:] = M[:, :-shift_frames]
    else:
        k = -shift_frames
        out[:, :-k] = M[:, k:]

    return out

def build_source_tf_maps_aligned_to_mic(
    sources,
    tdoas_hat_pairs,
    n_bands,
    n_frames,
    n_mics,
    hop,
    mic_positions=None,
    source_positions=None,
    fallback_ref="first_used",
    fs=None,
    c=343.0,
    reject_ambiguous_pair_tf=False,
    pair_ambiguity_tolerance_samples=None,
):
    """
    Build source TF maps aligned to a valid reference microphone per source.

    Raw TF supports are pair/member based. In extract_subband_tdoas_windowed_search,
    a support from pair (i, j) is anchored to mic i, because the analysis window
    is taken from mic i and mic j is searched around it. Therefore, when a source
    contains supports from several pairs, supports should be shifted to a common
    source-specific reference before merging.

    Unlike a global target mic, the source reference is chosen from the mics that
    actually appear in that source's TDOA graph. This avoids pretending a source
    can be aligned to an unrelated global ref mic when there is no TDOA path to it.

    If mic_positions and source_positions are provided, the reference is chosen
    as the closest used mic to the estimated source position. Otherwise, the
    first used mic is used.

    When ``reject_ambiguous_pair_tf`` is enabled, a member's TF contribution is
    skipped if another localized source predicts a TDOA on that microphone pair
    within the configured ambiguity tolerance. The TDOA member remains in the
    source and therefore still contributes to localization and delay alignment.
    If ``pair_ambiguity_tolerance_samples`` is None, each pair uses the
    ``support_tolerance_samples`` stored in ``tdoas_hat_pairs``.

    Delay convention:
        tau_ij = d_i - d_j

    A contribution anchored at mic i is shifted to source_ref_mic r by:
        shift_samples = d_r - d_i

    Returns
    -------
    aligned_maps : list of ndarray
        Source maps aligned to their own source_ref_mic.
    source_ref_mics : list of int or None
        Reference mic used for each source. None if no valid source members exist.
    debug : list of dict
        Per-source/per-member alignment diagnostics.
    """
    aligned_maps = []
    source_ref_mics = []
    debug = []

    if mic_positions is not None:
        mic_positions = np.asarray(mic_positions, dtype=float)
    if source_positions is not None:
        source_positions = list(source_positions)
    reject_ambiguous_pair_tf = bool(reject_ambiguous_pair_tf)
    if pair_ambiguity_tolerance_samples is not None:
        pair_ambiguity_tolerance_samples = float(
            pair_ambiguity_tolerance_samples
        )
        if pair_ambiguity_tolerance_samples < 0:
            raise ValueError(
                "pair_ambiguity_tolerance_samples must be non-negative"
            )
    if reject_ambiguous_pair_tf:
        if mic_positions is None or source_positions is None or fs is None:
            raise ValueError(
                "Ambiguous-pair TF rejection requires mic_positions, "
                "source_positions, and fs"
            )
        if float(c) <= 0:
            raise ValueError("c must be positive")

    def predicted_tdoa_samples(source_position, pair):
        position = np.asarray(source_position, dtype=float)
        if position.shape != (mic_positions.shape[1],):
            return None
        if not np.all(np.isfinite(position)):
            return None
        i, j = map(int, pair)
        di = np.linalg.norm(position - mic_positions[i])
        dj = np.linalg.norm(position - mic_positions[j])
        return float((di - dj) * float(fs) / float(c))

    for sid, src in enumerate(sources):
        delays, used_mics = solve_relative_delays_from_source(
            src,
            n_mics=n_mics,
            reference="mean",
        )

        src_map = np.zeros((int(n_bands), int(n_frames)), dtype=np.float32)
        src_debug = []

        if len(used_mics) == 0 or delays is None:
            source_ref_mic = None
            aligned_maps.append(src_map)
            source_ref_mics.append(source_ref_mic)
            debug.append({
                "source_idx": int(sid),
                "source_ref_mic": source_ref_mic,
                "used_mics": used_mics,
                "members": src_debug,
                "reason": "missing_delays_or_used_mics",
            })
            continue

        if (
            mic_positions is not None
            and source_positions is not None
            and sid < len(source_positions)
            and source_positions[sid] is not None
        ):
            src_pos = np.asarray(source_positions[sid], dtype=float)
            used_arr = np.asarray(used_mics, dtype=int)
            dists = np.linalg.norm(mic_positions[used_arr] - src_pos[None, :], axis=1)
            source_ref_mic = int(used_arr[int(np.argmin(dists))])
            ref_reason = "closest_used_mic_to_estimated_source"
        elif fallback_ref == "first_used":
            source_ref_mic = int(used_mics[0])
            ref_reason = "first_used_mic"
        else:
            raise ValueError(f"Unknown fallback_ref: {fallback_ref}")

        for mem_idx, mem in enumerate(src.get("members", [])):
            pair = mem.get("pair")
            if pair is None or len(pair) < 2:
                continue

            anchor_mic = int(pair[0])

            pair_key = tuple(pair)
            peak_idx = int(mem.get("peak_idx", -1))
            pair_data = tdoas_hat_pairs.get(pair_key)
            if pair_data is None:
                continue
            peaks = pair_data.get("peaks", [])
            if peak_idx < 0 or peak_idx >= len(peaks):
                continue

            if reject_ambiguous_pair_tf:
                tolerance = pair_ambiguity_tolerance_samples
                if tolerance is None:
                    tolerance = pair_data.get("support_tolerance_samples")*3*2 # This is just Qw
                if tolerance is not None:
                    tolerance = float(tolerance)
                    source_position = (
                        source_positions[sid]
                        if sid < len(source_positions)
                        else None
                    )
                    predicted_tau = (
                        predicted_tdoa_samples(source_position, pair_key)
                        if source_position is not None
                        else None
                    )
                    closest_source = None
                    closest_delta = np.inf
                    closest_tau = None
                    if predicted_tau is not None:
                        for other_sid, other_position in enumerate(
                            source_positions
                        ):
                            if other_sid == sid or other_position is None:
                                continue
                            other_tau = predicted_tdoa_samples(
                                other_position,
                                pair_key,
                            )
                            if other_tau is None:
                                continue
                            delta = abs(predicted_tau - other_tau)
                            if delta < closest_delta:
                                closest_delta = float(delta)
                                closest_source = int(other_sid)
                                closest_tau = float(other_tau)

                    if (
                        closest_source is not None
                        and closest_delta <= tolerance
                    ):
                        src_debug.append({
                            "member_idx": int(mem_idx),
                            "pair": pair_key,
                            "anchor_mic": int(anchor_mic),
                            "source_ref_mic": int(source_ref_mic),
                            "predicted_tau_samples": float(predicted_tau),
                            "ambiguous_with_source": closest_source,
                            "other_predicted_tau_samples": closest_tau,
                            "ambiguity_delta_samples": closest_delta,
                            "ambiguity_tolerance_samples": tolerance,
                            "reason": "ambiguous_predicted_tdoa_tf_rejected",
                        })
                        continue

            member_map = np.zeros(
                (int(n_bands), int(n_frames)),
                dtype=np.float32,
            )
            for b, t in peaks[peak_idx].get("tf_cells", []):
                b = int(b)
                t = int(t)
                if 0 <= b < n_bands and 0 <= t < n_frames:
                    member_map[b, t] += 1.0
            member_max = float(member_map.max())
            if member_max > 0:
                member_map /= member_max

            shift_samples = float(delays[source_ref_mic] - delays[anchor_mic])
            shift_frames = int(np.rint(shift_samples / float(hop)))

            member_map_shifted = shift_tf_map_time_zero_fill(member_map, shift_frames)
            src_map = np.maximum(src_map, member_map_shifted)

            src_debug.append({
                "member_idx": int(mem_idx),
                "anchor_mic": int(anchor_mic),
                "source_ref_mic": int(source_ref_mic),
                "shift_samples": float(shift_samples),
                "shift_frames": int(shift_frames),
                "reason": "ok",
            })

        aligned_maps.append(src_map)
        source_ref_mics.append(int(source_ref_mic))
        debug.append({
            "source_idx": int(sid),
            "source_ref_mic": int(source_ref_mic),
            "used_mics": used_mics,
            "members": src_debug,
            "n_ambiguous_pair_tf_rejected": sum(
                member.get("reason")
                == "ambiguous_predicted_tdoa_tf_rejected"
                for member in src_debug
            ),
            "reason": ref_reason,
        })

    return aligned_maps, source_ref_mics, debug


def shift_source_maps_between_mics(source_maps, sources, from_mic, target_mics, n_mics, hop):
    """
    Shift already-merged source maps from a known microphone reference to one
    target microphone per source.

    Use this after source maps have been built aligned to `from_mic`, for example
    to move maps from the plotting reference mic to each source's closest
    reconstruction mic.
    """
    shifted_maps = []
    debug = []
    if isinstance(from_mic, (int, np.integer)) or from_mic is None:
        from_mics = [from_mic] * len(source_maps)
    else:
        from_mics = list(from_mic)

    for sid, M in enumerate(source_maps):
        src = sources[sid] if sid < len(sources) else {}
        src_from_mic = from_mics[sid] if sid < len(from_mics) else None
        target_mic = int(target_mics[sid]) if sid < len(target_mics) else src_from_mic

        delays, used_mics = solve_relative_delays_from_source(
            src,
            n_mics=n_mics,
            reference="mean",
        )

        if delays is None or src_from_mic is None or target_mic is None:
            shift_samples = 0.0
            shift_frames = 0
            reason = "missing_delays_or_reference"
        else:
            src_from_mic = int(src_from_mic)
            target_mic = int(target_mic)
            shift_samples = float(delays[target_mic] - delays[src_from_mic])
            shift_frames = int(np.rint(shift_samples / float(hop)))
            reason = "ok"

        shifted_maps.append(shift_tf_map_time_zero_fill(M, shift_frames))
        debug.append({
            "source_idx": int(sid),
            "from_mic": (None if src_from_mic is None else int(src_from_mic)),
            "target_mic": (None if target_mic is None else int(target_mic)),
            "shift_samples": float(shift_samples),
            "shift_frames": int(shift_frames),
            "reason": reason,
        })

    return shifted_maps, debug


def select_reference_mics_by_source_roi_overlap(
    source_maps,
    mic_valid_mask,
    sources,
    from_mic,
    n_mics,
    hop,
    candidate_mics_by_source=None,
    fallback_mics=None,
    eps=1e-12,
):
    """
    Select one real microphone per source by source-mask/ROI overlap.

    Each source map is shifted from its source reference mic to every candidate
    mic. The selected mic is the one whose per-mic ROI mask overlaps the shifted
    source map most strongly, normalized by the source-map mass. If no candidate
    has positive overlap, the corresponding fallback mic is used.
    """
    mic_valid_mask = np.asarray(mic_valid_mask, dtype=np.float32)
    if mic_valid_mask.ndim != 3:
        raise ValueError("mic_valid_mask must have shape (n_mics, bands, frames)")
    if mic_valid_mask.shape[0] != int(n_mics):
        raise ValueError(
            f"mic_valid_mask has {mic_valid_mask.shape[0]} mics but n_mics={n_mics}"
        )

    if isinstance(from_mic, (int, np.integer)) or from_mic is None:
        from_mics = [from_mic] * len(source_maps)
    else:
        from_mics = list(from_mic)

    if fallback_mics is None:
        fallback_mics = [0] * len(source_maps)
    else:
        fallback_mics = list(fallback_mics)

    selected_mics = []
    debug = []

    for sid, source_map in enumerate(source_maps):
        src = sources[sid] if sid < len(sources) else {}
        src_from_mic = from_mics[sid] if sid < len(from_mics) else None
        fallback = int(fallback_mics[sid]) if sid < len(fallback_mics) else 0

        if candidate_mics_by_source is None:
            candidates = list(range(int(n_mics)))
        else:
            candidates = candidate_mics_by_source[sid]
            candidates = list(candidates) if candidates is not None else []
            if len(candidates) == 0:
                candidates = [fallback]

        candidates = sorted({
            int(m) for m in candidates
            if 0 <= int(m) < int(n_mics)
        })
        if len(candidates) == 0:
            candidates = [fallback]

        delays, _ = solve_relative_delays_from_source(
            src,
            n_mics=n_mics,
            reference="mean",
        )

        scores = []
        best_mic = fallback
        best_score = -np.inf
        best_shift_frames = 0

        for mic in candidates:
            if delays is None or src_from_mic is None:
                shift_samples = 0.0
                shift_frames = 0
            else:
                shift_samples = float(delays[mic] - delays[int(src_from_mic)])
                shift_frames = int(np.rint(shift_samples / float(hop)))

            shifted_map = shift_tf_map_time_zero_fill(source_map, shift_frames)
            if shifted_map.shape != mic_valid_mask[mic].shape:
                common_bands = min(shifted_map.shape[0], mic_valid_mask[mic].shape[0])
                common_frames = min(shifted_map.shape[1], mic_valid_mask[mic].shape[1])
                shifted_eval = shifted_map[:common_bands, :common_frames]
                roi_eval = mic_valid_mask[mic, :common_bands, :common_frames]
            else:
                shifted_eval = shifted_map
                roi_eval = mic_valid_mask[mic]

            source_mass = float(np.sum(np.clip(shifted_eval, 0.0, 1.0)))
            overlap = float(np.sum(np.clip(shifted_eval, 0.0, 1.0) * (roi_eval > 0)))
            score = overlap / (source_mass + eps)

            scores.append({
                "mic": int(mic),
                "score": float(score),
                "overlap": float(overlap),
                "source_mass": float(source_mass),
                "shift_frames": int(shift_frames),
                "shift_samples": float(shift_samples),
            })

            if score > best_score:
                best_score = score
                best_mic = int(mic)
                best_shift_frames = int(shift_frames)

        if best_score <= 0.0:
            best_mic = fallback

        selected_mics.append(int(best_mic))
        debug.append({
            "source_idx": int(sid),
            "selected_mic": int(best_mic),
            "fallback_mic": int(fallback),
            "selected_score": float(best_score),
            "selected_shift_frames": int(best_shift_frames),
            "candidates": scores,
            "reason": "roi_overlap" if best_score > 0.0 else "fallback_no_overlap",
        })

    return selected_mics, debug
