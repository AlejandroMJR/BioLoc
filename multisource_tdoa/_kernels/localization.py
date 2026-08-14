import numpy as np
from scipy.optimize import least_squares


def estimate_source_position_from_tdoas(
    mic_positions,
    tdoa_measurements,
    fs,
    c=343.0,
    x0=None,
    bounds=None,
    loss="huber",
    f_scale=0.002,
    max_nfev=2000,
    multistart=False,
    grid_spacing=10.0,
    n_starts=5,
    position_uncertainty_m=0.0,
):
    """
    Estimate a single source position from associated TDOA constraints.

    Model:
        (||x - m_i|| - ||x - m_j||) / c  =  tau_ij   (seconds)
    """
    mics = np.asarray(mic_positions, dtype=float)
    M, D = mics.shape

    # Parse measurements into arrays
    pairs_i = []
    pairs_j = []
    tau_sec = []

    for meas in tdoa_measurements:
        i = int(meas["i"]) if isinstance(meas, dict) else int(meas[0])
        j = int(meas["j"]) if isinstance(meas, dict) else int(meas[1])

        if isinstance(meas, dict):
            if "tau_seconds" in meas:
                tau = float(meas["tau_seconds"])
            else:
                tau = float(meas["tau_samples"]) / float(fs)
        else:
            tau = float(meas[2]) / float(fs)

        pairs_i.append(i)
        pairs_j.append(j)
        tau_sec.append(tau)

    pairs_i = np.asarray(pairs_i, dtype=int)
    pairs_j = np.asarray(pairs_j, dtype=int)
    tau_sec = np.asarray(tau_sec, dtype=float)

    # if tau_sec.size < D + 1: ### Already handled by association step
    #     raise ValueError(
    #         f"Need more TDOA constraints. Got {tau_sec.size}, recommended >= {D+1}."
    #     )

    # Bounds
    if bounds is None:
        lb = -np.inf * np.ones(D)
        ub = np.inf * np.ones(D)
        bounds = (lb, ub)
    else:
        bounds = (
            np.asarray(bounds[0], dtype=float),
            np.asarray(bounds[1], dtype=float),
        )

    position_uncertainty_m = float(position_uncertainty_m)
    if position_uncertainty_m < 0:
        raise ValueError("position_uncertainty_m must be non-negative")

    def apply_uncertainty_deadzone(distance_residuals):
        if position_uncertainty_m <= 0:
            return distance_residuals
        return np.sign(distance_residuals) * np.maximum(
            np.abs(distance_residuals) - position_uncertainty_m,
            0.0,
        )

    # Residual function (seconds), with an optional dead zone expressed in
    # distance units to represent uncertainty in the measured mic positions.
    def residuals(x):
        x = np.asarray(x, dtype=float)
        di = np.linalg.norm(x - mics[pairs_i], axis=1)
        dj = np.linalg.norm(x - mics[pairs_j], axis=1)
        distance_residuals = (di - dj) - float(c) * tau_sec
        return apply_uncertainty_deadzone(distance_residuals) / float(c)

    if x0 is None:
        x0 = np.mean(mics, axis=0)
    x0 = np.asarray(x0, dtype=float)

    if multistart:
        lb, ub = bounds
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise ValueError("multistart localization requires finite bounds")
        if float(grid_spacing) <= 0:
            raise ValueError("grid_spacing must be positive")
        if int(n_starts) < 1:
            raise ValueError("n_starts must be at least 1")

        axes = []
        for lo, hi in zip(lb, ub):
            axis = np.arange(lo, hi + 0.5 * float(grid_spacing), grid_spacing)
            axis = np.clip(axis, lo, hi)
            axes.append(np.unique(np.concatenate([axis, [hi]])))

        mesh = np.meshgrid(*axes, indexing="ij")
        grid = np.stack([axis.ravel() for axis in mesh], axis=1)
        grid = np.vstack([grid, np.clip(x0, lb, ub)])

        di = np.linalg.norm(
            grid[:, None, :] - mics[pairs_i][None, :, :],
            axis=2,
        )
        dj = np.linalg.norm(
            grid[:, None, :] - mics[pairs_j][None, :, :],
            axis=2,
        )
        grid_distance_residuals = (
            (di - dj) - float(c) * tau_sec[None, :]
        )
        grid_residuals = (
            apply_uncertainty_deadzone(grid_distance_residuals) / float(c)
        )
        grid_scores = np.mean(grid_residuals ** 2, axis=1)
        selected = np.argsort(grid_scores)[:min(int(n_starts), len(grid))]
        start_points = grid[selected]
    else:
        start_points = x0[None, :]

    results = [
        least_squares(
            residuals,
            x0=start,
            bounds=bounds,
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
            jac="3-point",
        )
        for start in start_points
    ]
    final_scores = [
        float(np.sum(np.asarray(result.fun, dtype=float) ** 2))
        for result in results
    ]
    best_idx = int(np.argmin(final_scores))
    res = results[best_idx]

    # One compact, approximate position-quality value. Convert the fitted
    # TDOA residual to path-length error and amplify it by the weakest
    # localization direction of the microphone geometry. A one-sample floor
    # prevents an exactly determined solution from claiming zero uncertainty.
    x_hat = np.asarray(res.x, dtype=float)
    di_vectors = x_hat - mics[pairs_i]
    dj_vectors = x_hat - mics[pairs_j]
    di_norms = np.maximum(
        np.linalg.norm(di_vectors, axis=1, keepdims=True),
        1e-12,
    )
    dj_norms = np.maximum(
        np.linalg.norm(dj_vectors, axis=1, keepdims=True),
        1e-12,
    )
    geometry_jacobian = di_vectors / di_norms - dj_vectors / dj_norms
    singular_values = np.linalg.svd(
        geometry_jacobian,
        compute_uv=False,
    )
    geometry_rank = int(np.linalg.matrix_rank(geometry_jacobian))
    if (
        geometry_rank < D
        or singular_values.size < D
        or float(singular_values[-1]) <= 1e-12
    ):
        geometry_dilution = np.inf
    else:
        geometry_covariance = np.linalg.pinv(
            geometry_jacobian.T @ geometry_jacobian,
            rcond=1e-12,
        )
        geometry_dilution = float(np.sqrt(np.max(
            np.linalg.eigvalsh(geometry_covariance)
        )))

    residuals_samples = np.asarray(res.fun, dtype=float) * float(fs)
    residual_rms_samples = float(np.sqrt(np.mean(
        residuals_samples ** 2
    )))
    residual_distance_m = (
        max(residual_rms_samples, 1.0) * float(c) / float(fs)
    )
    # sqrt(chi2.ppf(0.95, df=2)): approximate 95% worst-direction radius.
    location_uncertainty_m = (
        np.sqrt(5.991464547) * residual_distance_m * geometry_dilution
    )

    active_mask = np.asarray(
        getattr(res, "active_mask", np.zeros(D, dtype=int)),
        dtype=int,
    )
    boundary_contact = bool(np.any(active_mask != 0))
    if not boundary_contact:
        lb, ub = bounds
        finite_extent = np.abs(ub - lb)
        boundary_tolerance = np.maximum(
            1e-6,
            1e-8 * np.where(np.isfinite(finite_extent), finite_extent, 0.0),
        )
        boundary_contact = bool(np.any(
            (np.isfinite(lb) & (x_hat - lb <= boundary_tolerance))
            | (np.isfinite(ub) & (ub - x_hat <= boundary_tolerance))
        ))

    info = {
        "success": bool(res.success),
        "message": res.message,
        "cost": float(res.cost),
        "residuals_sec": res.fun,
        "jacobian": res.jac,
        "nfev": res.nfev,
        "multistart": bool(multistart),
        "n_starts": int(len(start_points)),
        "selected_start": np.asarray(start_points[best_idx], dtype=float),
        "final_squared_residual": final_scores[best_idx],
        "position_uncertainty_m": position_uncertainty_m,
        "residual_rms_samples": residual_rms_samples,
        "geometry_dilution": geometry_dilution,
        "location_uncertainty_without_position_m": float(
            location_uncertainty_m
        ),
        "location_uncertainty_m": float(location_uncertainty_m),
        "boundary_contact": boundary_contact,
    }
    return res.x, info


def include_mic_position_uncertainty(
    estimates,
    mic_position_uncertainty_m,
):
    """
    Add configured microphone-position uncertainty to location uncertainty.

    The two microphones in a TDOA pair contribute approximately
    ``sqrt(2) * mic_position_uncertainty_m`` path-difference uncertainty.
    This is combined in quadrature with the residual-based value.
    """
    mic_position_uncertainty_m = float(mic_position_uncertainty_m)
    if mic_position_uncertainty_m < 0:
        raise ValueError("mic_position_uncertainty_m must be non-negative")

    pair_uncertainty_m = np.sqrt(2.0) * mic_position_uncertainty_m
    confidence_scale = np.sqrt(5.991464547)
    for estimate in estimates:
        info = estimate.setdefault("info", {})
        base_uncertainty_m = float(info.get(
            "location_uncertainty_without_position_m",
            info.get("location_uncertainty_m", np.inf),
        ))
        geometry_dilution = float(
            info.get("geometry_dilution", np.inf)
        )
        position_component_m = (
            confidence_scale * pair_uncertainty_m * geometry_dilution
        )
        info["pair_mic_position_uncertainty_m"] = pair_uncertainty_m
        info["location_uncertainty_m"] = float(np.hypot(
            base_uncertainty_m,
            position_component_m,
        ))
    return estimates


def estimate_all_sources_positions(
    sources,
    mic_positions,
    fs,
    c=343.0,
    env_bounds=None,
    loss="huber",
    f_scale=0.002,
    multistart=False,
    grid_spacing=10.0,
    n_starts=5,
    position_uncertainty_m=0.0,
):
    """
    Estimate positions for all associated sources

    Parameters
    ----------
    sources : list of dict
         Clustered/associated sources.
    mic_positions : (M,D)
    fs : float
    env_bounds : tuple(lb, ub) or None
        Bounds for solver. For 2D square env:
          lb=[0,0], ub=[env_size, env_size]
    loss, f_scale : see estimate_source_position_from_tdoas
    multistart : bool
        If True, rank a coarse grid of finite env_bounds and optimize from the
        best n_starts candidates.
    grid_spacing : float
        Coarse-grid spacing in the same distance units as mic_positions.
    n_starts : int
        Number of coarse-grid candidates refined with least_squares.

    Returns
    -------
    estimates : list of dict
        Each entry:
          {
            "x_hat": (D,),
            "info": solver_info,
            "n_constraints": K
          }
    """
    estimates = []
    for src in sources:
        tdoa_meas = []
        for mem in src.get("members", []):
            i, j = mem["pair"]
            tdoa_meas.append({"i": int(i), "j": int(j), "tau_samples": float(mem["tau"])})

        if len(tdoa_meas) == 0:
            estimates.append({"x_hat": None, "info": {"success": False, "message": "No members"}, "n_constraints": 0})
            continue

        x_hat, info = estimate_source_position_from_tdoas(
            mic_positions=mic_positions,
            tdoa_measurements=tdoa_meas,
            fs=fs,
            c=c,
            x0=None,
            bounds=env_bounds,
            loss=loss,
            f_scale=f_scale,
            multistart=multistart,
            grid_spacing=grid_spacing,
            n_starts=n_starts,
            position_uncertainty_m=position_uncertainty_m,
        )

        estimates.append(
            {"x_hat": x_hat, "info": info, "n_constraints": len(tdoa_meas)}
        )

    return estimates


def fit_source_and_geometric_residuals(
    source,
    mic_positions,
    fs,
    c,
    env_bounds,
    grid_spacing=None,
    n_starts=5,
    position_uncertainty_m=0.0,
):
    """Fit one source and return uncertainty-adjusted residuals in samples."""
    if grid_spacing is None:
        grid_spacing = float(env_bounds[1][0]) // 4
        if grid_spacing <= 0:
            grid_spacing = float(env_bounds[1][0]) / 4.0

    estimate = estimate_all_sources_positions(
        sources=[source],
        mic_positions=mic_positions,
        fs=fs,
        c=c,
        env_bounds=env_bounds,
        loss="linear",
        f_scale=2.0 / fs,
        multistart=True,
        grid_spacing=grid_spacing,
        n_starts=n_starts,
        position_uncertainty_m=position_uncertainty_m,
    )[0]
    x_hat = estimate.get("x_hat")
    if x_hat is None:
        return estimate, np.full(len(source.get("members", [])), np.inf)

    x_hat = np.asarray(x_hat, dtype=float)
    mic_positions = np.asarray(mic_positions, dtype=float)
    residuals = []
    for member in source.get("members", []):
        i, j = map(int, member["pair"])
        di = np.linalg.norm(x_hat - mic_positions[i])
        dj = np.linalg.norm(x_hat - mic_positions[j])
        tau_pred_samples = ((di - dj) / float(c)) * float(fs)
        raw_residual_m = (
            (tau_pred_samples - float(member["tau"])) * float(c) / float(fs)
        )
        adjusted_residual_m = max(
            0.0,
            abs(raw_residual_m) - float(position_uncertainty_m),
        )
        residuals.append(adjusted_residual_m * float(fs) / float(c))

    return estimate, np.asarray(residuals, dtype=float)


def prune_source_geometric_outliers(
    source,
    mic_positions,
    fs,
    c,
    env_bounds,
    min_tdoas_per_source,
    min_relative_improvement=0.2,
    max_residual_samples=3.0,
    grid_spacing=None,
    n_starts=5,
    position_uncertainty_m=0.0,
):
    """
    Iteratively remove a TDOA when leave-one-out geometry improves enough.

    An accepted intermediate removal does not need to bring the maximum
    residual below the final threshold immediately. This allows multiple
    outliers to be removed over successive iterations while never going below
    ``min_tdoas_per_source``.
    """
    working = dict(source)
    working["members"] = list(source.get("members", []))
    removed = []
    eps = 1e-12

    while len(working["members"]) > int(min_tdoas_per_source):
        _, current_residuals = fit_source_and_geometric_residuals(
            working,
            mic_positions,
            fs,
            c,
            env_bounds,
            grid_spacing,
            n_starts,
            position_uncertainty_m,
        )
        current_rms = float(np.sqrt(np.mean(current_residuals ** 2)))
        current_max = float(np.max(current_residuals))
        if current_max <= float(max_residual_samples):
            break

        best = None
        for remove_idx, member in enumerate(working["members"]):
            candidate = dict(working)
            candidate["members"] = [
                kept
                for idx, kept in enumerate(working["members"])
                if idx != remove_idx
            ]
            candidate_estimate, candidate_residuals = (
                fit_source_and_geometric_residuals(
                    candidate,
                    mic_positions,
                    fs,
                    c,
                    env_bounds,
                    grid_spacing,
                    n_starts,
                    position_uncertainty_m,
                )
            )
            candidate_rms = float(np.sqrt(np.mean(candidate_residuals ** 2)))
            candidate_max = float(np.max(candidate_residuals))
            relative_improvement = (
                current_rms - candidate_rms
            ) / max(current_rms, eps)

            if (
                relative_improvement >= float(min_relative_improvement)
                and (
                    best is None
                    or candidate_rms < best["rms_residual_samples"]
                )
            ):
                best = {
                    "member": member,
                    "source": candidate,
                    "estimate": candidate_estimate,
                    "residuals": candidate_residuals,
                    "rms_residual_samples": candidate_rms,
                    "max_residual_samples": candidate_max,
                    "relative_improvement": float(relative_improvement),
                    "previous_rms_residual_samples": current_rms,
                }

        if best is None:
            break

        removed.append({
            "pair": tuple(best["member"]["pair"]),
            "tau": float(best["member"]["tau"]),
            "previous_rms_residual_samples": best[
                "previous_rms_residual_samples"
            ],
            "new_rms_residual_samples": best["rms_residual_samples"],
            "new_max_residual_samples": best["max_residual_samples"],
            "relative_improvement": best["relative_improvement"],
        })
        working = best["source"]

    final_estimate, final_residuals = fit_source_and_geometric_residuals(
        working,
        mic_positions,
        fs,
        c,
        env_bounds,
        grid_spacing,
        n_starts,
        position_uncertainty_m,
    )
    final_rms = float(np.sqrt(np.mean(final_residuals ** 2)))
    final_max = float(np.max(final_residuals))
    keep = (
        len(working["members"]) >= int(min_tdoas_per_source)
        and final_max <= float(max_residual_samples)
    )
    return {
        "source": working,
        "estimate": final_estimate,
        "residuals_samples": final_residuals,
        "rms_residual_samples": final_rms,
        "max_residual_samples": final_max,
        "removed": removed,
        "keep": bool(keep),
    }


def prune_geometrically_inconsistent_sources(
    sources,
    mic_positions,
    fs,
    c,
    env_bounds,
    min_tdoas_per_source,
    min_relative_improvement=0.2,
    max_residual_samples=3.0,
    grid_spacing=None,
    n_starts=5,
    position_uncertainty_m=0.0,
    verbose=True,
):
    """Apply geometric leave-one-out pruning independently to each source."""
    kept_sources = []
    reports = []

    for source_idx, source in enumerate(sources):
        report = prune_source_geometric_outliers(
            source,
            mic_positions,
            fs,
            c,
            env_bounds,
            min_tdoas_per_source,
            min_relative_improvement=min_relative_improvement,
            max_residual_samples=max_residual_samples,
            grid_spacing=grid_spacing,
            n_starts=n_starts,
            position_uncertainty_m=position_uncertainty_m,
        )
        report["original_source_idx"] = int(source_idx)
        reports.append(report)

        if verbose:
            for removed in report["removed"]:
                print(
                    f"[GEOMETRY] Source {source_idx}: removed pair "
                    f"{removed['pair']} tau={removed['tau']:.2f} samples; "
                    f"RMS {removed['previous_rms_residual_samples']:.2f} -> "
                    f"{removed['new_rms_residual_samples']:.2f} samples "
                    f"({100.0 * removed['relative_improvement']:.1f}% better), "
                    f"max={removed['new_max_residual_samples']:.2f}"
                )

        if report["keep"]:
            kept_sources.append(report["source"])
            if verbose:
                print(
                    f"[GEOMETRY] Source {source_idx}: kept "
                    f"{len(report['source']['members'])} TDOAs, "
                    f"RMS={report['rms_residual_samples']:.2f}, "
                    f"max={report['max_residual_samples']:.2f} samples."
                )
        else:
            if verbose:
                print(
                    f"[GEOMETRY] Source {source_idx}: removed source; final "
                    f"{len(report['source']['members'])} TDOAs, "
                    f"RMS={report['rms_residual_samples']:.2f}, "
                    f"max={report['max_residual_samples']:.2f} samples."
                )

    return kept_sources, reports
