"""Source position estimation from associated TDOAs."""

from __future__ import annotations

import numpy as np

from ._kernels.localization import (
    estimate_all_sources_positions,
    include_mic_position_uncertainty,
)

from ._geometry import default_environment_bounds
from .config import LocalizationConfig, LocalTDOAConfig
from .models import AssociationResult, LocalizationResult, Recording, SourceLocation


def localize_sources(
    association: AssociationResult,
    recording: Recording,
    config: LocalizationConfig | None = None,
    local_tdoa_config: LocalTDOAConfig | None = None,
) -> LocalizationResult:
    """Estimate one position and quality record per associated source."""
    config = config or LocalizationConfig()
    local_config = local_tdoa_config or LocalTDOAConfig()
    if not association.raw_sources:
        return LocalizationResult(sources=[])
    if config.environment_bounds is None:
        bounds = default_environment_bounds(recording.microphone_positions)
    else:
        bounds = (
            np.asarray(config.environment_bounds[0], dtype=float),
            np.asarray(config.environment_bounds[1], dtype=float),
        )
    grid_spacing = config.grid_spacing_m
    if grid_spacing is None:
        grid_spacing = max(float(np.max(bounds[1] - bounds[0])) / 4.0, 1.0)
    f_scale = (
        2.0 / recording.sample_rate
        if config.f_scale_seconds is None
        else config.f_scale_seconds
    )
    raw_estimates = estimate_all_sources_positions(
        sources=association.raw_sources,
        mic_positions=recording.microphone_positions,
        fs=recording.sample_rate,
        c=local_config.speed_of_sound_m_s,
        env_bounds=bounds,
        loss=config.loss,
        f_scale=f_scale,
        multistart=config.multistart,
        grid_spacing=grid_spacing,
        n_starts=config.n_starts,
    )
    include_mic_position_uncertainty(
        raw_estimates,
        local_config.microphone_position_uncertainty_m,
    )
    output = []
    for source_id, estimate in enumerate(raw_estimates):
        info = estimate.get("info", {})
        position = estimate.get("x_hat")
        uncertainty = info.get("location_uncertainty_m")
        residual = info.get("residual_rms_samples")
        output.append(
            SourceLocation(
                source_id=source_id,
                position=(None if position is None else np.asarray(position, dtype=float)),
                uncertainty_m=(None if uncertainty is None else float(uncertainty)),
                residual_samples=(None if residual is None else float(residual)),
                boundary_contact=bool(info.get("boundary_contact", False)),
                raw=estimate,
            )
        )
    return LocalizationResult(sources=output)
