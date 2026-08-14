"""Configuration objects for each public pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ROIConfig:
    nperseg: int = 1024
    noverlap: int = 512
    db_range: float = 96.0
    smooth_std: float = 0.5
    bin_std: float = 10.0
    bin_per: float = 0.5
    min_roi: int = 10
    max_roi: int | None = None


@dataclass(frozen=True)
class LocalTDOAConfig:
    lowcut_hz: float | None = None
    highcut_hz: float | None = 10_000.0
    band_width_hz: float = 800.0
    band_overlap_hz: float | None = None
    frame_size: int = 2048
    k_neighbors: int = 3
    speed_of_sound_m_s: float = 333.0
    microphone_position_uncertainty_m: float = 2.0
    filter_mode: str = "fft_mask"
    correlation_mode: str = "norm_cc"
    workers: int = 4
    use_roi_mask: bool = True


@dataclass(frozen=True)
class PeakExtractionConfig:
    narrow_width_divisor: float = 40.0
    wide_to_narrow_ratio: float = 3.0
    histogram_sigma_divisor: float = 20.0
    support_tolerance_divisor: float = 3.0


@dataclass(frozen=True)
class AssociationConfig:
    feature: str = "time_frequency_candidates"
    cosine_threshold: float = 0.7
    same_pair_cosine_threshold: float = 0.7
    frames_per_bin: int = 1
    frequency_weight: float = 0.5
    max_exact_combinations: int = 10_000
    prune_geometric_outliers: bool = True
    prune_min_relative_improvement: float = 0.3
    environment_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class LocalizationConfig:
    loss: str = "linear"
    f_scale_seconds: float | None = None
    multistart: bool = True
    grid_spacing_m: float | None = None
    n_starts: int = 5
    environment_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class ReconstructionConfig:
    frame_size: int = 2048
    reject_ambiguous_pair_tf: bool = True
    pair_ambiguity_tolerance_samples: float | None = None
    morph_threshold: float = 0.0
    morph_core_gain: float = 1.0
    morph_context_gain: float = 1.0
    morph_min_area: int = 4
    morph_open_freq_hz: float = 0.0
    morph_open_time_seconds: float = 0.0
    morph_close_freq_hz: float = 0.0
    morph_close_time_seconds: float = 0.0
    morph_dilate_freq_hz: float = 0.0
    morph_dilate_time_seconds: float = 0.0
    morph_post_close_freq_hz: float = 800.0
    morph_post_close_time_seconds: float = 0.1
    mask_taper_freq_hz: float = 0.0
    mask_taper_time_seconds: float = 0.0
    fill_mode: str = "white_noise"
    noise_level: float = 0.0
    target_snr_db: float = 0.0
    white_noise_level_mode: str = "inside_noise_floor"
    random_state: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    roi: ROIConfig = field(default_factory=ROIConfig)
    local_tdoa: LocalTDOAConfig = field(default_factory=LocalTDOAConfig)
    peaks: PeakExtractionConfig = field(default_factory=PeakExtractionConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
