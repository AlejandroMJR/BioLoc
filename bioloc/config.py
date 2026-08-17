"""Configuration objects for each public pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeneralConfig:
    lowcut_hz: float | None = None
    highcut_hz: float | None = 10_000.0
    speed_of_sound_m_s: float = 333.0
    microphone_position_uncertainty_m: float = 2.0


@dataclass(frozen=True)
class ROIConfig:
    bin_std: float = 10.0
    bin_per: float = 0.5
    min_roi: int = 10


@dataclass(frozen=True)
class LocalTDOAConfig:
    band_width_hz: float = 800.0
    frame_size: int = 2048
    k_neighbors: int = 3
    filter_mode: str = "fft_mask"
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
    frequency_weight: float = 0.5
    prune_geometric_outliers: bool = True
    prune_min_relative_improvement: float = 0.3


@dataclass(frozen=True)
class LocalizationConfig:
    loss: str = "linear"
    multistart: bool = True
    grid_spacing_m: float | None = None
    n_starts: int = 5


@dataclass(frozen=True)
class ReconstructionConfig:
    reconstruction_frame_size: int = 2048
    reject_ambiguous_pair_tf: bool = True
    pair_ambiguity_tolerance_samples: float | None = None


@dataclass(frozen=True)
class ClassificationConfig:
    backend: str = "birdnet"
    min_confidence: float = 0.1
    perch_model: str = "perch_8"
    perch_hop_duration: float = 1.0
    perch_warm_up: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    local_tdoa: LocalTDOAConfig = field(default_factory=LocalTDOAConfig)
    peaks: PeakExtractionConfig = field(default_factory=PeakExtractionConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)


_CONFIG_DESCRIPTIONS = {
    "general": {
        "lowcut_hz": "Lower frequency limit in Hz; None keeps frequencies down to 0 Hz.",
        "highcut_hz": "Upper frequency limit in Hz; None keeps frequencies to Nyquist.",
        "speed_of_sound_m_s": "Speed of sound used for TDOA geometry, in metres per second.",
        "microphone_position_uncertainty_m": "Expected uncertainty of each microphone position, in metres.",
    },
    "roi": {
        "bin_std": "Relative threshold width used to create the MAAD ROI mask.",
        "bin_per": "Proportion parameter used by the MAAD relative threshold.",
        "min_roi": "Minimum connected ROI area, in time-frequency cells.",
    },
    "local_tdoa": {
        "band_width_hz": "Width of each overlapping frequency band in Hz.",
        "frame_size": "Local-TDOA analysis frame and FFT size in samples.",
        "k_neighbors": "Number of nearest microphones connected to each microphone.",
        "filter_mode": "Per-band filter: 'fft_mask' or 'butterworth'.",
        "workers": "Number of frequency bands processed concurrently.",
        "use_roi_mask": "Restrict local-TDOA estimation to projected ROI cells.",
    },
    "peaks": {
        "narrow_width_divisor": "Derives the narrow matching-pursuit region Qn from maximum delay.",
        "wide_to_narrow_ratio": "Ratio used to derive Qw from Qn.",
        "histogram_sigma_divisor": "Derives Gaussian histogram smoothing from Qw.",
        "support_tolerance_divisor": "Derives peak-observation support tolerance from Qn.",
    },
    "association": {
        "feature": "Association representation: frequency, time_frequency, or time_frequency_candidates.",
        "cosine_threshold": "Minimum cross-pair feature cosine similarity.",
        "same_pair_cosine_threshold": "Minimum similarity for candidates from the same pair.",
        "frequency_weight": "Frequency contribution to time-frequency similarity, from 0 to 1.",
        "prune_geometric_outliers": "Remove associated TDOAs that are geometrically inconsistent.",
        "prune_min_relative_improvement": "Required residual improvement when pruning an outlier.",
    },
    "localization": {
        "loss": "Least-squares loss used by source localization.",
        "multistart": "Run localization from multiple initial positions.",
        "grid_spacing_m": "Initial-search grid spacing in metres; None derives it from the environment.",
        "n_starts": "Number of best grid positions refined by localization.",
    },
    "reconstruction": {
        "reconstruction_frame_size": "Reconstruction STFT frame and FFT size in samples.",
        "reject_ambiguous_pair_tf": "Reject pair/time-frequency evidence shared by nearby source delays.",
        "pair_ambiguity_tolerance_samples": "Ambiguity tolerance in samples; None derives it per pair.",
    },
    "classification": {
        "backend": "Classifier backend: 'birdnet' or 'perch'.",
        "min_confidence": "Minimum retained classifier confidence, from 0 to 1.",
        "perch_model": "Perch model name used when backend='perch'.",
        "perch_hop_duration": "Perch inference hop duration in seconds.",
        "perch_warm_up": "Warm up compatible Perch models before pipeline processing.",
    },
}


def config_help(section: str | None = None) -> str:
    """Return readable help and defaults for public pipeline parameters."""
    config = PipelineConfig()
    requested = None if section is None else str(section).strip().lower().replace("-", "_")
    if requested in (None, "", "all"):
        section_names = list(_CONFIG_DESCRIPTIONS)
    elif requested in _CONFIG_DESCRIPTIONS:
        section_names = [requested]
    else:
        valid = ", ".join(_CONFIG_DESCRIPTIONS)
        raise ValueError(f"Unknown configuration section {section!r}; choose from {valid}")

    lines = ["BioLoc pipeline parameters"]
    for section_name in section_names:
        values = getattr(config, section_name)
        lines.append(f"\n{section_name}")
        for parameter, description in _CONFIG_DESCRIPTIONS[section_name].items():
            lines.append(f"  {parameter} = {getattr(values, parameter)!r}")
            lines.append(f"    {description}")
    return "\n".join(lines)
