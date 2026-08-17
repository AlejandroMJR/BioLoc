"""Public result objects for the BioLoc pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray


Pair = tuple[int, int]


@dataclass(frozen=True)
class Recording:
    """A synchronized microphone-array recording.

    Parameters
    ----------
    signals:
        Floating-point array shaped ``(n_microphones, n_samples)``.
    sample_rate:
        Sampling rate in hertz.
    microphone_positions:
        Microphone coordinates in metres, shaped ``(n_microphones, dimensions)``.
    microphone_names:
        Optional names in the same order as ``signals``.
    """

    signals: NDArray[np.floating]
    sample_rate: float
    microphone_positions: NDArray[np.floating]
    microphone_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        signals = np.asarray(self.signals, dtype=float)
        positions = np.asarray(self.microphone_positions, dtype=float)
        if signals.ndim != 2:
            raise ValueError("signals must have shape (microphones, samples)")
        if positions.ndim != 2 or positions.shape[0] != signals.shape[0]:
            raise ValueError(
                "microphone_positions must have one row per microphone"
            )
        if signals.shape[1] == 0:
            raise ValueError("signals must contain at least one sample")
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.microphone_names and len(self.microphone_names) != signals.shape[0]:
            raise ValueError("microphone_names must match the number of signals")
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "microphone_positions", positions)
        object.__setattr__(self, "sample_rate", float(self.sample_rate))
        object.__setattr__(self, "microphone_names", tuple(self.microphone_names))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_microphones(self) -> int:
        return int(self.signals.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.signals.shape[1])


@dataclass
class ROIResult:
    """Region-of-interest masks on the native MAAD time-frequency grid."""

    masks_by_pair: dict[Pair, NDArray[np.bool_]]
    microphone_masks: list[NDArray[np.bool_]] | None = None
    filtered_signals: NDArray[np.floating] | None = field(default=None, repr=False)
    lowcut_hz: float | None = None
    highcut_hz: float | None = 10_000.0
    speed_of_sound_m_s: float = 333.0
    microphone_position_uncertainty_m: float = 2.0

    def for_pair(self, first: int, second: int) -> NDArray[np.bool_]:
        return self.masks_by_pair[_ordered_pair(first, second)]


@dataclass
class PairLocalTDOAs:
    pair: Pair
    delays: NDArray[np.floating]
    band_indices: NDArray[np.integer]
    frame_indices: NDArray[np.integer]
    max_delay_samples: int


@dataclass
class LocalTDOAResult:
    pairs: dict[Pair, PairLocalTDOAs]
    frequency_bands_hz: tuple[tuple[float, float], ...]
    frame_times_seconds: NDArray[np.floating]
    frame_size: int
    hop_size: int
    sample_rate: float
    algorithm_frequency_bins: tuple[NDArray[np.integer], ...] = field(
        repr=False
    )
    projected_roi_mask: NDArray[np.bool_] | None = field(
        default=None, repr=False
    )
    filtered_signals: NDArray[np.floating] | None = field(
        default=None, repr=False
    )
    speed_of_sound_m_s: float = 333.0
    microphone_position_uncertainty_m: float = 2.0

    def for_pair(self, first: int, second: int) -> PairLocalTDOAs:
        return self.pairs[_ordered_pair(first, second)]


@dataclass
class TDOAPeak:
    delay_samples: float
    supporting_observation_indices: NDArray[np.integer]
    band_support: NDArray[np.floating]
    time_frequency_support: NDArray[np.floating]
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class PairPeakResult:
    pair: Pair
    histogram: NDArray[np.floating]
    delay_bins: NDArray[np.floating]
    peaks: list[TDOAPeak]
    assignment_tolerance_samples: float
    matching_pursuit_residual: NDArray[np.floating] | None = field(
        default=None, repr=False
    )


@dataclass
class PeakExtractionResult:
    pairs: dict[Pair, PairPeakResult]

    def for_pair(self, first: int, second: int) -> PairPeakResult:
        return self.pairs[_ordered_pair(first, second)]


@dataclass
class PeakFeature:
    pair: Pair
    peak_index: int
    delay_samples: float
    vector: NDArray[np.floating]


@dataclass
class AssociatedTDOA:
    pair: Pair
    peak_index: int
    delay_samples: float
    feature: NDArray[np.floating]
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AssociatedSource:
    source_id: int
    members: list[AssociatedTDOA]
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AssociationResult:
    features: list[PeakFeature]
    sources: list[AssociatedSource]
    raw_sources: list[dict[str, Any]] = field(default_factory=list, repr=False)
    pruning_reports: list[dict[str, Any]] = field(default_factory=list)
    speed_of_sound_m_s: float = 333.0
    microphone_position_uncertainty_m: float = 2.0

    def features_for_pair(self, first: int, second: int) -> list[PeakFeature]:
        pair = _ordered_pair(first, second)
        return [feature for feature in self.features if feature.pair == pair]


@dataclass
class SourceLocation:
    source_id: int
    position: NDArray[np.floating] | None
    uncertainty_m: float | None
    residual_samples: float | None
    boundary_contact: bool
    gps_position: NDArray[np.floating] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def position_m(self) -> NDArray[np.floating] | None:
        """Estimated position in BioLoc's internal local-metre system."""
        return self.position

    @property
    def latitude(self) -> float | None:
        return None if self.gps_position is None else float(self.gps_position[0])

    @property
    def longitude(self) -> float | None:
        return None if self.gps_position is None else float(self.gps_position[1])


@dataclass
class LocalizationResult:
    sources: list[SourceLocation]

    @property
    def positions(self) -> NDArray[np.floating]:
        values = [source.position for source in self.sources if source.position is not None]
        if not values:
            return np.empty((0, 2), dtype=float)
        return np.asarray(values, dtype=float)

    @property
    def gps_positions(self) -> NDArray[np.floating]:
        """Estimated source positions as ``[latitude, longitude]`` rows."""
        values = [
            source.gps_position
            for source in self.sources
            if source.gps_position is not None
        ]
        if not values:
            return np.empty((0, 2), dtype=float)
        return np.asarray(values, dtype=float)


@dataclass
class ReconstructedSource:
    source_id: int
    signal: NDArray[np.floating]
    reconstruction_mask: NDArray[np.floating]
    reference_microphone: int
    relative_delays_samples: NDArray[np.floating]


@dataclass
class ReconstructionResult:
    sources: list[ReconstructedSource]
    sample_rate: float
    mask_hop_samples: int | None = None
    mask_frequency_bands_hz: tuple[tuple[float, float], ...] = ()
    display_masks: list[NDArray[np.floating]] = field(
        default_factory=list, repr=False
    )

    def __iter__(self) -> Iterator[ReconstructedSource]:
        return iter(self.sources)


@dataclass
class SourceClassification:
    source_id: int
    label: str | None
    confidence: float | None
    detections: Any
    input_signal: NDArray[np.floating] | None = field(default=None, repr=False)


@dataclass
class ClassificationResult:
    sources: list[SourceClassification]
    classifier_name: str
    sample_rate: float


@dataclass
class PipelineResult:
    roi: ROIResult
    local_tdoas: LocalTDOAResult
    peaks: PeakExtractionResult
    association: AssociationResult
    localization: LocalizationResult
    reconstruction: ReconstructionResult
    classification: ClassificationResult
    timings: dict[str, float] = field(default_factory=dict)


def _ordered_pair(first: int, second: int) -> Pair:
    first, second = int(first), int(second)
    return (first, second) if first < second else (second, first)
