"""TDOA-based multisource separation and localization."""

from .association import associate_tdoas, build_feature_vectors
from .classifiers import BirdNETClassifier, PerchClassifier
from .config import (
    AssociationConfig,
    ClassificationConfig,
    LocalizationConfig,
    LocalTDOAConfig,
    PeakExtractionConfig,
    PipelineConfig,
    ROIConfig,
    ReconstructionConfig,
)
from .local_tdoa import estimate_local_tdoas
from .io import load_recording, read_position_csv
from .localization import localize_sources
from .models import (
    AssociatedSource,
    AssociatedTDOA,
    AssociationResult,
    ClassificationResult,
    LocalizationResult,
    LocalTDOAResult,
    PairLocalTDOAs,
    PairPeakResult,
    PeakExtractionResult,
    PeakFeature,
    PipelineResult,
    ROIResult,
    Recording,
    ReconstructedSource,
    ReconstructionResult,
    SourceClassification,
    SourceLocation,
    TDOAPeak,
)
from .peak_extraction import extract_tdoa_peaks
from .pipeline import run_pipeline
from .reconstruction import (
    classify_reconstructed_sources,
    reconstruct_sources,
)
from .roi import extract_regions_of_interest

__all__ = [
    "AssociationConfig",
    "BirdNETClassifier",
    "AssociatedSource",
    "AssociatedTDOA",
    "AssociationResult",
    "ClassificationResult",
    "ClassificationConfig",
    "LocalizationConfig",
    "LocalizationResult",
    "LocalTDOAConfig",
    "LocalTDOAResult",
    "PairLocalTDOAs",
    "PairPeakResult",
    "PeakExtractionConfig",
    "PeakExtractionResult",
    "PeakFeature",
    "PipelineConfig",
    "PipelineResult",
    "PerchClassifier",
    "ROIConfig",
    "ROIResult",
    "Recording",
    "ReconstructedSource",
    "ReconstructionConfig",
    "ReconstructionResult",
    "SourceClassification",
    "SourceLocation",
    "TDOAPeak",
    "associate_tdoas",
    "build_feature_vectors",
    "classify_reconstructed_sources",
    "estimate_local_tdoas",
    "extract_regions_of_interest",
    "extract_tdoa_peaks",
    "localize_sources",
    "load_recording",
    "read_position_csv",
    "reconstruct_sources",
    "run_pipeline",
]
