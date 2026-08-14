"""One-shot composition of the six public processing stages."""

from __future__ import annotations

from time import perf_counter

from .association import associate_tdoas
from .config import PipelineConfig
from .local_tdoa import estimate_local_tdoas
from .localization import localize_sources
from .models import PipelineResult, Recording
from .peak_extraction import extract_tdoa_peaks
from .reconstruction import SourceClassifier, classify_reconstructed_sources, reconstruct_sources
from .roi import extract_regions_of_interest


def run_pipeline(
    recording: Recording,
    config: PipelineConfig | None = None,
    *,
    classifier: SourceClassifier | None = None,
    plot: bool = False,
    save_plots: bool = False,
) -> PipelineResult:
    """Run the full pipeline and retain every important intermediate result."""
    config = config or PipelineConfig()
    timings: dict[str, float] = {}

    roi = _timed(
        timings,
        "roi",
        extract_regions_of_interest,
        recording,
        config.roi,
        config.local_tdoa,
    )
    local_tdoas = _timed(
        timings,
        "local_tdoa",
        estimate_local_tdoas,
        recording,
        roi,
        config.local_tdoa,
        config.roi,
    )
    peaks = _timed(
        timings,
        "peak_extraction",
        extract_tdoa_peaks,
        local_tdoas,
        config.peaks,
    )
    association = _timed(
        timings,
        "association",
        associate_tdoas,
        peaks,
        local_tdoas,
        recording,
        config.association,
        config.local_tdoa,
    )
    localization = _timed(
        timings,
        "localization",
        localize_sources,
        association,
        recording,
        config.localization,
        config.local_tdoa,
    )
    reconstruction = _timed(
        timings,
        "reconstruction",
        reconstruct_sources,
        recording,
        peaks,
        local_tdoas,
        association,
        localization,
        config.reconstruction,
        config.local_tdoa,
    )
    classification = None
    if classifier is not None:
        classification = _timed(
            timings,
            "classification",
            classify_reconstructed_sources,
            reconstruction,
            classifier,
        )
    pipeline_result = PipelineResult(
        roi=roi,
        local_tdoas=local_tdoas,
        peaks=peaks,
        association=association,
        localization=localization,
        reconstruction=reconstruction,
        classification=classification,
        timings=timings,
    )
    if plot or save_plots:
        from .plotting.standard import make_standard_plots

        _timed(
            timings,
            "plotting",
            make_standard_plots,
            recording,
            pipeline_result,
            config,
            show=bool(plot),
            save=bool(save_plots),
        )
    return pipeline_result


def _timed(timings: dict[str, float], name: str, function, *args, **kwargs):
    started = perf_counter()
    result = function(*args, **kwargs)
    timings[name] = perf_counter() - started
    return result
