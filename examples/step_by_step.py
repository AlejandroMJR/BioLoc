"""Minimal stage-by-stage API example using arrays supplied by the user."""

from multisource_tdoa import (
    PipelineConfig,
    Recording,
    associate_tdoas,
    estimate_local_tdoas,
    extract_regions_of_interest,
    extract_tdoa_peaks,
    localize_sources,
    reconstruct_sources,
)


def analyze(signals, sample_rate, microphone_positions):
    recording = Recording(signals, sample_rate, microphone_positions)
    config = PipelineConfig()
    roi = extract_regions_of_interest(recording, config.roi, config.local_tdoa)
    local = estimate_local_tdoas(recording, roi, config.local_tdoa, config.roi)
    peaks = extract_tdoa_peaks(local, config.peaks)
    association = associate_tdoas(
        peaks, local, recording, config.association, config.local_tdoa
    )
    locations = localize_sources(
        association, recording, config.localization, config.local_tdoa
    )
    return reconstruct_sources(
        recording,
        peaks,
        local,
        association,
        locations,
        config.reconstruction,
        config.local_tdoa,
    )
