"""Run the complete BioLoc pipeline from standardized recording folders."""

from dataclasses import replace
from pathlib import Path

from bioloc import PipelineConfig, load_recording, run_pipeline


# Edit this input block, then use the PyCharm Run or Debug button.
RECORDING_ROOT = Path("recordings")
POSITIONS_CSV = RECORDING_ROOT / "positions.csv"
RECORDER_IDS = ["mic-a", "mic-b", "mic-c"]
RECORDING_DATE = "20260102"  # Set to None when WAVs are directly inside ID folders.
RECORDING_TIME = "030405"    # Set to None when each selected folder has one WAV.
START_SECONDS = 0.0
DURATION_SECONDS = 5.0
ENVIRONMENT_MARGIN_M = 100.0

CLASSIFIER = "birdnet"       # "birdnet" or "perch"
MIN_CONFIDENCE = 0.1
PERCH_MODEL = "perch_8"
PLOT = True
SAVE_PLOTS = False


def main():
    recording = load_recording(
        root_directory=RECORDING_ROOT,
        positions_csv=POSITIONS_CSV,
        recorder_ids=RECORDER_IDS,
        recording_date=RECORDING_DATE,
        recording_time=RECORDING_TIME,
        start_seconds=START_SECONDS,
        duration_seconds=DURATION_SECONDS,
        environment_margin_m=ENVIRONMENT_MARGIN_M,
    )

    config = PipelineConfig()
    config = replace(
        config,

        # General
        general=replace(
            config.general,
            lowcut_hz=None,
            highcut_hz=10_000.0,
            speed_of_sound_m_s=333.0,
            microphone_position_uncertainty_m=2.0,
        ),

        # Region-of-interest extraction
        roi=replace(
            config.roi,
            bin_std=10.0,
            bin_per=0.5,
            min_roi=10,
        ),

        # Local TDOA estimation
        local_tdoa=replace(
            config.local_tdoa,
            band_width_hz=800.0,
            frame_size=2048,
            k_neighbors=3,
            filter_mode="fft_mask",
            workers=4,
            use_roi_mask=True,
        ),

        # Histogram and peak extraction
        peaks=replace(
            config.peaks,
            narrow_width_divisor=40.0,
            wide_to_narrow_ratio=3.0,
            histogram_sigma_divisor=20.0,
            support_tolerance_divisor=3.0,
        ),

        # TDOA association
        association=replace(
            config.association,
            feature="time_frequency_candidates",
            cosine_threshold=0.7,
            same_pair_cosine_threshold=0.7,
            frequency_weight=0.5,
            prune_geometric_outliers=True,
            prune_min_relative_improvement=0.3,
        ),

        # Source localization
        localization=replace(
            config.localization,
            loss="linear",
            multistart=True,
            grid_spacing_m=None,
            n_starts=5,
        ),

        # Source separation and reconstruction
        reconstruction=replace(
            config.reconstruction,
            reconstruction_frame_size=2048,
            reject_ambiguous_pair_tf=True,
            pair_ambiguity_tolerance_samples=None,
        ),

        # Source classification
        classification=replace(
            config.classification,
            backend="birdnet",
            min_confidence=0.1,
            perch_model="perch_8",
            perch_hop_duration=1.0,
            perch_warm_up=True,
        ),
    )
    result = run_pipeline(
        recording,
        config,
        plot=PLOT,
        save_plots=SAVE_PLOTS,
    )

    print(f"Localized and reconstructed {len(result.localization.sources)} sources")
    for location, classification in zip(
        result.localization.sources,
        result.classification.sources,
    ):
        print(
            f"source={location.source_id} "
            f"gps={location.gps_position} "
            f"label={classification.label!r} "
            f"confidence={classification.confidence}"
        )
    print("Stage timings (s):", result.timings)


if __name__ == "__main__":
    main()
