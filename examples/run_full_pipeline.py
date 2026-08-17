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
        classification=replace(
            config.classification,
            backend=CLASSIFIER,
            min_confidence=MIN_CONFIDENCE,
            perch_model=PERCH_MODEL,
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
