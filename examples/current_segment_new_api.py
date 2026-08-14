"""Run the retained regression segment through the public API."""

from dataclasses import replace
from pathlib import Path

from multisource_tdoa import (
    BirdNETClassifier,
    PerchClassifier,
    PipelineConfig,
    load_recording,
    run_pipeline,
)
from multisource_tdoa.regression import save_pipeline_snapshot


# Point this at the recording root before running the example.
DATA_PATH = Path("recordings")
POSITIONS_CSV = DATA_PATH / "REAL_GPS_Audiomoth.csv"
RECORDER_IDS = ["AUD34", "AUD35", "AUD36", "AUD37", "AUD38"]
RECORDING_DATE = "20260308"
RECORDING_TIME = "053501"
START_SECONDS = 459
DURATION_SECONDS = 5

# PyCharm-friendly run settings. Edit these and press Run or Debug.
CLASSIFIER = None  # None, "birdnet", or "perch"
MIN_CONFIDENCE = 0.1
PERCH_MODEL = "perch_8"
PLOT = True
SAVE_PLOTS = True


def main():
    recording = load_recording(
        root_directory=DATA_PATH,
        positions_csv=POSITIONS_CSV,
        recorder_ids=RECORDER_IDS,
        recording_date=RECORDING_DATE,
        recording_time=RECORDING_TIME,
        start_seconds=START_SECONDS,
        duration_seconds=DURATION_SECONDS,
    )
    environment_size = float(recording.metadata["map_info"]["environment_size"])
    bounds = ((0.0, 0.0), (environment_size, environment_size))

    config = PipelineConfig()
    config = replace(
        config,
        association=replace(config.association, environment_bounds=bounds),
        localization=replace(config.localization, environment_bounds=bounds),
    )
    if CLASSIFIER == "birdnet":
        classifier = BirdNETClassifier(min_confidence=MIN_CONFIDENCE)
    elif CLASSIFIER == "perch":
        classifier = PerchClassifier(
            PERCH_MODEL,
            min_confidence=MIN_CONFIDENCE,
        )
    elif CLASSIFIER is None:
        classifier = None
    else:
        raise ValueError("CLASSIFIER must be None, 'birdnet', or 'perch'")
    result = run_pipeline(
        recording,
        config,
        classifier=classifier,
        plot=PLOT,
        save_plots=SAVE_PLOTS,
    )
    path = save_pipeline_snapshot(result, "current_segment_new_api.json")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
