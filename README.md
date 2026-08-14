# multisource-tdoa

Research-oriented Python package for extracting local time differences of
arrival (TDOAs), associating pairwise delays into concurrent acoustic sources,
reconstructing those sources, and estimating their locations.

The package is currently an alpha refactor of the original `main.py` research
pipeline. The numerical kernels remain unchanged while the public interface is
being stabilized.

## Installation

Clone the private repository, enter its directory, and create an isolated
environment:

```bash
git clone <private-repository-url>
cd <repository-directory>
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs the package from the current project directory (`.`), including
all processing, loading, localization, and plotting dependencies. The editable
flag (`-e`) is convenient while testing because changes to the source code take
effect without reinstalling the package.

For contributors who also need test, build, and lint tools:

```bash
python -m pip install -e ".[dev]"
```

## Input convention

```python
from multisource_tdoa import Recording

recording = Recording(
    signals=signals,                         # (microphones, samples)
    sample_rate=48_000,                     # Hz
    microphone_positions=positions_metres,  # (microphones, 2) or (..., 3)
    microphone_names=("AUD34", "AUD35", "AUD36"),
)
```

Signals must be synchronized. Microphone positions and localization outputs use
metres. Delays use samples unless their field name explicitly says seconds.
Local TDOA estimation uses the projected ROI mask by default, so only admitted
pair/band/frame cells contribute delay observations.

## Standard recording loader

The general loader accepts this directory structure:

```text
recordings/
├── mic-a/
│   └── 20260102/
│       └── device_20260102_030405_SYNC.WAV
├── mic-b/
│   └── 20260102/
│       └── device_20260102_030405_SYNC.WAV
└── positions.csv
```

The CSV can be headerless and compatible with the current project:

```text
49.83592,14.09566,mic-a
49.83616,14.09531,mic-b
```

or headered:

```text
recorder_id,latitude,longitude
mic-a,49.83592,14.09566
mic-b,49.83616,14.09531
```

Load and standardize the signals and geometry with:

```python
from multisource_tdoa import load_recording

recording = load_recording(
    root_directory="recordings",
    positions_csv="recordings/positions.csv",
    recorder_ids=["mic-a", "mic-b"],
    recording_date="20260102",
    recording_time="030405",
    start_seconds=10,
    duration_seconds=5,
)
```

`recorder_ids` selects microphones and defines their order. When
`recording_date=None`, files are read directly from each recorder-ID folder.
With no date or time selector, each selected folder must contain exactly one
eligible synchronized WAV file.

## Complete pipeline

```python
from multisource_tdoa import PipelineConfig, run_pipeline

config = PipelineConfig()
result = run_pipeline(
    recording,
    config,
    plot=True,
    save_plots=True,
)

result.roi.masks_by_pair
result.local_tdoas.for_pair(0, 1).delays
result.peaks.for_pair(0, 1).peaks
result.association.features
result.association.sources
result.localization.sources
result.reconstruction.sources[0].reconstruction_mask
result.reconstruction.sources[0].signal
```

No-source cases return empty lists rather than `None`.

## Stage-by-stage use

```python
from multisource_tdoa import (
    associate_tdoas,
    estimate_local_tdoas,
    extract_regions_of_interest,
    extract_tdoa_peaks,
    localize_sources,
    reconstruct_sources,
)

roi = extract_regions_of_interest(recording, config.roi, config.local_tdoa)
local = estimate_local_tdoas(recording, roi, config.local_tdoa, config.roi)
peaks = extract_tdoa_peaks(local, config.peaks)
association = associate_tdoas(
    peaks, local, recording, config.association, config.local_tdoa
)
locations = localize_sources(
    association, recording, config.localization, config.local_tdoa
)
reconstruction = reconstruct_sources(
    recording,
    peaks,
    local,
    association,
    locations,
    config.reconstruction,
    config.local_tdoa,
)
```

Feature construction is part of association. `association.features` exposes the
exact vectors used by that step.

## Optional classification

Classification is an additional operation on reconstructed signals. Ready-to-use
BirdNET and Perch adapters classify the most active window of every reconstructed
source while retaining all detections returned by the selected backend.

Install one backend and pass its classifier to the pipeline:

```bash
python -m pip install -e ".[birdnet]"
```

```python
from multisource_tdoa import BirdNETClassifier

result = run_pipeline(
    recording,
    config,
    classifier=BirdNETClassifier(min_confidence=0.1),
)

result.classification.sources[0].label
result.classification.sources[0].confidence
result.classification.sources[0].detections
```

For Perch, install `.[perch]` and use `PerchClassifier()`. Users can also supply
another classifier implementing `classify(signal, sample_rate)`:

```python
from multisource_tdoa import classify_reconstructed_sources

class MyClassifier:
    name = "my-classifier"

    def classify(self, signal, sample_rate):
        return {
            "label": "target",
            "confidence": 0.91,
            "detections": [],
        }

classifications = classify_reconstructed_sources(
    reconstruction,
    MyClassifier(),
)
```

Passing the same classifier to `run_pipeline(..., classifier=...)` stores the
result in `result.classification`.

## Plotting

The public pipeline exposes only two plot controls:

```python
result = run_pipeline(
    recording,
    config,
    plot=True,        # display the standard figures
    save_plots=True,  # save them under Figures/multisource_tdoa
)
```

The standard report uses the established plotting routines and fixed project
style. It contains the ROI overlay, local-TDOA colour map, associated pair
histograms, peak time-frequency support, final source-mask overlays,
reconstructed-source spectrograms, and the final hyperbola/localization plot.
Classifier-input plots and paper-specific debugging figures are not generated.

## Regression comparison

`multisource_tdoa.regression.save_pipeline_snapshot` saves compact intermediate
values to JSON. It includes ROI activity, local TDOA counts and samples, peak
delays, associated members, locations, mask summaries, reconstructed-signal
summaries, and classifications.

`examples/current_segment_new_api.py` retains the microphone IDs, date, time,
and segment boundaries used for the accepted regression segment. Set its
`DATA_PATH` to a local copy of the synchronized recordings before running it.
It writes `current_segment_new_api.json` for comparison.

## Project status

This is version `0.1.0` and the API should be considered provisional until the
legacy segment comparison is accepted. A public release also requires the
project owner to choose a software license and final package name.
