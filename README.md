# BioLoc

Python package for extracting local time differences of
arrival (TDOAs), associating pairwise delays into concurrent acoustic sources,
reconstructing those sources, classifying them and estimating their locations.

## Installation

Clone the private repository, enter its directory, and create an isolated
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the same `.venv` before installation with either:

```powershell
.venv\Scripts\Activate.ps1
```

or, from Command Prompt:

```bat
.venv\Scripts\activate.bat
```

On Linux systems where Python is exposed as `python3`, use `python3` in the
environment-creation command. The environment directory is still named
`.venv` on every operating system.

This installs the package from the current project directory (`.`), including
all processing, loading, localization, classification, and plotting
dependencies.

## Loading recordings

BioLoc has one standard user-facing loading function: `load_recording`. Input
signals must already be synchronized. The loader reads the selected microphone
files, checks their compatibility, loads the requested segment, and converts
the microphone GPS coordinates to the local-metre geometry used internally.

The recording directory follows this structure:

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

The same loader accepts the following CSV representation with the microphone names and their positions.

```text
recorder_id,latitude,longitude
mic-a,49.83592,14.09566
mic-b,49.83616,14.09531
```

Load and standardize the signals and geometry with:

```python
from bioloc import load_recording

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

Geometry is computed in local metres. Every estimated source is returned both
as local metres (`position_m`) and GPS coordinates (`gps_position`, ordered as
latitude then longitude). Delays use samples unless their field name explicitly
says seconds.

## Complete pipeline

```python
from bioloc import PipelineConfig, run_pipeline

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
result.localization.gps_positions                 # latitude, longitude
result.localization.sources[0].latitude
result.localization.sources[0].longitude
result.reconstruction.sources[0].reconstruction_mask
result.reconstruction.sources[0].signal
```

No-source cases return empty lists rather than `None`.

## Examples

`examples/run_full_pipeline.py` is the recommended complete example. Edit its
input block and run it directly or with the PyCharm Run/Debug buttons:

```bash
python examples/run_full_pipeline.py
```

`examples/step_by_step.ipynb` runs the same standardized input through each
pipeline stage separately. It exposes the intermediate result objects and
plots ROI masks, local TDOAs, peak histograms and time-frequency support,
association features, localization, reconstruction masks, and reconstructed
signals. JupyterLab is included in the standard BioLoc installation:

```bash
jupyter lab examples/step_by_step.ipynb
```

## Stage-by-stage use

```python
from bioloc import (
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

## Classification

Classification is the final pipeline stage. BirdNET is the default and is
installed with the package. It classifies the most active window of every
reconstructed source and retains every returned detection:

```python
result = run_pipeline(recording, config)

result.classification.sources[0].label
result.classification.sources[0].confidence
result.classification.sources[0].detections
```

To use Perch instead, install its model runtime and select it in the pipeline
configuration:

```bash
python -m pip install -e ".[perch]"
```

```python
from dataclasses import replace

config = replace(
    config,
    classification=replace(config.classification, backend="perch"),
)
result = run_pipeline(recording, config)
```

The standalone classification function remains available for users supplying a
different classifier implementing `classify(signal, sample_rate)`:

```python
from bioloc import classify_reconstructed_sources

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

## Plotting

The public pipeline exposes only two plot controls:

```python
result = run_pipeline(
    recording,
    config,
    plot=True,        # display the standard figures
    save_plots=True,  # save them under Figures/BioLoc
)
```

The standard report uses the established plotting routines and fixed project
style. It contains the ROI overlay, local-TDOA colour map, associated pair
histograms, peak time-frequency support, final source-mask overlays,
reconstructed-source spectrograms, and the final hyperbola/localization plot.
Classifier-input plots and paper-specific debugging figures are not generated.
