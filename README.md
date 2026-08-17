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

`config.general.environment_margin_m` controls the searchable area surrounding
the microphone array. BioLoc passes the same bounds to association,
localization, and plotting automatically.

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
result.classification.sources[0].label
result.classification.sources[0].confidence
result.classification.sources[0].detections
```

To see what every returned variable means, ask the result itself:

```python
print(result.help())
print(result.help("association"))
print(result.help("localization"))
```

The same output reference is available without running the pipeline:

```python
from bioloc import output_help

print(output_help())
print(output_help("association"))
```

Or from a terminal with `bioloc --output-help` and, for example,
`bioloc --output-help association`.

No-source cases return empty lists rather than `None`.

### Configuring the complete pipeline

Every supported processing parameter remains configurable when using
`run_pipeline`. Update the relevant immutable configuration section with
`dataclasses.replace`:

```python
from dataclasses import replace
from bioloc import PipelineConfig, run_pipeline

config = PipelineConfig()
config = replace(
    config,
    general=replace(
        config.general,
        highcut_hz=8_000,
    ),
    local_tdoa=replace(
        config.local_tdoa,
        band_width_hz=600,
    ),
    reconstruction=replace(
        config.reconstruction,
        reconstruction_frame_size=4096,
    ),
)
result = run_pipeline(recording, config)
```

Frequency-band overlap is fixed internally to half of `band_width_hz`.
`grid_spacing_m=None` derives the localization grid from the recording
environment. `pair_ambiguity_tolerance_samples=None` derives the ambiguity
tolerance from each pair's peak-support tolerance.

Shared physical and frequency-range settings live in `config.general`:

```python
config.general.lowcut_hz
config.general.highcut_hz
config.general.speed_of_sound_m_s
config.general.microphone_position_uncertainty_m
```

BioLoc filters the synchronized signals once during ROI extraction. The same
filtered signals are reused for local TDOA estimation and source
reconstruction, matching the original pipeline without repeating the filter.

To see every available parameter, its default and a short explanation:

```python
from bioloc import config_help

print(config_help())
print(config_help("local_tdoa"))
```

The same help is available from a terminal:

```bash
bioloc --config-help
bioloc --config-help local_tdoa
```

Printing `PipelineConfig()` also gives a compact view of all current values.
BirdNET is the default classifier. To use Perch, install its runtime and select
the backend:

```bash
python -m pip install -e ".[perch]"
```

```python
config = replace(
    config,
    classification=replace(config.classification, backend="perch"),
)
```

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

roi = extract_regions_of_interest(recording, config.roi, config.general)
local = estimate_local_tdoas(recording, roi, config.local_tdoa)
peaks = extract_tdoa_peaks(local, config.peaks)
association = associate_tdoas(
    peaks, local, recording, config.association
)
locations = localize_sources(
    association, recording, config.localization
)
reconstruction = reconstruct_sources(
    recording,
    peaks,
    local,
    association,
    locations,
    config.reconstruction,
)
```

Feature construction is part of association. `association.features` exposes the
exact vectors used by that step.

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
