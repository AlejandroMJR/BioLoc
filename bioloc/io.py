"""Standardized loading of synchronized microphone-array recordings."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import Recording


def load_recording(
    root_directory: str | Path,
    positions_csv: str | Path,
    recorder_ids: Iterable[str],
    *,
    recording_date: str | int | None = None,
    recording_time: str | int | None = None,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    require_sync: bool = True,
) -> Recording:
    """Load a standardized synchronized microphone-array recording.

    Directory layout::

        root_directory/
            recorder_id/
                optional_YYYYMMDD/
                    [prefix_]YYYYMMDD_HHMMSS[_SYNC].wav

    If ``recording_date`` is omitted, WAV files are read directly from each
    recorder-ID folder. If no date/time selector is supplied, each selected
    folder must contain exactly one eligible WAV file.

    The positions CSV may be headerless ``latitude,longitude,id`` (compatible
    with ``REAL_GPS_Audiomoth.csv``) or contain named columns such as
    ``recorder_id,latitude,longitude``. ``recorder_ids`` selects microphones
    and defines their order in the returned signals and position arrays.
    """
    root = Path(root_directory).expanduser()
    csv_path = Path(positions_csv).expanduser()
    requested_ids = tuple(str(value).strip() for value in recorder_ids)
    if not requested_ids or any(not value for value in requested_ids):
        raise ValueError("recorder_ids must contain at least one non-empty ID")
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("recorder_ids contains duplicates")
    if not root.is_dir():
        raise FileNotFoundError(f"Recording root does not exist: {root}")

    gps_by_id = read_position_csv(csv_path)
    missing_positions = [value for value in requested_ids if value not in gps_by_id]
    if missing_positions:
        raise ValueError(
            f"Recorder IDs missing from {csv_path}: {missing_positions}"
        )

    date = _normalize_date(recording_date)
    time = _normalize_time(recording_time)
    paths = [
        _select_recording_path(
            root,
            recorder_id,
            date=date,
            time=time,
            require_sync=require_sync,
        )
        for recorder_id in requested_ids
    ]
    signals, sample_rate = _load_audio_files(
        paths,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )

    gps_positions = np.asarray([gps_by_id[value] for value in requested_ids])
    microphone_positions, map_info = gps_to_local_plot_coordinates(gps_positions)
    return Recording(
        signals=signals,
        sample_rate=sample_rate,
        microphone_positions=microphone_positions,
        microphone_names=requested_ids,
        metadata={
            "recording_paths": tuple(paths),
            "recording_date": date,
            "recording_time": time,
            "gps_positions": gps_positions,
            "gps_groups_for_map": gps_positions,
            "local_groups_for_map": microphone_positions,
            "map_info": map_info,
        },
    )


def read_position_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read recorder GPS positions from a headered or headerless CSV."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Position CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row and any(x.strip() for x in row)]
    if not rows:
        raise ValueError(f"Position CSV is empty: {path}")

    header = [value.strip().lower() for value in rows[0]]
    aliases = {
        "id": {"id", "recorder_id", "recorder", "microphone_id", "mic_id", "name"},
        "latitude": {"latitude", "lat"},
        "longitude": {"longitude", "lon", "lng", "long"},
    }
    indices = {
        name: next((header.index(value) for value in choices if value in header), None)
        for name, choices in aliases.items()
    }
    has_header = all(value is not None for value in indices.values())
    if has_header:
        data_rows = rows[1:]
        latitude_index = int(indices["latitude"])
        longitude_index = int(indices["longitude"])
        id_index = int(indices["id"])
    else:
        data_rows = rows
        latitude_index, longitude_index, id_index = 0, 1, 2

    output: dict[str, np.ndarray] = {}
    for row_number, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) <= max(latitude_index, longitude_index, id_index):
            raise ValueError(f"Incomplete position row {row_number} in {path}: {row}")
        recorder_id = row[id_index].strip()
        if not recorder_id:
            raise ValueError(f"Empty recorder ID on row {row_number} in {path}")
        if recorder_id in output:
            raise ValueError(f"Duplicate recorder ID {recorder_id!r} in {path}")
        try:
            latitude = float(row[latitude_index])
            longitude = float(row[longitude_index])
        except ValueError as error:
            raise ValueError(
                f"Invalid GPS coordinates on row {row_number} in {path}: {row}"
            ) from error
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(f"Latitude outside [-90, 90] for {recorder_id!r}")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError(f"Longitude outside [-180, 180] for {recorder_id!r}")
        output[recorder_id] = np.asarray([latitude, longitude], dtype=float)
    if not output:
        raise ValueError(f"Position CSV contains no recorder positions: {path}")
    return output


def gps_to_local_plot_coordinates(
    gps_positions: np.ndarray,
    *,
    context_margin_m: float = 100.0,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Convert latitude/longitude to a shared square local-metre domain."""
    gps_positions = np.asarray(gps_positions, dtype=float)
    if gps_positions.ndim != 2 or gps_positions.shape[1] != 2:
        raise ValueError("gps_positions must have shape (recorders, 2)")
    reference_latitude = float(np.mean(gps_positions[:, 0]))
    reference_longitude = float(np.mean(gps_positions[:, 1]))
    earth_radius_m = 6_371_000.0
    latitudes = np.deg2rad(gps_positions[:, 0])
    longitudes = np.deg2rad(gps_positions[:, 1])
    reference_latitude_rad = np.deg2rad(reference_latitude)
    reference_longitude_rad = np.deg2rad(reference_longitude)
    east = (
        (longitudes - reference_longitude_rad)
        * np.cos(reference_latitude_rad)
        * earth_radius_m
    )
    north = (latitudes - reference_latitude_rad) * earth_radius_m
    local = np.column_stack([east, north])

    local_min = np.min(local, axis=0)
    local_max = np.max(local, axis=0)
    center = 0.5 * (local_min + local_max)
    side = float(max(np.max(local_max - local_min) + 2.0 * context_margin_m, 1.0))
    plot_origin_local = center - side / 2.0
    centered = local - plot_origin_local
    map_info = {
        "reference_latitude": reference_latitude,
        "reference_longitude": reference_longitude,
        "earth_radius_m": earth_radius_m,
        "plot_origin_local": plot_origin_local,
        "plot_min": np.asarray([0.0, 0.0]),
        "plot_max": np.asarray([side, side]),
        "environment_size": side,
    }
    return centered, map_info


def local_to_gps_coordinates(
    local_positions: np.ndarray,
    map_info: dict,
) -> np.ndarray:
    """Convert BioLoc's internal metre coordinates to latitude/longitude."""
    positions = np.asarray(local_positions, dtype=float)
    single_position = positions.ndim == 1
    positions = np.atleast_2d(positions)
    if positions.shape[1] != 2:
        raise ValueError("local_positions must have shape (positions, 2)")
    required = (
        "reference_latitude",
        "reference_longitude",
        "earth_radius_m",
        "plot_origin_local",
    )
    missing = [key for key in required if key not in map_info]
    if missing:
        raise ValueError(f"map_info is missing GPS transform values: {missing}")
    uncentered = positions + np.asarray(map_info["plot_origin_local"], dtype=float)
    reference_latitude = float(map_info["reference_latitude"])
    reference_longitude = float(map_info["reference_longitude"])
    earth_radius_m = float(map_info["earth_radius_m"])
    latitude = reference_latitude + np.rad2deg(uncentered[:, 1] / earth_radius_m)
    longitude = reference_longitude + np.rad2deg(
        uncentered[:, 0]
        / (earth_radius_m * np.cos(np.deg2rad(reference_latitude)))
    )
    gps = np.column_stack([latitude, longitude])
    return gps[0] if single_position else gps


def _select_recording_path(
    root: Path,
    recorder_id: str,
    *,
    date: str | None,
    time: str | None,
    require_sync: bool,
) -> Path:
    folder = root / recorder_id
    if date is not None:
        folder = folder / date
    if not folder.is_dir():
        raise FileNotFoundError(f"Recording folder does not exist for {recorder_id}: {folder}")
    candidates = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".wav"
    )
    if require_sync:
        candidates = [path for path in candidates if path.stem.upper().endswith("_SYNC")]
    if date is not None and time is not None:
        timestamp = f"{date}_{time}"
        candidates = [path for path in candidates if timestamp in path.stem]
    elif date is not None:
        candidates = [path for path in candidates if date in path.stem]
    elif time is not None:
        token = re.compile(rf"(?:^|_)(?:\d{{8}}_)?{re.escape(time)}(?:_|$)")
        candidates = [path for path in candidates if token.search(path.stem)]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one matching WAV for recorder {recorder_id!r} in "
            f"{folder}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _load_audio_files(
    paths: list[Path],
    *,
    start_seconds: float,
    duration_seconds: float | None,
) -> tuple[np.ndarray, float]:
    try:
        import soundfile as sf
    except ImportError as error:
        raise ImportError("Recording loading requires the `soundfile` package") from error
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    loaded = []
    sample_rate = None
    for path in paths:
        info = sf.info(path)
        if info.channels != 1:
            raise ValueError(f"Expected a mono WAV for each recorder, got {info.channels}: {path}")
        if sample_rate is None:
            sample_rate = int(info.samplerate)
        elif int(info.samplerate) != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch: expected {sample_rate} Hz, got "
                f"{info.samplerate} Hz in {path}"
            )
        start_frame = int(round(start_seconds * sample_rate))
        frame_count = (
            -1
            if duration_seconds is None
            else int(round(duration_seconds * sample_rate))
        )
        if start_frame >= info.frames:
            raise ValueError(f"start_seconds lies beyond the end of {path}")
        values, actual_rate = sf.read(
            path,
            start=start_frame,
            frames=frame_count,
            dtype="float32",
            always_2d=False,
        )
        if int(actual_rate) != sample_rate:
            raise RuntimeError(f"Unexpected sample rate while reading {path}")
        if duration_seconds is not None and values.size != frame_count:
            raise ValueError(
                f"Requested {duration_seconds}s from {path}, but only "
                f"{values.size / sample_rate:.3f}s was available"
            )
        loaded.append(np.asarray(values, dtype=np.float32))
    shortest = min(values.size for values in loaded)
    return np.stack([values[:shortest] for values in loaded], axis=0), float(sample_rate)


def _normalize_date(value: str | int | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^0-9]", "", str(value))
    if len(normalized) != 8:
        raise ValueError("recording_date must use YYYYMMDD")
    return normalized


def _normalize_time(value: str | int | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^0-9]", "", str(value)).zfill(6)
    if len(normalized) != 6:
        raise ValueError("recording_time must use HHMMSS")
    hours, minutes, seconds = int(normalized[:2]), int(normalized[2:4]), int(normalized[4:])
    if hours > 23 or minutes > 59 or seconds > 59:
        raise ValueError(f"Invalid recording_time: {value!r}")
    return normalized
