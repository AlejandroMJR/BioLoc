"""Optional, general-purpose classifiers for reconstructed sources."""

from __future__ import annotations

from fractions import Fraction
from time import perf_counter
from typing import Any

import numpy as np

from .models import ReconstructedSource


class BirdNETClassifier:
    """Classify reconstructed sources with BirdNET.

    BirdNET analyzes the most active three-second window of each source and
    returns every detection above ``min_confidence``.
    """

    name = "birdnet"

    def __init__(
        self,
        *,
        min_confidence: float = 0.1,
        analyzer: Any | None = None,
    ) -> None:
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if analyzer is None:
            try:
                from birdnetlib.analyzer import Analyzer
            except ImportError as error:
                raise ImportError(
                    "BirdNET classification requires "
                    "Reinstall `BioLoc` to restore its BirdNET dependency."
                ) from error
            analyzer = Analyzer()
        self.analyzer = analyzer
        self.min_confidence = float(min_confidence)
        self.mask_hop_samples = 1

    def classify(self, signal: np.ndarray, sample_rate: float) -> dict[str, Any]:
        return self._classify_window(np.asarray(signal), sample_rate, None)

    def classify_source(
        self,
        source: ReconstructedSource,
        sample_rate: float,
    ) -> dict[str, Any]:
        start, end, window = _best_mask_window(
            source.reconstruction_mask,
            sample_rate,
            source.signal.size,
            hop=self.mask_hop_samples,
            window_seconds=3.0,
        )
        signal = _crop_or_pad(source.signal, start, end)
        return self._classify_window(signal, sample_rate, window)

    def _classify_window(
        self,
        signal: np.ndarray,
        sample_rate: float,
        window: dict[str, Any] | None,
    ) -> dict[str, Any]:
        analysis_rate = 48_000
        analyzed = _resample(signal, sample_rate, analysis_rate)
        try:
            from birdnetlib import RecordingBuffer
        except ImportError as error:
            raise ImportError(
                "BirdNET classification requires "
                "Reinstall `BioLoc` to restore its BirdNET dependency."
            ) from error
        recording = RecordingBuffer(
            self.analyzer,
            analyzed,
            analysis_rate,
            min_conf=self.min_confidence,
        )
        recording.analyze()
        detections = list(recording.detections)
        label, confidence = _best_detection(detections)
        return {
            "label": label,
            "confidence": confidence,
            "detections": detections,
            "input_signal": analyzed,
            "analysis_sample_rate": analysis_rate,
            "window": window,
        }


class PerchClassifier:
    """Classify reconstructed sources with a Perch Hoplite model.

    All model heads and all classes above ``min_confidence`` are retained.
    """

    name = "perch"

    def __init__(
        self,
        model_name: str = "perch_8",
        *,
        min_confidence: float = 0.1,
        hop_duration: float = 1.0,
        model: Any | None = None,
        warm_up: bool = True,
    ) -> None:
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if float(hop_duration) <= 0:
            raise ValueError("hop_duration must be positive")
        if model is None:
            try:
                from perch_hoplite.zoo import model_configs
            except ImportError as error:
                raise ImportError(
                    "Perch classification requires "
                    "`pip install BioLoc[perch]`."
                ) from error
            model = model_configs.load_model_by_name(model_name)
        self.model = model
        self.model_name = str(model_name)
        self.min_confidence = float(min_confidence)
        self.hop_duration = float(hop_duration)
        self.mask_hop_samples = 1
        if warm_up and self.model_name.startswith("perch_v2"):
            self._warm_up()

    def _warm_up(self) -> None:
        sample_rate = int(self.model.sample_rate)
        count = int(round(sample_rate * float(self.model.window_size_s)))
        times = np.arange(count, dtype=np.float32) / sample_rate
        signal = 0.01 * np.sin(2.0 * np.pi * 1000.0 * times)
        started = perf_counter()
        self.model.embed(signal.astype(np.float32))
        self.warm_up_seconds = perf_counter() - started

    def classify(self, signal: np.ndarray, sample_rate: float) -> dict[str, Any]:
        return self._classify_window(np.asarray(signal), sample_rate, None)

    def classify_source(
        self,
        source: ReconstructedSource,
        sample_rate: float,
    ) -> dict[str, Any]:
        duration = float(self.model.window_size_s)
        start, end, window = _best_mask_window(
            source.reconstruction_mask,
            sample_rate,
            source.signal.size,
            hop=self.mask_hop_samples,
            window_seconds=duration,
        )
        signal = _crop_or_pad(source.signal, start, end)
        return self._classify_window(signal, sample_rate, window)

    def _classify_window(
        self,
        signal: np.ndarray,
        sample_rate: float,
        window: dict[str, Any] | None,
    ) -> dict[str, Any]:
        analysis_rate = int(self.model.sample_rate)
        analyzed = _resample(signal, sample_rate, analysis_rate)
        previous_hop = float(self.model.hop_size_s)
        self.model.hop_size_s = self.hop_duration
        try:
            outputs = self.model.embed(analyzed)
        finally:
            self.model.hop_size_s = previous_hop
        if outputs.logits is None:
            raise ValueError("Perch output does not contain classification logits")
        detections = self._detections_from_logits(outputs.logits)
        label, confidence = _best_detection(detections)
        return {
            "label": label,
            "confidence": confidence,
            "detections": detections,
            "input_signal": analyzed,
            "analysis_sample_rate": analysis_rate,
            "window": window,
        }

    def _detections_from_logits(self, logits_by_head: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for output_head, values in logits_by_head.items():
            logits = np.asarray(values)
            logits = logits.reshape(-1, logits.shape[-1])
            class_head = _match_perch_class_head(
                self.model.class_list,
                str(output_head),
                logits.shape[-1],
            )
            class_list = self.model.class_list[class_head]
            namespace = str(class_list.namespace)
            if (
                namespace.endswith("_fsd50k")
                and float(getattr(self.model, "logit_slope", 1.0)) == 1.0
                and float(getattr(self.model, "logit_intercept", 0.0)) == 0.0
            ):
                logits = 0.97 * logits - 10.0
            confidences = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
            for frame_index, frame in enumerate(confidences):
                indices = np.flatnonzero(frame >= self.min_confidence)
                indices = sorted(indices, key=lambda index: float(frame[index]), reverse=True)
                for class_index in indices:
                    label = str(class_list.classes[int(class_index)])
                    detections.append({
                        "classifier": "perch",
                        "class_name": label,
                        "label": label,
                        "label_namespace": namespace,
                        "confidence": float(frame[int(class_index)]),
                        "start_time": frame_index * self.hop_duration,
                        "end_time": (
                            frame_index * self.hop_duration
                            + float(self.model.window_size_s)
                        ),
                    })
        return detections


def _match_perch_class_head(class_lists: Any, output_head: str, size: int) -> str:
    if output_head in class_lists and len(class_lists[output_head].classes) == size:
        return output_head
    matches = [
        name for name, labels in class_lists.items()
        if len(labels.classes) == size
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot match Perch output head {output_head!r} ({size} classes) "
            f"to class lists {list(class_lists)}"
        )
    return matches[0]


def _best_detection(detections: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    if not detections:
        return None, None
    best = max(detections, key=lambda item: float(item.get("confidence", 0.0)))
    label = (
        best.get("common_name")
        or best.get("label")
        or best.get("scientific_name")
        or best.get("class_name")
    )
    return (str(label) if label is not None else None, float(best.get("confidence", 0.0)))


def _resample(signal: np.ndarray, source_rate: float, target_rate: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if int(source_rate) == int(target_rate):
        return values
    from scipy.signal import resample_poly

    ratio = Fraction(int(target_rate), int(round(source_rate))).limit_denominator()
    return np.asarray(
        resample_poly(values, ratio.numerator, ratio.denominator),
        dtype=np.float32,
    )


def _best_mask_window(
        source_mask: np.ndarray,
        sample_rate: float,
        signal_length: int,
        *,
        hop: int,
        window_seconds: float,
) -> tuple[int, int, dict[str, Any]]:
    mask = np.asarray(source_mask, dtype=np.float32)
    window_length = int(round(window_seconds * float(sample_rate)))
    signal_length = int(signal_length)
    if signal_length <= window_length:
        return 0, signal_length, {
            "start_seconds": 0.0,
            "end_seconds": signal_length / float(sample_rate),
            "activity": float(np.sum(mask)) if mask.size else 0.0,
        }
    if mask.ndim != 2 or mask.shape[1] == 0:
        return 0, window_length, {
            "start_seconds": 0.0,
            "end_seconds": window_seconds,
            "activity": 0.0,
        }
    activity = np.sum(np.clip(mask, 0.0, 1.0), axis=0)
    frames = max(1, int(np.ceil(window_length / max(1, int(hop)))))
    if activity.size <= frames:
        best_frame = 0
        score = float(np.sum(activity))
    else:
        scores = np.convolve(activity, np.ones(frames, dtype=np.float32), mode="valid")
        best_frame = int(np.argmax(scores))
        score = float(scores[best_frame])
    start = min(best_frame * max(1, int(hop)), signal_length - window_length)
    end = start + window_length
    return start, end, {
        "start_seconds": start / float(sample_rate),
        "end_seconds": end / float(sample_rate),
        "activity": score,
    }


def _crop_or_pad(signal: np.ndarray, start: int, end: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    target = max(0, int(end) - int(start))
    cropped = values[max(0, int(start)):min(values.size, int(end))]
    if cropped.size < target:
        cropped = np.pad(cropped, (0, target - cropped.size))
    return np.asarray(cropped, dtype=np.float32)


__all__ = ["BirdNETClassifier", "PerchClassifier"]
