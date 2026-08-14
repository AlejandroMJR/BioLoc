import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from multisource_tdoa import BirdNETClassifier, PerchClassifier
from multisource_tdoa.models import ReconstructedSource


def _source(sample_rate=100, seconds=2):
    return ReconstructedSource(
        source_id=0,
        signal=np.ones(sample_rate * seconds, dtype=np.float32),
        reconstruction_mask=np.ones((4, 20), dtype=np.float32),
        reference_microphone=0,
        relative_delays_samples=np.zeros(2),
    )


class ClassifierAdapterTest(unittest.TestCase):
    def test_birdnet_adapter_retains_all_detections(self):
        class RecordingBuffer:
            def __init__(self, analyzer, data, rate, min_conf):
                self.detections = []

            def analyze(self):
                self.detections = [
                    {"common_name": "Species A", "confidence": 0.4},
                    {"common_name": "Species B", "confidence": 0.9},
                ]

        fake_module = types.SimpleNamespace(RecordingBuffer=RecordingBuffer)
        with patch.dict(sys.modules, {"birdnetlib": fake_module}):
            classifier = BirdNETClassifier(analyzer=object())
            classifier.mask_hop_samples = 10
            result = classifier.classify_source(_source(), 100)

        self.assertEqual(result["label"], "Species B")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(len(result["detections"]), 2)

    def test_perch_adapter_is_not_species_specific(self):
        class Labels:
            namespace = "test"
            classes = ("Species A", "Species B")

        class Output:
            logits = {"head": np.array([[0.0, 2.0]])}

        class Model:
            sample_rate = 100
            window_size_s = 1.0
            hop_size_s = 1.0
            class_list = {"head": Labels()}

            @staticmethod
            def embed(signal):
                return Output()

        classifier = PerchClassifier(model=Model(), warm_up=False)
        classifier.mask_hop_samples = 10
        result = classifier.classify_source(_source(), 100)

        self.assertEqual(result["label"], "Species B")
        self.assertGreater(result["confidence"], 0.8)
        self.assertEqual(
            {item["label"] for item in result["detections"]},
            {"Species A", "Species B"},
        )


if __name__ == "__main__":
    unittest.main()
