import unittest
from dataclasses import fields

import numpy as np

from bioloc import (
    AssociationResult,
    LocalTDOAResult,
    PairLocalTDOAs,
    Recording,
    config_help,
    classify_reconstructed_sources,
    extract_tdoa_peaks,
)
from bioloc.config import PeakExtractionConfig
from bioloc.config import (
    AssociationConfig,
    ClassificationConfig,
    GeneralConfig,
    LocalizationConfig,
    LocalTDOAConfig,
    ROIConfig,
    ReconstructionConfig,
)
from bioloc.localization import localize_sources
from bioloc.models import (
    ReconstructedSource,
    ReconstructionResult,
)


class PublicAPITest(unittest.TestCase):
    def test_config_help_lists_defaults_and_supports_sections(self):
        all_help = config_help()
        local_help = config_help("local_tdoa")
        self.assertIn("general", all_help)
        self.assertIn("classification", all_help)
        self.assertIn("band_width_hz = 800.0", local_help)
        self.assertNotIn("speed_of_sound_m_s", local_help)
        with self.assertRaises(ValueError):
            config_help("missing")

    def test_configuration_exposes_only_supported_user_controls(self):
        expected = {
            GeneralConfig: {
                "lowcut_hz", "highcut_hz", "speed_of_sound_m_s",
                "microphone_position_uncertainty_m",
            },
            ROIConfig: {"bin_std", "bin_per", "min_roi"},
            LocalTDOAConfig: {
                "band_width_hz", "frame_size", "k_neighbors", "filter_mode",
                "workers", "use_roi_mask",
            },
            PeakExtractionConfig: {
                "narrow_width_divisor", "wide_to_narrow_ratio",
                "histogram_sigma_divisor", "support_tolerance_divisor",
            },
            AssociationConfig: {
                "feature", "cosine_threshold", "same_pair_cosine_threshold",
                "frequency_weight", "prune_geometric_outliers",
                "prune_min_relative_improvement",
            },
            LocalizationConfig: {
                "loss", "multistart", "grid_spacing_m", "n_starts",
            },
            ReconstructionConfig: {
                "reconstruction_frame_size", "reject_ambiguous_pair_tf",
                "pair_ambiguity_tolerance_samples",
            },
            ClassificationConfig: {
                "backend", "min_confidence", "perch_model",
                "perch_hop_duration", "perch_warm_up",
            },
        }
        for config_type, names in expected.items():
            self.assertEqual({field.name for field in fields(config_type)}, names)

    def test_roi_mask_is_enabled_by_default(self):
        self.assertTrue(LocalTDOAConfig().use_roi_mask)

    def test_recording_validates_and_normalizes_arrays(self):
        recording = Recording(
            signals=[[0, 1, 2], [3, 4, 5]],
            sample_rate=8_000,
            microphone_positions=[[0, 0], [2, 0]],
            microphone_names=("left", "right"),
        )
        self.assertEqual(recording.signals.shape, (2, 3))
        self.assertEqual(recording.n_microphones, 2)
        self.assertEqual(recording.n_samples, 3)
        self.assertTrue(np.issubdtype(recording.signals.dtype, np.floating))

    def test_peak_result_retains_histogram_and_support(self):
        delays = np.concatenate(
            [np.full(30, -3.0), np.full(40, 5.0)]
        ).astype(np.float32)
        local = LocalTDOAResult(
            pairs={
                (0, 1): PairLocalTDOAs(
                    pair=(0, 1),
                    delays=delays,
                    band_indices=np.arange(delays.size, dtype=np.int16) % 2,
                    frame_indices=np.arange(delays.size, dtype=np.int32) % 10,
                    max_delay_samples=10,
                )
            },
            frequency_bands_hz=((1000.0, 1800.0), (1400.0, 2200.0)),
            frame_times_seconds=np.arange(10) * 0.01,
            frame_size=256,
            hop_size=128,
            sample_rate=8_000,
            algorithm_frequency_bins=(np.arange(3), np.arange(2, 5)),
        )
        result = extract_tdoa_peaks(local, PeakExtractionConfig())
        pair = result.for_pair(0, 1)
        self.assertEqual(pair.histogram.shape, pair.delay_bins.shape)
        self.assertGreater(len(pair.peaks), 0)
        self.assertEqual(pair.peaks[0].time_frequency_support.shape, (2, 10))
        self.assertGreater(pair.peaks[0].supporting_observation_indices.size, 0)

    def test_no_source_localization_returns_empty_result(self):
        recording = Recording(
            np.zeros((2, 128)),
            8_000,
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        )
        association = AssociationResult(features=[], sources=[], raw_sources=[])
        result = localize_sources(association, recording)
        self.assertEqual(result.sources, [])
        self.assertEqual(result.positions.shape, (0, 2))
        self.assertEqual(result.gps_positions.shape, (0, 2))

    def test_reconstructed_sources_can_be_classified(self):
        reconstruction = ReconstructionResult(
            sources=[
                ReconstructedSource(
                    source_id=0,
                    signal=np.ones(100),
                    reconstruction_mask=np.ones((4, 5)),
                    reference_microphone=1,
                    relative_delays_samples=np.asarray([1.0, 0.0]),
                )
            ],
            sample_rate=8_000,
        )

        class Classifier:
            name = "test-classifier"

            def classify(self, signal, sample_rate):
                return {"label": "bird", "confidence": 0.75, "detections": [1]}

        result = classify_reconstructed_sources(reconstruction, Classifier())
        self.assertEqual(result.classifier_name, "test-classifier")
        self.assertEqual(result.sources[0].label, "bird")
        self.assertEqual(result.sources[0].confidence, 0.75)


if __name__ == "__main__":
    unittest.main()
