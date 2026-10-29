import unittest
from unittest.mock import patch

import numpy as np

from bioloc import LocalTDOAConfig, Recording, estimate_local_tdoas


class LocalTDOAPairsTest(unittest.TestCase):
    def setUp(self):
        self.recording = Recording(
            signals=np.random.default_rng(0).normal(size=(3, 4096)),
            sample_rate=24000,
            microphone_positions=[[0, 0], [1, 0], [3, 0]],
        )
        self.config = LocalTDOAConfig(frame_size=256, workers=1)

    def test_explicit_pair_limits_computation_and_normalizes_duplicates(self):
        with patch(
            "bioloc.local_tdoa.extract_subband_tdoas_windowed_search",
            return_value=([np.array([2.0])], [np.array([0])], [np.array([0])]),
        ) as kernel:
            result = estimate_local_tdoas(
                self.recording, config=self.config, pairs=[(2, 0), (0, 2)],
            )
        self.assertEqual(list(result.pairs), [(0, 2)])
        np.testing.assert_array_equal(kernel.call_args.kwargs["iu"], [0])
        np.testing.assert_array_equal(kernel.call_args.kwargs["ju"], [2])
        np.testing.assert_array_equal(result.for_pair(2, 0).delays, [2.0])

    def test_default_and_explicit_pair_run_real_estimator(self):
        default = estimate_local_tdoas(self.recording, config=self.config)
        selected = estimate_local_tdoas(
            self.recording, config=self.config, pairs=[(0, 2)],
        )
        self.assertEqual(set(default.pairs), {(0, 1), (0, 2), (1, 2)})
        self.assertEqual(list(selected.pairs), [(0, 2)])
        np.testing.assert_array_equal(
            default.for_pair(0, 2).delays, selected.for_pair(0, 2).delays,
        )

    def test_invalid_selections(self):
        for pairs in ([], [(0, 0)], [(-1, 1)], [(0, 3)], [(0.5, 1)], [(0,)], [(0, 1, 2)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                estimate_local_tdoas(self.recording, config=self.config, pairs=pairs)
