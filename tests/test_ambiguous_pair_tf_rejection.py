import unittest

import numpy as np

from bioloc._kernels.reconstruction import build_source_tf_maps_aligned_to_mic


class AmbiguousPairTfRejectionTest(unittest.TestCase):
    def setUp(self):
        self.mic_positions = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
            ]
        )
        # Both sources lie on the perpendicular bisector of pair (0, 2), so
        # that pair predicts tau=0 for both. Pair (0, 1) distinguishes them.
        self.source_positions = [
            np.array([2.0, 5.0]),
            np.array([8.0, 5.0]),
        ]
        self.sources = [
            {
                "members": [
                    {"pair": (0, 2), "peak_idx": 0, "tau": 0.0},
                    {"pair": (0, 1), "peak_idx": 0, "tau": -12.0},
                ]
            },
            {
                "members": [
                    {"pair": (0, 2), "peak_idx": 0, "tau": 0.0},
                ]
            },
        ]
        self.pairs = {
            (0, 2): {
                "peaks": [{"tf_cells": np.array([[0, 0]])}],
                "support_tolerance_samples": 1.0,
            },
            (0, 1): {
                "peaks": [{"tf_cells": np.array([[1, 1]])}],
                "support_tolerance_samples": 1.0,
            },
        }

    def build(self, reject):
        return build_source_tf_maps_aligned_to_mic(
            sources=self.sources,
            tdoas_hat_pairs=self.pairs,
            n_bands=2,
            n_frames=2,
            n_mics=3,
            hop=1000,
            mic_positions=self.mic_positions,
            source_positions=self.source_positions,
            fs=1000,
            c=343.0,
            reject_ambiguous_pair_tf=reject,
        )

    def test_hard_gate_rejects_only_ambiguous_pair_tf(self):
        maps, _, debug = self.build(reject=True)

        self.assertEqual(float(np.sum(maps[0])), 1.0)
        self.assertEqual(float(maps[0][1, 1]), 1.0)
        self.assertEqual(float(maps[0][0, 0]), 0.0)
        self.assertEqual(float(np.sum(maps[1])), 0.0)
        self.assertTrue(
            any(
                member["reason"]
                == "ambiguous_predicted_tdoa_tf_rejected"
                for member in debug[0]["members"]
            )
        )

    def test_disabled_gate_recovers_previous_union_behavior(self):
        maps, _, _ = self.build(reject=False)

        self.assertEqual(float(np.sum(maps[0])), 2.0)
        self.assertEqual(float(np.sum(maps[1])), 1.0)


if __name__ == "__main__":
    unittest.main()
