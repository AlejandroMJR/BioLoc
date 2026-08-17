import unittest

import numpy as np

from bioloc._kernels.association import (
    associate_tdoa_peaks_by_timefreq_candidates_geometric,
)


class InternalKernelImportTest(unittest.TestCase):
    def test_geometric_association_uses_renamed_localization_kernel(self):
        result = associate_tdoa_peaks_by_timefreq_candidates_geometric(
            {},
            n_bands=1,
            n_frames=1,
            mic_positions=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            fs=8_000,
            c=343.0,
            env_bounds=(np.zeros(2), np.ones(2)),
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
