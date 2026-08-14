import unittest

import numpy as np

from bioloc._kernels.tdoa import (
    fill_outside_source_mask_with_white_noise,
)


class MaskNoiseFallbackTest(unittest.TestCase):
    def setUp(self):
        self.fs = 8000
        self.nperseg = 256
        self.noverlap = 128
        self.nfft = 256
        self.signal = np.random.default_rng(7).standard_normal(8192).astype(
            np.float32
        )
        self.n_frames = 65
        self.rejected_mask = np.zeros((1, self.n_frames), dtype=np.float32)
        self.original_mask = np.zeros_like(self.rejected_mask)
        self.original_mask[:, 4:24] = 1.0
        self.omega_list = [np.arange(self.nfft // 2 + 1)]

    def run_fill(self, fallback_masks):
        return fill_outside_source_mask_with_white_noise(
            [self.signal],
            [self.rejected_mask],
            self.fs,
            self.nperseg,
            self.noverlap,
            self.nfft,
            Omega_list=self.omega_list,
            mask_hop=self.nperseg - self.noverlap,
            mask_nperseg=self.nperseg,
            mask_nfft=self.nfft,
            replacement_mode="white_noise",
            white_noise_level_mode="inside_noise_floor",
            fallback_source_masks=fallback_masks,
            fallback_source_mask_allowed=[True],
            return_diagnostics=True,
            random_state=0,
        )

    def test_original_mask_is_used_when_rejected_mask_is_too_sparse(self):
        outputs, diagnostics = self.run_fill([self.original_mask])

        self.assertEqual(outputs[0].shape, self.signal.shape)
        self.assertTrue(diagnostics[0]["mask_fallback_used"])
        self.assertTrue(
            diagnostics[0]["mask_contamination_possible"]
        )
        self.assertEqual(
            diagnostics[0]["mask_fallback_reason"],
            "ambiguous_pair_rejection_left_too_few_core_bins",
        )

    def test_missing_fallback_still_reports_unrecoverable_sparse_mask(self):
        with self.assertRaisesRegex(ValueError, "too few core bins"):
            self.run_fill(None)


if __name__ == "__main__":
    unittest.main()
