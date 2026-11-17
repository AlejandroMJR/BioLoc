import inspect
import unittest
from unittest.mock import patch

import numpy as np

from bioloc import reconstruction
from bioloc._kernels import tdoa


class EventNoiseFloorTest(unittest.TestCase):
    def setUp(self):
        self.signal = np.random.default_rng(7).standard_normal(8192).astype(np.float32)
        self.mask = np.zeros((1, 65), dtype=np.float32)

    def fill(self, mask=None, **kwargs):
        return tdoa.fill_outside_source_mask_with_white_noise(
            [self.signal], [self.mask if mask is None else mask],
            8000, 256, 128, 256,
            Omega_list=[np.arange(129)], mask_hop=128,
            mask_nperseg=256, mask_nfft=256,
            random_state=0, return_diagnostics=True, **kwargs,
        )

    def test_event_floor_is_default(self):
        self.assertEqual(reconstruction._WHITE_NOISE_LEVEL_MODE, 'event_noise_floor')
        parameter = inspect.signature(tdoa.fill_outside_source_mask_with_white_noise).parameters
        self.assertEqual(parameter['white_noise_level_mode'].default, 'event_noise_floor')

    def test_full_event_estimation_excludes_padded_frames_and_ignores_source_mask(self):
        weights = []
        estimate = tdoa._estimate_inside_mask_noise_db

        def capture(stft, weight, **kwargs):
            weights.append(weight.copy())
            return estimate(stft, weight, **kwargs)

        with patch.object(tdoa, '_estimate_inside_mask_noise_db', side_effect=capture):
            outputs, diagnostics = self.fill(
                fallback_source_masks=[np.ones_like(self.mask)],
                fallback_source_mask_allowed=[True],
            )
            self.fill(mask=np.ones_like(self.mask))
        np.testing.assert_array_equal(weights[0], weights[1])
        self.assertTrue(np.all(weights[0][:, 0] == 0))
        self.assertTrue(np.all(weights[0][:, -1] == 0))
        self.assertTrue(np.all(weights[0][:, 1:-1] > 0))
        self.assertFalse(diagnostics[0]['mask_fallback_used'])
        self.assertEqual(outputs[0].shape, self.signal.shape)
        self.assertTrue(np.all(np.isfinite(outputs[0])))
        # Stationary unit-power noise should be replaced at a comparable level.
        self.assertGreater(np.std(outputs[0]), 0.5)
        self.assertLess(np.std(outputs[0]), 1.5)

    def test_target_snr_does_not_override_event_floor(self):
        first, _ = self.fill(target_snr_db=-20)
        second, _ = self.fill(target_snr_db=40)
        np.testing.assert_array_equal(first[0], second[0])

    def test_too_short_event_reports_missing_complete_frames(self):
        self.signal = self.signal[:192]
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, 'too few analyzed bins'):
            self.fill()
