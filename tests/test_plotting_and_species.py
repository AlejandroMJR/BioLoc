import inspect
import sys
from unittest.mock import patch, MagicMock
import unittest
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bioloc import PerchClassifier, PipelineConfig
from bioloc.classifiers import _perch_detection_names
from bioloc._defaults import SPEED_OF_SOUND_M_S
from bioloc.models import ROIResult, LocalTDOAResult, AssociationResult
# Map dependencies are unrelated to these support plots; no tiles are requested.
with patch.dict(sys.modules, {
    'contextily': MagicMock(), 'xyzservices': MagicMock(),
    'xyzservices.providers': MagicMock(), 'pyproj': MagicMock(),
}):
    from bioloc._kernels import plotting, localization, reconstruction
    from bioloc.plotting import plot_association_stage


class PlottingAndSpeciesTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_unassociated_support_plot(self):
        raw = {'tau': 2., 'bands': np.array([0]), 'frames': np.array([0]),
               'tf_cells': np.array([[0, 0]])}
        peaks = SimpleNamespace(pairs={(0, 1): SimpleNamespace(
            peaks=[SimpleNamespace(raw=raw, delay_samples=2.)],
            histogram=np.ones(3), delay_bins=np.arange(3),
            assignment_tolerance_samples=1.)})
        local = SimpleNamespace(frequency_bands_hz=((0, 800),),
                                frame_times_seconds=np.array([0., 1.]),
                                hop_size=128, sample_rate=24000)
        figures = plot_association_stage(peaks, local, None, PipelineConfig(), show=False)
        self.assertEqual(len(figures), 1)
        self.assertIn('unassigned', figures[0][0].axes[0].get_title(loc='left'))
        association = SimpleNamespace(sources=[object()], raw_sources=[{
            'members': [{'pair': (0, 1), 'peak_idx': 0, 'tau': 2.}],
        }])
        associated = plot_association_stage(
            peaks, local, association, PipelineConfig(), show=False,
        )
        self.assertIn('final selected', associated[0][0].axes[0].get_title(loc='left'))

    def test_sound_speed_defaults_agree(self):
        self.assertEqual(PipelineConfig().general.speed_of_sound_m_s, SPEED_OF_SOUND_M_S)
        for cls in (ROIResult, LocalTDOAResult, AssociationResult):
            self.assertEqual(cls.__dataclass_fields__['speed_of_sound_m_s'].default,
                             SPEED_OF_SOUND_M_S)
        for module in (plotting, localization, reconstruction):
            for _, function in inspect.getmembers(module, inspect.isfunction):
                parameter = inspect.signature(function).parameters.get('c')
                if parameter and parameter.default is not inspect.Parameter.empty:
                    self.assertEqual(parameter.default, SPEED_OF_SOUND_M_S)

    def test_perch_names_and_species_headline(self):
        codes = ('gretit1', 'dunnoc1', 'comcha', 'trepip', 'yellow2', 'sonthr1')
        model = SimpleNamespace(
            sample_rate=100, window_size_s=5., hop_size_s=1.,
            class_list={
                'species': SimpleNamespace(namespace='ebird2021', classes=codes),
                'order': SimpleNamespace(namespace='ebird2021_orders', classes=('gaviiformes',)),
            },
            embed=lambda signal: SimpleNamespace(logits={
                'species': np.array([[1., 2., 3., 4., 5., 6.]]),
                'order': np.array([[10.]]),
            }),
        )
        output = PerchClassifier(model=model, warm_up=False).classify(np.zeros(500), 100)
        by_code = {d['label']: d for d in output['detections']}
        for code, common, scientific in (
            ('gretit1', 'Great Tit', 'Parus major'),
            ('dunnoc1', 'Dunnock', 'Prunella modularis'),
            ('comcha', 'Common Chaffinch', 'Fringilla coelebs'),
            ('trepip', 'Tree Pipit', 'Anthus trivialis'),
            ('yellow2', 'Yellowhammer', 'Emberiza citrinella'),
            ('sonthr1', 'Song Thrush', 'Turdus philomelos'),
        ):
            self.assertEqual(by_code[code]['common_name'], common)
            self.assertEqual(by_code[code]['scientific_name'], scientific)
        self.assertEqual(output['label'], 'Song Thrush')
        self.assertNotIn('common_name', by_code['gaviiformes'])
        self.assertEqual(_perch_detection_names('Parus major', 'inat2024_fsd50k')['common_name'], 'Great Tit')
        self.assertEqual(_perch_detection_names('unknown', 'ebird2021'), {})
