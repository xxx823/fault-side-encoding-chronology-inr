import unittest

import numpy as np

from ablation.r1b_experiment import pad_variant_features, relative_reduction, sample_eikonal_coordinates


class ExperimentProtocolTests(unittest.TestCase):
    def test_eikonal_points_are_deterministic_and_inside_domain(self):
        extent = (-1800.0, 164.0, -160.0, 1400.0, 500.0, 1450.0)
        a = sample_eikonal_coordinates(7, extent, 32)
        b = sample_eikonal_coordinates(7, extent, 32)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(a[:, 0] >= extent[0]))
        self.assertTrue(np.all(a[:, 0] <= extent[1]))
        self.assertTrue(np.all(a[:, 1] >= extent[2]))
        self.assertTrue(np.all(a[:, 1] <= extent[3]))
        self.assertTrue(np.all(a[:, 2] >= extent[4]))
        self.assertTrue(np.all(a[:, 2] <= extent[5]))

    def test_relative_reduction_uses_baseline_as_denominator(self):
        self.assertAlmostEqual(relative_reduction(10.0, 8.0), 0.2)
        self.assertAlmostEqual(relative_reduction(10.0, 12.0), -0.2)

    def test_variant_padding_equalizes_model_input_width(self):
        domain = np.zeros((3, 3))
        interface = np.zeros((3, 4))
        orientation = np.zeros((3, 6))
        self.assertEqual(pad_variant_features(domain, "domain").shape[1], 5)
        self.assertEqual(pad_variant_features(interface, "interface").shape[1], 6)
        self.assertEqual(pad_variant_features(orientation, "orientation").shape[1], 8)
