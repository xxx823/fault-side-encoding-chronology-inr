import math
import unittest

import torch

from ablation.r1b_core import (
    normalize_coordinates,
    physical_gradient_from_normalized,
    orientation_cosine_loss,
)


class GradientCorrectionTests(unittest.TestCase):
    def test_physical_gradient_matches_chain_rule_for_anisotropic_domain(self):
        ranges = torch.tensor([1964.0, 1560.0, 950.0])
        g_u = torch.tensor([[1.0, 2.0, 3.0]])
        expected = g_u * (2.0 / ranges)
        actual = physical_gradient_from_normalized(g_u, ranges)
        torch.testing.assert_close(actual, expected)

    def test_physical_gradient_matches_finite_difference(self):
        ranges = torch.tensor([1964.0, 1560.0, 950.0], dtype=torch.float64)
        center = torch.tensor([10.0, -20.0, 30.0], dtype=torch.float64)
        point = torch.tensor([400.0, 200.0, 500.0], dtype=torch.float64)
        coefficient = torch.tensor([1.5, -2.0, 0.5], dtype=torch.float64)
        normalized_gradient = coefficient.unsqueeze(0)
        analytical = physical_gradient_from_normalized(normalized_gradient, ranges)[0]
        step = 1e-2

        def scalar_at(value):
            normalized = 2.0 * (value - center) / ranges
            return torch.sum(coefficient * normalized)

        finite_difference = torch.stack(
            [
                (scalar_at(point + torch.eye(3, dtype=torch.float64)[axis] * step) - scalar_at(point - torch.eye(3, dtype=torch.float64)[axis] * step))
                / (2.0 * step)
                for axis in range(3)
            ]
        )
        torch.testing.assert_close(analytical, finite_difference, rtol=1e-8, atol=1e-10)


    def test_normalized_coordinates_use_per_axis_domain_ranges(self):
        x = torch.tensor([[982.0, 780.0, 475.0]])
        center = torch.tensor([0.0, 0.0, 0.0])
        ranges = torch.tensor([1964.0, 1560.0, 950.0])
        actual = normalize_coordinates(x, center, ranges)
        torch.testing.assert_close(actual, torch.ones_like(actual))

    def test_isotropic_scaling_preserves_gradient_direction(self):
        g_u = torch.tensor([[2.0, -1.0, 3.0]])
        ranges = torch.tensor([100.0, 100.0, 100.0])
        physical = physical_gradient_from_normalized(g_u, ranges)
        torch.testing.assert_close(physical / physical.norm(), g_u / g_u.norm())


    def test_orientation_loss_is_zero_after_physical_gradient_transform(self):
        ranges = torch.tensor([1964.0, 1560.0, 950.0])
        physical_normal = torch.tensor([[1.0, 1.0, 1.0]])
        physical_normal = physical_normal / torch.linalg.vector_norm(physical_normal, dim=1, keepdim=True)
        normalized_gradient = physical_normal * (ranges / 2.0)
        loss = orientation_cosine_loss(normalized_gradient, physical_normal, ranges)
        self.assertLess(float(loss), 1e-7)


    def test_orientation_loss_is_finite_for_zero_gradient(self):
        ranges = torch.tensor([1964.0, 1560.0, 950.0])
        loss = orientation_cosine_loss(torch.zeros((1, 3)), torch.tensor([[1.0, 0.0, 0.0]]), ranges)
        self.assertTrue(math.isfinite(float(loss)))
