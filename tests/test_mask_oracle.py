import itertools
import unittest

import numpy as np

from ablation.r1b_core import apply_hierarchical_masks, enumerate_binary_states


RULES = (("F1", "F2", 1, 1), ("F4", "F2", 1, 1), ("F3", "F4", 1, 1))
FAULTS = ("F1", "F2", "F3", "F4")


class MaskOracleTests(unittest.TestCase):
    def test_declared_f1_f4_f3_graph_has_no_cycle(self):
        states = enumerate_binary_states(FAULTS)
        result = apply_hierarchical_masks(states, FAULTS, RULES)
        self.assertEqual(result.shape, states.shape)


    def test_rule_order_is_invariant_for_declared_binary_rules(self):
        states = enumerate_binary_states(FAULTS)
        baseline = apply_hierarchical_masks(states, FAULTS, RULES)
        for order in itertools.permutations(RULES):
            actual = apply_hierarchical_masks(states, FAULTS, order)
            np.testing.assert_array_equal(actual, baseline)


    def test_effective_encoding_has_known_state_collision_without_claiming_failure(self):
        states = enumerate_binary_states(FAULTS)
        effective = apply_hierarchical_masks(states, FAULTS, RULES)
        self.assertEqual(len(np.unique(effective, axis=0)), 8)
        self.assertTrue(any(np.sum(np.all(effective == row, axis=1)) > 1 for row in np.unique(effective, axis=0)))
