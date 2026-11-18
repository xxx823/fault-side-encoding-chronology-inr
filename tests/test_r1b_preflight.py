import unittest

from ablation.r1b_experiment import preflight_inputs


class R1BPreflightTests(unittest.TestCase):
    def test_preflight_checks_the_frozen_bundle_without_training(self):
        report = preflight_inputs()

        self.assertEqual(report["status"], "R1B_PREFLIGHT_PASS")
        self.assertFalse(report["training_performed"])
        self.assertEqual(report["formal_training_units"], 20)
        self.assertEqual(report["surface_rows"], 184)
        self.assertEqual(report["orientation_rows"], 98)


if __name__ == "__main__":
    unittest.main()
