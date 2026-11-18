import hashlib
import tempfile
import unittest
from pathlib import Path

from ablation.r1b_bundle import (
    bundle_paths,
    require_training_opt_in,
    validate_asset_hashes,
)


class ValidateAssetHashesTests(unittest.TestCase):
    def test_bundle_paths_are_repository_relative(self):
        paths = bundle_paths(Path("repository"))

        self.assertEqual(
            paths["fault_meshes"]["fault1"],
            Path("repository/data/case1/fault_meshes/fault1_raw.vtp"),
        )

    def test_validates_repository_relative_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "data" / "case1" / "surface.csv"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"x,y,z\n1,2,3\n")
            expected = hashlib.sha256(asset.read_bytes()).hexdigest().upper()

            actual = validate_asset_hashes(root, {"data/case1/surface.csv": expected})

            self.assertEqual(actual, {"data/case1/surface.csv": expected})

    def test_rejects_a_changed_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "surface.csv"
            asset.write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_asset_hashes(root, {"surface.csv": "0" * 64})

    def test_training_requires_an_explicit_flag(self):
        with self.assertRaisesRegex(ValueError, "--run-formal"):
            require_training_opt_in(run_formal=False, preflight_only=False, finalize_only=False)

    def test_preflight_never_requests_training(self):
        self.assertFalse(
            require_training_opt_in(run_formal=False, preflight_only=True, finalize_only=False)
        )


if __name__ == "__main__":
    unittest.main()
