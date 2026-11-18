"""Repository-local inputs and integrity checks for the R1B reference case."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


R1B_ASSET_SHA256 = {
    "data/case1/surface.csv": "BF96D0D8139683CE8ADF6D43F259957459EBA258E04689497085B503C2ECE337",
    "data/case1/orientation.csv": "C7204DA2BE30626ED61EEC0A958072F72056A8A911BD903BD9998FA978CC3987",
    "data/case1/fault_meshes/fault1_raw.vtp": "B934827188A8BABBEE9F45064CD4DFD115DFDF62F3E1771283326DA380BFDB57",
    "data/case1/fault_meshes/fault2_raw.vtp": "1DA1223AC440B4FA96C77F0A67BA4B3A51FD4F6EE87FE5E2956B870758FE6026",
}


def bundle_paths(repository_root: Path) -> dict[str, object]:
    """Return input paths rooted at a checkout, without machine-specific paths."""

    root = Path(repository_root)
    return {
        "surface": root / "data" / "case1" / "surface.csv",
        "orientation": root / "data" / "case1" / "orientation.csv",
        "fault_meshes": {
            "fault1": root / "data" / "case1" / "fault_meshes" / "fault1_raw.vtp",
            "fault2": root / "data" / "case1" / "fault_meshes" / "fault2_raw.vtp",
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_asset_hashes(
    repository_root: Path,
    expected_hashes: Mapping[str, str] = R1B_ASSET_SHA256,
) -> dict[str, str]:
    """Fail closed when a required R1B input is absent or has changed."""

    root = Path(repository_root)
    actual_hashes: dict[str, str] = {}
    for relative_path, expected in expected_hashes.items():
        asset = root / Path(relative_path)
        if not asset.is_file():
            raise FileNotFoundError(f"Required R1B input is missing: {relative_path}")
        actual = _sha256(asset)
        if actual.upper() != expected.upper():
            raise ValueError(
                f"SHA-256 mismatch for {relative_path}: expected {expected}, got {actual}"
            )
        actual_hashes[relative_path] = actual
    return actual_hashes


def require_training_opt_in(
    *, run_formal: bool, preflight_only: bool, finalize_only: bool
) -> bool:
    """Keep the default invocation read-only; training requires an explicit flag."""

    if preflight_only or finalize_only:
        return False
    if not run_formal:
        raise ValueError("Pass --run-formal to start the 20-cell neural experiment")
    return True
