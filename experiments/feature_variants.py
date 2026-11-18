"""Build fair input variants for fault-feature ablation experiments.

The three data arrays follow the conventions used by the original project:

* domain: ``[X, Y, Z, fault_1, ..., fault_M]``
* interface: ``[label, X, Y, Z, fault_1, ..., fault_M]``
* orientation: ``[X, Y, Z, dx, dy, dz, fault_1, ..., fault_M]``
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


VALID_VARIANTS = ("coord", "coord_zero", "global_side", "effective_domain")


@dataclass(frozen=True)
class TemporalRule:
    """Deactivate an older fault feature outside the side retained by a younger fault."""

    older_fault: str
    younger_fault: str
    retained_younger_side: int

    def __post_init__(self) -> None:
        if self.retained_younger_side not in (0, 1):
            raise ValueError("retained_younger_side must be 0 or 1")
        if self.older_fault == self.younger_fault:
            raise ValueError("older_fault and younger_fault must be different")


def _fault_index(fault_names: Sequence[str], name: str) -> int:
    try:
        return list(fault_names).index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown fault {name!r}; available faults: {list(fault_names)}") from exc


def _apply_rules(
    features: np.ndarray,
    fault_names: Sequence[str],
    rules: Iterable[TemporalRule],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply ``m_j * h_j`` without modifying the caller's array."""

    raw = np.asarray(features)
    if raw.ndim != 2 or raw.shape[1] != len(fault_names):
        raise ValueError(
            f"Expected feature shape (N, {len(fault_names)}), got {raw.shape}"
        )

    result = raw.copy()
    metadata: Dict[str, float] = {}
    for rule in rules:
        older_idx = _fault_index(fault_names, rule.older_fault)
        younger_idx = _fault_index(fault_names, rule.younger_fault)

        # The mask is always inferred from the original side encoding. This is
        # important when several chronological rules are composed.
        mask = np.isclose(
            raw[:, younger_idx],
            float(rule.retained_younger_side),
            atol=1e-6,
        )
        result[:, older_idx] *= mask.astype(result.dtype, copy=False)
        metadata[
            f"{rule.older_fault}_active_fraction_by_{rule.younger_fault}"
        ] = float(mask.mean()) if mask.size else float("nan")

    return result, metadata


def transform_domain(
    domain: np.ndarray,
    variant: str,
    fault_names: Sequence[str],
    rules: Iterable[TemporalRule],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Transform one ``[xyz, fault features]`` array for model inference."""

    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VALID_VARIANTS}")

    domain = np.asarray(domain)
    expected = 3 + len(fault_names)
    if domain.ndim != 2 or domain.shape[1] != expected:
        raise ValueError(f"Expected domain shape (N, {expected}), got {domain.shape}")

    xyz = domain[:, :3].copy()
    raw_features = domain[:, 3:].copy()

    if variant == "coord":
        return xyz, {}
    if variant == "coord_zero":
        return np.column_stack((xyz, np.zeros_like(raw_features))), {}
    if variant == "global_side":
        return np.column_stack((xyz, raw_features)), {}

    effective, metadata = _apply_rules(raw_features, fault_names, rules)
    return np.column_stack((xyz, effective)), metadata


def build_variant(
    domain: np.ndarray,
    interface: np.ndarray,
    orientation: np.ndarray,
    variant: str,
    fault_names: Sequence[str],
    rules: Iterable[TemporalRule],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Return domain, interface and orientation arrays for one ablation group."""

    n_faults = len(fault_names)
    if interface.ndim != 2 or interface.shape[1] != 4 + n_faults:
        raise ValueError(
            f"Expected interface shape (N, {4 + n_faults}), got {interface.shape}"
        )
    if orientation.ndim != 2 or orientation.shape[1] != 6 + n_faults:
        raise ValueError(
            f"Expected orientation shape (N, {6 + n_faults}), got {orientation.shape}"
        )

    domain_out, domain_meta = transform_domain(domain, variant, fault_names, rules)

    interface_domain = np.column_stack((interface[:, 1:4], interface[:, 4:]))
    interface_x, interface_meta = transform_domain(
        interface_domain, variant, fault_names, rules
    )
    interface_out = np.column_stack((interface[:, 0], interface_x))

    orientation_domain = np.column_stack((orientation[:, :3], orientation[:, 6:]))
    orientation_x, orientation_meta = transform_domain(
        orientation_domain, variant, fault_names, rules
    )
    orientation_out = np.column_stack(
        (orientation_x[:, :3], orientation[:, 3:6], orientation_x[:, 3:])
    )

    metadata = {}
    for prefix, values in (
        ("domain", domain_meta),
        ("interface", interface_meta),
        ("orientation", orientation_meta),
    ):
        metadata.update({f"{prefix}_{key}": value for key, value in values.items()})

    return domain_out, interface_out, orientation_out, metadata

