"""Build fair input variants for fault-feature ablation experiments.

The three data arrays follow the conventions used by the original project:

* domain: ``[X, Y, Z, fault_1, ..., fault_M]``
* interface: ``[label, X, Y, Z, fault_1, ..., fault_M]``
* orientation: ``[X, Y, Z, dx, dy, dz, fault_1, ..., fault_M]``
"""

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from feature_encoding.chronology_mask import TemporalRule, apply_chronology_mask


VALID_VARIANTS = ("coord", "coord_zero", "global_side", "effective_domain")


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

    effective, metadata = apply_chronology_mask(raw_features, fault_names, rules)
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

    rules = tuple(rules)
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

