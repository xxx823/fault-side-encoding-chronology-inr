"""Chronology-conditioned gating for binary fault-side features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TemporalRule:
    """Deactivate an older fault feature outside a retained side of a younger fault."""

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
        raise ValueError(
            f"Unknown fault {name!r}; available faults: {list(fault_names)}"
        ) from exc


def apply_chronology_mask(
    side_features: np.ndarray,
    fault_names: Sequence[str],
    rules: Iterable[TemporalRule],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply chronology rules to a two-dimensional fault-feature array.

    Each mask is derived from the original side features so rule composition is
    independent of rule order. The function returns a copy and does not mutate
    the caller's input.
    """

    raw = np.asarray(side_features)
    if raw.ndim != 2 or raw.shape[1] != len(fault_names):
        raise ValueError(
            f"Expected feature shape (N, {len(fault_names)}), got {raw.shape}"
        )

    result = raw.copy()
    metadata: Dict[str, float] = {}
    for rule in rules:
        older_idx = _fault_index(fault_names, rule.older_fault)
        younger_idx = _fault_index(fault_names, rule.younger_fault)
        retained = np.isclose(
            raw[:, younger_idx],
            float(rule.retained_younger_side),
            atol=1e-6,
        )
        result[:, older_idx] = raw[:, older_idx] * retained.astype(
            result.dtype, copy=False
        )
        metadata[
            f"{rule.older_fault}_active_fraction_by_{rule.younger_fault}"
        ] = float(retained.mean()) if retained.size else float("nan")

    return result, metadata


def gate_domain_features(
    domain: np.ndarray,
    fault_names: Sequence[str],
    rules: Iterable[TemporalRule],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply chronology masks to rows arranged as xyz followed by fault features."""

    values = np.asarray(domain)
    expected_width = 3 + len(fault_names)
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(
            f"Expected domain shape (N, {expected_width}), got {values.shape}"
        )

    gated, metadata = apply_chronology_mask(
        values[:, 3:], fault_names=fault_names, rules=rules
    )
    return np.column_stack((values[:, :3], gated)), metadata
