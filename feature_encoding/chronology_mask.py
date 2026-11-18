"""Chronology-conditioned gating for binary fault-side features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TemporalRule:
    """Gate an older fault feature by one or more retained younger-fault sides.

    ``younger_fault`` and ``retained_younger_side`` preserve the original
    single-controller API. ``control_sides`` adds any number of additional
    ``(fault_name, retained_binary_side)`` conditions. All conditions are
    combined by logical AND, as in the product of indicators in Equation (5).
    """

    older_fault: str
    younger_fault: str | None = None
    retained_younger_side: int | None = None
    control_sides: Tuple[Tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if (self.younger_fault is None) != (self.retained_younger_side is None):
            raise ValueError(
                "younger_fault and retained_younger_side must be supplied together"
            )
        extra_controls = tuple(self.control_sides)
        controls = list(extra_controls)
        if self.younger_fault is not None:
            controls.insert(0, (self.younger_fault, self.retained_younger_side))
        if not controls:
            raise ValueError("A temporal rule needs at least one controlling fault")
        names = [name for name, _ in controls]
        if self.older_fault in names:
            raise ValueError("An older fault cannot control its own valid-domain mask")
        if len(names) != len(set(names)):
            raise ValueError("A controlling fault may appear only once per rule")
        if any(side not in (0, 1) for _, side in controls):
            raise ValueError("Retained binary sides must be 0 or 1")
        object.__setattr__(self, "control_sides", extra_controls)

    @property
    def conditions(self) -> Tuple[Tuple[str, int], ...]:
        """Return all controlling-fault side conditions for this older fault."""
        if self.younger_fault is None:
            return self.control_sides
        return ((self.younger_fault, self.retained_younger_side), *self.control_sides)


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
    masks_by_older: Dict[int, np.ndarray] = {}
    for rule in rules:
        older_idx = _fault_index(fault_names, rule.older_fault)
        rule_mask = np.ones(raw.shape[0], dtype=bool)
        for controller, retained_side in rule.conditions:
            younger_idx = _fault_index(fault_names, controller)
            retained = np.isclose(
                raw[:, younger_idx], float(retained_side), atol=1e-6
            )
            rule_mask &= retained
            metadata[
                f"{rule.older_fault}_active_fraction_by_{controller}"
            ] = float(retained.mean()) if retained.size else float("nan")
        if older_idx in masks_by_older:
            masks_by_older[older_idx] &= rule_mask
        else:
            masks_by_older[older_idx] = rule_mask

    for older_idx, combined_mask in masks_by_older.items():
        result[:, older_idx] = raw[:, older_idx] * combined_mask.astype(
            result.dtype, copy=False
        )
        metadata[
            f"{fault_names[older_idx]}_active_fraction_by_all_controls"
        ] = float(combined_mask.mean()) if combined_mask.size else float("nan")

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
