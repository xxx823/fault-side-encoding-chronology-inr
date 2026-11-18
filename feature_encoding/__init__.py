"""Feature encoding utilities for fault-conditioned geological modeling."""

from .chronology_mask import (
    TemporalRule,
    apply_chronology_mask,
    gate_domain_features,
)

__all__ = [
    "TemporalRule",
    "apply_chronology_mask",
    "gate_domain_features",
]
