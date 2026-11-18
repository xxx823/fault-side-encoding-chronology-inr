"""Independent R1B utilities.

This module is deliberately separate from the original D: drive project.  It
contains only the corrected coordinate-gradient transform and deterministic
binary-mask utilities used by the R1B controlled experiment.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np
import torch


def normalize_coordinates(
    coordinates: torch.Tensor,
    center: torch.Tensor,
    ranges: torch.Tensor,
) -> torch.Tensor:
    """Map physical coordinates to ``[-1, 1]`` axis by axis."""

    coordinates = torch.as_tensor(coordinates)
    center = torch.as_tensor(center, dtype=coordinates.dtype, device=coordinates.device)
    ranges = torch.as_tensor(ranges, dtype=coordinates.dtype, device=coordinates.device)
    if torch.any(ranges <= 0):
        raise ValueError("all coordinate ranges must be positive")
    return 2.0 * (coordinates - center) / ranges


def physical_gradient_from_normalized(
    gradient_normalized: torch.Tensor,
    ranges: torch.Tensor,
) -> torch.Tensor:
    """Apply ``g_x = J^T g_u`` for axis-wise coordinate normalization."""

    gradient_normalized = torch.as_tensor(gradient_normalized)
    ranges = torch.as_tensor(
        ranges,
        dtype=gradient_normalized.dtype,
        device=gradient_normalized.device,
    )
    if gradient_normalized.shape[-1] != 3 or ranges.shape != (3,):
        raise ValueError("gradient must end in 3 coordinates and ranges must have shape (3,)")
    if torch.any(ranges <= 0):
        raise ValueError("all coordinate ranges must be positive")
    jacobian = torch.diag(2.0 / ranges)
    return gradient_normalized @ jacobian.T


def orientation_cosine_loss(
    gradient_normalized: torch.Tensor,
    physical_normals: torch.Tensor,
    ranges: torch.Tensor,
    *,
    sign_invariant: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compare a normalized-coordinate gradient with physical normals.

    ``sign_invariant=False`` is the registered signed-normal convention.  The
    optional sign-invariant mode is explicit and is not used in R1B.
    """

    gradient_physical = physical_gradient_from_normalized(gradient_normalized, ranges)
    normal = torch.as_tensor(
        physical_normals,
        dtype=gradient_physical.dtype,
        device=gradient_physical.device,
    )
    if gradient_physical.shape != normal.shape:
        raise ValueError("gradient and physical normals must have the same shape")
    grad_norm = torch.linalg.vector_norm(gradient_physical, dim=-1, keepdim=True)
    normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    grad_unit = gradient_physical / grad_norm.clamp_min(eps)
    normal_unit = normal / normal_norm.clamp_min(eps)
    cosine = torch.sum(grad_unit * normal_unit, dim=-1)
    if sign_invariant:
        cosine = cosine.abs()
    return torch.mean(1.0 - cosine)


def enumerate_binary_states(fault_names: Sequence[str]) -> np.ndarray:
    """Return every binary state in deterministic lexicographic order."""

    return np.asarray(list(product((0, 1), repeat=len(fault_names))), dtype=np.int8)


def apply_hierarchical_masks(
    raw_features: np.ndarray,
    fault_names: Sequence[str],
    rules: Iterable[tuple[str, str, int, int]],
) -> np.ndarray:
    """Apply the registered ``m_i h_i`` binary gate without mutating input."""

    raw = np.asarray(raw_features, dtype=np.int8)
    if raw.ndim != 2 or raw.shape[1] != len(fault_names):
        raise ValueError("raw_features shape does not match fault_names")
    indices = {name: i for i, name in enumerate(fault_names)}
    if len(indices) != len(fault_names):
        raise ValueError("fault_names must be unique")
    result = raw.copy()
    masks = np.ones_like(result, dtype=np.int8)
    for child, parent, child_value, parent_value in rules:
        if child not in indices or parent not in indices:
            raise ValueError("mask rule references an unknown fault")
        child_index = indices[child]
        parent_index = indices[parent]
        keep = ~(
            (raw[:, child_index] == int(child_value))
            & (raw[:, parent_index] == int(parent_value))
        )
        masks[:, child_index] &= keep.astype(np.int8)
        masks[:, child_index] &= masks[:, parent_index]
    result *= masks
    return result


def eikonal_loss_normalized(
    gradient_normalized: torch.Tensor,
) -> torch.Tensor:
    """The registered normalized-coordinate Eikonal regularizer."""

    norm = torch.linalg.vector_norm(gradient_normalized, dim=-1)
    return torch.mean((norm - 1.0) ** 2)
