"""Run the reproducible ablation study on the truncated two-fault case.

The script deliberately keeps fault geometry fixed and retrains only the
stratigraphic network for each feature variant and random seed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lossf  # noqa: E402
import models  # noqa: E402
import utils  # noqa: E402
from feature_ecoding.clip_surface import Fault_feature  # noqa: E402
from mlps import ConcatMLP  # noqa: E402

from ablation.feature_variants import (  # noqa: E402
    VALID_VARIANTS,
    TemporalRule,
    build_variant,
    transform_domain,
)


@dataclass(frozen=True)
class ExperimentConfig:
    surface_csv: str = "data/case1/surface.csv"
    orientation_csv: str = "data/case1/orientation.csv"
    extent: Tuple[float, ...] = (-1800, 164, -160, 1400, 500, 1450)
    fault_resolution: Tuple[int, ...] = (37, 27, 27)
    grid_resolution: Tuple[int, ...] = (140, 96, 96)
    fault_names: Tuple[str, ...] = ("fault1", "fault2")
    movement: Tuple[str, ...] = ("up", "down")
    fault_direction: Tuple[str, ...] = ("left", "right")
    older_fault: str = "fault1"
    younger_fault: str = "fault2"
    fault_hidden_dim: int = 256
    fault_hidden_layers: int = 2
    fault_beta: Tuple[float, ...] = (10.0, 10.0)
    fault_epochs: int = 1000
    fault_lr: float = 0.001
    strat_hidden_dim: int = 512
    strat_hidden_layers: int = 4
    strat_beta: float = 210.0
    strat_epochs: int = 2000
    strat_lr: float = 0.001
    orientation_weight: float = 0.05
    terminal_window_cells: float = 8.0
    terminal_holdout_per_layer: int = 5
    cross_fault_epsilon_cells: float = 0.5
    boundary_exclusion_epsilon: float = 2.0
    max_cross_fault_samples: int = 2000
    prediction_batch_size: int = 131072
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    variants: Tuple[str, ...] = (
        "coord",
        "coord_zero",
        "global_side",
        "effective_domain",
    )


def quick_config(config: ExperimentConfig) -> ExperimentConfig:
    """Small smoke-test configuration; it is not suitable for paper results."""

    return replace(
        config,
        fault_resolution=(25, 19, 19),
        grid_resolution=(45, 33, 33),
        fault_hidden_dim=64,
        fault_epochs=20,
        strat_hidden_dim=96,
        strat_hidden_layers=2,
        strat_epochs=30,
        max_cross_fault_samples=300,
        prediction_batch_size=32768,
        seeds=(0,),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "ablation_outputs" / "paper_case",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--quick", action="store_true", help="Run a non-paper smoke test")
    parser.add_argument("--rebuild-faults", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--variants", nargs="+", choices=VALID_VARIANTS)
    parser.add_argument("--retained-younger-side", type=int, choices=(0, 1))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({name}) but is not available")
    if device.type == "cuda":
        # Materialize the CUDA context before the first autograd call. This
        # avoids a one-time cuBLAS context warning on recent Blackwell GPUs.
        warmup = torch.ones((1, 1), device=device)
        torch.matmul(warmup, warmup)
        torch.cuda.synchronize(device)
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def numeric_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_integer_dtype(result[column]):
            result[column] = result[column].astype(float)
    return result


def load_case(config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    surface = numeric_dataframe(pd.read_csv(REPO_ROOT / config.surface_csv))
    orientation = numeric_dataframe(pd.read_csv(REPO_ROOT / config.orientation_csv))

    observed_fault_order = tuple(
        surface.loc[surface["type"] == "fault", "formation"].drop_duplicates()
    )
    if observed_fault_order != config.fault_names:
        raise ValueError(
            "Fault order is part of the feature definition. "
            f"Expected {config.fault_names}, observed {observed_fault_order}."
        )
    return surface, orientation


def train_or_load_faults(
    config: ExperimentConfig,
    surface: pd.DataFrame,
    orientation: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    rebuild: bool,
) -> List[pv.PolyData]:
    cache_dir = output_dir / "fault_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = [cache_dir / f"{name}.vtp" for name in config.fault_names]
    metadata_path = cache_dir / "cache_config.json"
    source_paths = [
        (REPO_ROOT / config.surface_csv).resolve(),
        (REPO_ROOT / config.orientation_csv).resolve(),
    ]
    cache_config = {
        "surface_csv": config.surface_csv,
        "orientation_csv": config.orientation_csv,
        "source_size": [path.stat().st_size for path in source_paths],
        "source_mtime_ns": [path.stat().st_mtime_ns for path in source_paths],
        "extent": config.extent,
        "fault_names": config.fault_names,
        "fault_resolution": config.fault_resolution,
        "fault_hidden_dim": config.fault_hidden_dim,
        "fault_hidden_layers": config.fault_hidden_layers,
        "fault_beta": config.fault_beta,
        "fault_epochs": config.fault_epochs,
        "fault_lr": config.fault_lr,
    }
    cached_config = None
    if metadata_path.exists():
        cached_config = json.loads(metadata_path.read_text(encoding="utf-8"))

    if (
        not rebuild
        and cached_config == json.loads(json.dumps(cache_config))
        and all(path.exists() for path in cache_paths)
    ):
        print("Loading fixed fault geometry from cache")
        return [pv.read(path) for path in cache_paths]

    print("Training fixed fault geometry (performed once for all ablation groups)")
    set_seed(0)
    meshes = models.fault_ConcatMLP(
        surface,
        orientation,
        list(config.extent),
        resolution=list(config.fault_resolution),
        in_dim=3,
        hidden_dim=config.fault_hidden_dim,
        out_dim=1,
        n_hidden_layers=config.fault_hidden_layers,
        activation="Softplus",
        beta_list=list(config.fault_beta),
        concat=True,
        epochs=config.fault_epochs,
        lr=config.fault_lr,
        above_below=False,
        device=device,
    )
    if len(meshes) != len(config.fault_names):
        raise RuntimeError(f"Expected {len(config.fault_names)} faults, got {len(meshes)}")
    for mesh, path in zip(meshes, cache_paths):
        mesh.extract_surface().save(path)
    metadata_path.write_text(
        json.dumps(cache_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meshes


def encoded_side(
    mesh: pv.DataSet,
    coordinates: np.ndarray,
    movement: str,
    direction: str,
) -> np.ndarray:
    points = pv.PolyData(np.asarray(coordinates, dtype=float))
    return np.asarray(
        Fault_feature.feature_encoding_clip(
            mesh.extract_surface(),
            points,
            move=movement,
            decimate=0.0,
            fault_direct=direction,
        ),
        dtype=float,
    ).reshape(-1)


def implicit_distance(mesh: pv.DataSet, coordinates: np.ndarray) -> np.ndarray:
    points = pv.PolyData(np.asarray(coordinates, dtype=float))
    result = points.compute_implicit_distance(mesh.extract_surface(), inplace=False)
    return np.asarray(result.point_data["implicit_distance"], dtype=float).reshape(-1)


def infer_retained_side(
    config: ExperimentConfig,
    surface: pd.DataFrame,
    meshes: Sequence[pv.DataSet],
) -> Tuple[int, Dict[str, float]]:
    older_idx = config.fault_names.index(config.older_fault)
    younger_idx = config.fault_names.index(config.younger_fault)
    del older_idx  # The observation coordinates, rather than the old fitted surface, are used.

    older_observations = surface.loc[
        (surface["type"] == "fault")
        & (surface["formation"] == config.older_fault),
        ["X", "Y", "Z"],
    ].to_numpy(dtype=float)
    sides = encoded_side(
        meshes[younger_idx],
        older_observations,
        config.movement[younger_idx],
        config.fault_direction[younger_idx],
    )
    side_one_fraction = float(np.isclose(sides, 1.0).mean())
    retained = int(side_one_fraction >= 0.5)
    agreement = max(side_one_fraction, 1.0 - side_one_fraction)
    if agreement < 0.7:
        raise RuntimeError(
            "The retained side cannot be inferred reliably: only "
            f"{agreement:.1%} of older-fault observations agree. "
            "Pass --retained-younger-side explicitly after checking the geometry."
        )
    return retained, {
        "older_fault_observations": int(len(sides)),
        "younger_side_one_fraction": side_one_fraction,
        "retained_side_agreement": agreement,
    }


def feature_encode_all(
    config: ExperimentConfig,
    meshes: Sequence[pv.DataSet],
    surface: pd.DataFrame,
    orientation: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return Fault_feature.encoding(
        extents=list(config.extent),
        resolution=list(config.grid_resolution),
        mesh_list=list(meshes),
        surface_points=surface,
        orientation_points=orientation,
        movement=list(config.movement),
        decimate=0.0,
        fault_direct=list(config.fault_direction),
    )


def cell_spacing(config: ExperimentConfig) -> np.ndarray:
    extent = np.asarray(config.extent, dtype=float)
    resolution = np.asarray(config.grid_resolution, dtype=float)
    return np.array(
        [
            (extent[1] - extent[0]) / (resolution[0] - 1),
            (extent[3] - extent[2]) / (resolution[1] - 1),
            (extent[5] - extent[4]) / (resolution[2] - 1),
        ]
    )


def choose_terminal_holdout(
    config: ExperimentConfig,
    interface: np.ndarray,
    meshes: Sequence[pv.DataSet],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Hold out interface points near the old/young fault intersection."""

    old_idx = config.fault_names.index(config.older_fault)
    young_idx = config.fault_names.index(config.younger_fault)
    xyz = interface[:, 1:4]
    old_distance = np.abs(implicit_distance(meshes[old_idx], xyz))
    young_distance = np.abs(implicit_distance(meshes[young_idx], xyz))
    base_window = config.terminal_window_cells * float(cell_spacing(config).min())

    holdout = np.zeros(len(interface), dtype=bool)
    labels = np.unique(interface[:, 0])
    selection_by_label: Dict[str, object] = {}
    proximity_score = np.sqrt(
        (old_distance / base_window) ** 2 + (young_distance / base_window) ** 2
    )

    for label in labels:
        layer_indices = np.flatnonzero(np.isclose(interface[:, 0], label))
        target = min(config.terminal_holdout_per_layer, layer_indices.size - 1)
        if target < 1:
            raise RuntimeError(f"Layer {label} has too few interface points")

        eligible = np.array([], dtype=int)
        selected_factor = None
        for factor in (1.0, 2.0, 3.0, 4.0):
            within = (
                (old_distance[layer_indices] <= factor * base_window)
                & (young_distance[layer_indices] <= factor * base_window)
            )
            eligible = layer_indices[within]
            if eligible.size >= target:
                selected_factor = factor
                break
        if eligible.size < target:
            # Keep the sample count fixed even for a sparsely sampled layer,
            # but record that the nearest-point fallback was required.
            eligible = layer_indices
            selection_rule = "nearest_intersection_fallback"
        else:
            selection_rule = f"both_faults_within_{selected_factor:g}x_window"

        ordered = eligible[np.argsort(proximity_score[eligible], kind="stable")]
        selected = ordered[:target]
        holdout[selected] = True
        selection_by_label[str(float(label))] = {
            "selection_rule": selection_rule,
            "eligible_count": int(eligible.size),
            "holdout_count": int(selected.size),
            "max_old_fault_distance": float(old_distance[selected].max()),
            "max_young_fault_distance": float(young_distance[selected].max()),
        }

    if holdout.sum() < 3:
        raise RuntimeError(
            "Too few terminal holdout points were found. Increase terminal_window_cells."
        )
    if np.any(
        [
            np.all(holdout[np.isclose(interface[:, 0], label)])
            for label in labels
        ]
    ):
        raise RuntimeError("Holdout selection removed every training point for a layer")

    metadata: Dict[str, object] = {
        "window_physical_units": base_window,
        "holdout_count": int(holdout.sum()),
        "target_per_label": config.terminal_holdout_per_layer,
        "selection_by_label": selection_by_label,
    }
    return holdout, metadata


def normalize_inputs(values: np.ndarray, extent: Sequence[float]) -> np.ndarray:
    # utils.normalize intentionally normalizes only the xyz columns and leaves
    # fault encodings unchanged, which is exactly the required behavior.
    return utils.normalize(np.asarray(values).copy(), list(extent))


def train_stratigraphic_model(
    config: ExperimentConfig,
    interface: np.ndarray,
    orientation: np.ndarray,
    seed: int,
    device: torch.device,
) -> Tuple[ConcatMLP, List[Dict[str, float]], Dict[str, object]]:
    train_x = interface[:, 1:].astype(np.float64)
    train_y = interface[:, 0].astype(np.float64)
    orientation_x = np.delete(orientation.copy(), [3, 4, 5], axis=1).astype(
        np.float64
    )
    orientation_y = orientation[:, 3:6].astype(np.float64)

    normalized_interface = normalize_inputs(train_x, config.extent)
    normalized_orientation = normalize_inputs(orientation_x, config.extent)
    combined = np.vstack((normalized_interface, normalized_orientation))

    x_tensor = torch.tensor(
        combined, dtype=torch.float32, device=device, requires_grad=True
    )
    y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
    dy_tensor = torch.tensor(orientation_y, dtype=torch.float32, device=device)
    n_interface = normalized_interface.shape[0]
    n_orientation = normalized_orientation.shape[0]

    set_seed(seed)
    model = ConcatMLP(
        in_dim=combined.shape[1],
        hidden_dim=config.strat_hidden_dim,
        out_dim=1,
        n_hidden_layers=config.strat_hidden_layers,
        activation="Softplus",
        beta=config.strat_beta,
        concat=True,
    ).to(device)
    initialization_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        initialization_hash.update(name.encode("utf-8"))
        initialization_hash.update(value.detach().cpu().numpy().tobytes())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.strat_lr)

    best_loss = float("inf")
    best_state = None
    best_terms: Dict[str, object] = {}
    history: List[Dict[str, float]] = []
    started = time.time()

    for epoch in range(config.strat_epochs):
        optimizer.zero_grad(set_to_none=True)
        x_tensor.grad = None
        prediction = model(x_tensor)
        interface_loss = lossf.loss_intf_sum(
            prediction[:n_interface].squeeze(), y_tensor
        )
        orientation_loss = lossf.loss_grad_with_fault_features(
            x_tensor, prediction, dy_tensor, n_orientation
        )
        loss = interface_loss + config.orientation_weight * orientation_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at seed={seed}, epoch={epoch + 1}"
            )

        loss_value = float(loss.detach().cpu())
        history.append(
            {
                "epoch": epoch + 1,
                "total_loss": loss_value,
                "interface_loss": float(interface_loss.detach().cpu()),
                "orientation_loss": float(orientation_loss.detach().cpu()),
            }
        )
        if loss_value < best_loss:
            best_loss = loss_value
            # A deep copy is required because state_dict tensors otherwise
            # continue changing with later optimizer steps.
            best_state = copy.deepcopy(model.state_dict())
            best_terms = history[-1].copy()

        loss.backward()
        optimizer.step()

    if best_state is None:
        raise RuntimeError("No finite model checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()
    best_terms["training_seconds"] = time.time() - started
    best_terms["parameter_count"] = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    best_terms["initialization_sha256"] = initialization_hash.hexdigest()
    return model, history, best_terms


def predict_in_batches(
    model: torch.nn.Module,
    inputs: np.ndarray,
    extent: Sequence[float],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    normalized = normalize_inputs(inputs, extent)
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = torch.tensor(
                normalized[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs)


def raw_domain_for_points(
    config: ExperimentConfig,
    meshes: Sequence[pv.DataSet],
    coordinates: np.ndarray,
) -> np.ndarray:
    columns = [np.asarray(coordinates, dtype=float)]
    for index, mesh in enumerate(meshes):
        columns.append(
            encoded_side(
                mesh,
                coordinates,
                config.movement[index],
                config.fault_direction[index],
            )[:, None]
        )
    return np.column_stack(columns)


def inside_extent(points: np.ndarray, extent: Sequence[float]) -> np.ndarray:
    extent = np.asarray(extent, dtype=float)
    return (
        (points[:, 0] >= extent[0])
        & (points[:, 0] <= extent[1])
        & (points[:, 1] >= extent[2])
        & (points[:, 1] <= extent[3])
        & (points[:, 2] >= extent[4])
        & (points[:, 2] <= extent[5])
    )


def prepare_cross_fault_pairs(
    config: ExperimentConfig,
    meshes: Sequence[pv.DataSet],
    retained_side: int,
) -> Dict[str, np.ndarray]:
    old_idx = config.fault_names.index(config.older_fault)
    young_idx = config.fault_names.index(config.younger_fault)
    old_surface = (
        meshes[old_idx]
        .extract_surface()
        .triangulate()
        .compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=False,
        )
    )
    centers = np.asarray(old_surface.points, dtype=float)
    normals = np.asarray(old_surface.point_data["Normals"], dtype=float)

    epsilon = config.cross_fault_epsilon_cells * float(cell_spacing(config).min())
    plus = centers + epsilon * normals
    minus = centers - epsilon * normals
    valid = inside_extent(plus, config.extent) & inside_extent(minus, config.extent)

    young_side = encoded_side(
        meshes[young_idx],
        centers,
        config.movement[young_idx],
        config.fault_direction[young_idx],
    )
    young_distance = np.abs(implicit_distance(meshes[young_idx], centers))
    valid &= young_distance >= config.boundary_exclusion_epsilon * epsilon

    active = np.isclose(young_side, retained_side)
    rng = np.random.default_rng(20260727)
    result: Dict[str, np.ndarray] = {"epsilon": np.array([epsilon])}
    for name, region_mask in (("effective", active), ("invalid", ~active)):
        indices = np.flatnonzero(valid & region_mask)
        if indices.size > config.max_cross_fault_samples:
            indices = rng.choice(
                indices, size=config.max_cross_fault_samples, replace=False
            )
        if indices.size == 0:
            raise RuntimeError(f"No cross-fault samples available in {name} region")
        result[f"{name}_plus"] = plus[indices]
        result[f"{name}_minus"] = minus[indices]
        result[f"{name}_center"] = centers[indices]
    return result


def cross_fault_metric(
    config: ExperimentConfig,
    model: torch.nn.Module,
    variant: str,
    meshes: Sequence[pv.DataSet],
    rules: Sequence[TemporalRule],
    pairs: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    metrics: Dict[str, float] = {}
    point_values: Dict[str, np.ndarray] = {}
    for region in ("effective", "invalid"):
        plus_raw = raw_domain_for_points(config, meshes, pairs[f"{region}_plus"])
        minus_raw = raw_domain_for_points(config, meshes, pairs[f"{region}_minus"])
        plus_x, _ = transform_domain(
            plus_raw, variant, config.fault_names, rules
        )
        minus_x, _ = transform_domain(
            minus_raw, variant, config.fault_names, rules
        )
        plus_scalar = predict_in_batches(
            model,
            plus_x,
            config.extent,
            device,
            config.prediction_batch_size,
        )
        minus_scalar = predict_in_batches(
            model,
            minus_x,
            config.extent,
            device,
            config.prediction_batch_size,
        )
        difference = np.abs(plus_scalar - minus_scalar)
        metrics[f"delta_{region}_mean"] = float(difference.mean())
        metrics[f"delta_{region}_std"] = float(difference.std(ddof=1))
        metrics[f"delta_{region}_n"] = int(difference.size)
        point_values[region] = difference
    return metrics, point_values


def scalar_grid(
    config: ExperimentConfig,
    predictions: np.ndarray,
) -> pv.ImageData:
    grid = pv.ImageData()
    grid.dimensions = list(config.grid_resolution)
    grid.origin = [config.extent[0], config.extent[2], config.extent[4]]
    spacing = cell_spacing(config)
    grid.spacing = spacing.tolist()
    grid.point_data["scalar"] = np.asarray(predictions).reshape(-1)
    return grid


def terminal_geometry_rmse(
    grid: pv.ImageData,
    holdout_interface: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    squared_distances: List[np.ndarray] = []
    per_label: Dict[str, float] = {}
    for label in np.unique(holdout_interface[:, 0]):
        points = holdout_interface[
            np.isclose(holdout_interface[:, 0], label), 1:4
        ]
        surface = grid.contour(isosurfaces=[float(label)], scalars="scalar")
        if surface.n_points == 0:
            per_label[str(float(label))] = float("nan")
            continue
        distance = np.abs(implicit_distance(surface, points))
        squared_distances.append(distance**2)
        per_label[str(float(label))] = float(np.sqrt(np.mean(distance**2)))
    if not squared_distances:
        return float("nan"), per_label
    all_squared = np.concatenate(squared_distances)
    return float(np.sqrt(np.mean(all_squared))), per_label


def write_history(path: Path, history: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_sample_points(
    path: Path,
    pairs: Dict[str, np.ndarray],
    values: Dict[str, np.ndarray],
) -> None:
    centers = np.vstack((pairs["effective_center"], pairs["invalid_center"]))
    region = np.concatenate(
        (
            np.ones(len(pairs["effective_center"]), dtype=np.int8),
            np.zeros(len(pairs["invalid_center"]), dtype=np.int8),
        )
    )
    delta = np.concatenate((values["effective"], values["invalid"]))
    cloud = pv.PolyData(centers)
    cloud.point_data["effective_region"] = region
    cloud.point_data["cross_fault_delta"] = delta
    cloud.save(path)


def summarize(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "metrics_all_runs.csv", index=False)

    metric_columns = [
        "delta_effective_mean",
        "delta_invalid_mean",
        "terminal_interface_rmse",
        "best_total_loss",
        "training_seconds",
    ]
    summary_rows: List[Dict[str, object]] = []
    for variant, group in frame.groupby("variant", sort=False):
        row: Dict[str, object] = {"variant": variant, "runs": len(group)}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    labels = summary["variant"].tolist()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_specs = [
        ("delta_effective_mean", "Effective-region cross-fault difference ↑"),
        ("delta_invalid_mean", "Invalid-region residual difference ↓"),
        ("terminal_interface_rmse", "Terminal interface RMSE ↓"),
    ]
    for axis, (column, title) in zip(axes, plot_specs):
        axis.bar(
            labels,
            summary[f"{column}_mean"],
            yerr=summary[f"{column}_std"].fillna(0),
            capsize=4,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "ablation_summary.png", dpi=300)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    if args.quick:
        config = quick_config(config)
    if args.seeds:
        config = replace(config, seeds=tuple(args.seeds))
    if args.variants:
        config = replace(config, variants=tuple(args.variants))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Repository: {REPO_ROOT}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    if args.quick:
        print("QUICK MODE: outputs are only for code validation, not for the paper")

    surface, orientation = load_case(config)
    meshes = train_or_load_faults(
        config,
        surface,
        orientation,
        output_dir,
        device,
        rebuild=args.rebuild_faults,
    )

    inferred_side, side_metadata = infer_retained_side(config, surface, meshes)
    retained_side = (
        args.retained_younger_side
        if args.retained_younger_side is not None
        else inferred_side
    )
    rules = (
        TemporalRule(
            older_fault=config.older_fault,
            younger_fault=config.younger_fault,
            retained_younger_side=retained_side,
        ),
    )
    print(
        f"Temporal rule: {config.older_fault} remains active where "
        f"{config.younger_fault} side == {retained_side}"
    )

    domain_raw, interface_raw, orientation_raw = feature_encode_all(
        config, meshes, surface, orientation
    )
    holdout_mask, holdout_metadata = choose_terminal_holdout(
        config, interface_raw, meshes
    )
    train_interface_raw = interface_raw[~holdout_mask]
    holdout_interface_raw = interface_raw[holdout_mask]
    pairs = prepare_cross_fault_pairs(config, meshes, retained_side)

    run_metadata = {
        "config": asdict(config),
        "device": str(device),
        "quick_mode": bool(args.quick),
        "retained_younger_side": retained_side,
        "retained_side_inference": side_metadata,
        "terminal_holdout": holdout_metadata,
        "cross_fault_epsilon_physical_units": float(pairs["epsilon"][0]),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "pyvista_version": pv.__version__,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(output_dir / "terminal_holdout_mask.npy", holdout_mask)

    all_rows: List[Dict[str, object]] = []
    for seed in config.seeds:
        for variant in config.variants:
            print(f"\n=== seed={seed} variant={variant} ===")
            run_dir = output_dir / f"seed_{seed}" / variant
            run_dir.mkdir(parents=True, exist_ok=True)

            domain, interface, orientation_variant, feature_metadata = build_variant(
                domain_raw,
                train_interface_raw,
                orientation_raw,
                variant,
                config.fault_names,
                rules,
            )
            holdout_domain = np.column_stack(
                (holdout_interface_raw[:, 1:4], holdout_interface_raw[:, 4:])
            )
            holdout_x, _ = transform_domain(
                holdout_domain, variant, config.fault_names, rules
            )
            holdout_interface = np.column_stack(
                (holdout_interface_raw[:, 0], holdout_x)
            )

            model, history, best = train_stratigraphic_model(
                config, interface, orientation_variant, seed, device
            )
            write_history(run_dir / "loss_history.csv", history)
            if args.save_checkpoints:
                torch.save(model.state_dict(), run_dir / "model.pt")

            grid_prediction = predict_in_batches(
                model,
                domain,
                config.extent,
                device,
                config.prediction_batch_size,
            )
            grid = scalar_grid(config, grid_prediction)
            grid.save(run_dir / "scalar_field.vti")
            interface_mesh = grid.contour(
                isosurfaces=np.unique(interface_raw[:, 0]).tolist(),
                scalars="scalar",
            )
            interface_mesh.save(run_dir / "interfaces.vtp")

            cross_metrics, sample_values = cross_fault_metric(
                config, model, variant, meshes, rules, pairs, device
            )
            terminal_rmse, terminal_by_label = terminal_geometry_rmse(
                grid, holdout_interface
            )
            save_sample_points(
                run_dir / "cross_fault_samples.vtp", pairs, sample_values
            )

            row: Dict[str, object] = {
                "seed": seed,
                "variant": variant,
                "input_dim": int(interface.shape[1] - 1),
                "best_epoch": int(best["epoch"]),
                "best_total_loss": best["total_loss"],
                "best_interface_loss": best["interface_loss"],
                "best_orientation_loss": best["orientation_loss"],
                "training_seconds": best["training_seconds"],
                "parameter_count": int(best["parameter_count"]),
                "initialization_sha256": best["initialization_sha256"],
                "terminal_interface_rmse": terminal_rmse,
                "terminal_holdout_n": int(len(holdout_interface)),
                **cross_metrics,
            }
            all_rows.append(row)
            details = {
                "metrics": row,
                "terminal_rmse_by_label": terminal_by_label,
                "feature_metadata": feature_metadata,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(details, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(row, ensure_ascii=False, indent=2))

            del model, grid
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summarize(all_rows, output_dir)
    print(f"\nFinished. Summary: {output_dir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
