"""Run the preregistered R1B controlled experiment in an independent directory."""

from __future__ import annotations

import copy
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation.r1b_core import (
    apply_hierarchical_masks,
    eikonal_loss_normalized,
    enumerate_binary_states,
    orientation_cosine_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
R1B_ROOT = REPO_ROOT
GEOMETRY_ROOT = REPO_ROOT / "data" / "case1" / "fault_meshes"
RESULTS_ROOT = REPO_ROOT / "ablation_outputs" / "r1b_formal"
FIGURES_ROOT = RESULTS_ROOT / "figures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ablation.case_ablation as exp  # noqa: E402
from ablation.r1b_bundle import (  # noqa: E402
    R1B_ASSET_SHA256,
    require_training_opt_in,
    validate_asset_hashes,
)


FAULT_NAMES = ("fault1", "fault2")
VARIANTS = ("coord", "coord_zero", "global_side", "effective_domain")
SEEDS = (0, 1, 2, 3, 4)
RETAINED_SIDE = 0
MASK_RULES = (("fault1", "fault2", 1, 1),)
EIKONAL_SAMPLES = 4096
EIKONAL_WEIGHT = 0.01


@dataclass(frozen=True)
class R1BConfig:
    extent: Tuple[float, ...] = (-1800.0, 164.0, -160.0, 1400.0, 500.0, 1450.0)
    grid_resolution: Tuple[int, ...] = (100, 100, 100)
    fault_resolution: Tuple[int, ...] = (60, 30, 30)
    hidden_dim: int = 256
    hidden_layers: int = 2
    softplus_beta: float = 1.0
    epochs: int = 1000
    learning_rate: float = 0.001
    orientation_weight: float = 0.1
    eikonal_weight: float = 0.01
    eikonal_samples: int = 4096
    terminal_holdout_per_label: int = 5
    terminal_window_physical: float = 80.0
    cross_fault_epsilon_physical: float = 5.0
    boundary_exclusion_epsilon: float = 2.0
    max_cross_fault_samples: int = 2000
    prediction_batch_size: int = 131072
    seeds: Tuple[int, ...] = SEEDS
    variants: Tuple[str, ...] = VARIANTS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sample_eikonal_coordinates(seed: int, extent: Sequence[float], count: int) -> np.ndarray:
    """Generate the registered fixed physical-domain Eikonal point set."""

    bounds = np.asarray(extent, dtype=float).reshape(3, 2)
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(lo, hi, count) for lo, hi in bounds])


def relative_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return (baseline - candidate) / baseline


def pad_variant_features(array: np.ndarray, role: str, n_faults: int = 2) -> np.ndarray:
    """Give every formal variant the same input width with zero placeholders."""

    values = np.asarray(array)
    target = {
        "domain": 3 + n_faults,
        "interface": 1 + 3 + n_faults,
        "orientation": 3 + 3 + n_faults,
    }[role]
    if values.shape[1] == target:
        return values
    base = {"domain": 3, "interface": 4, "orientation": 6}[role]
    if values.shape[1] != base:
        raise ValueError(f"Unexpected {role} width {values.shape[1]}; expected {base} or {target}")
    zeros = np.zeros((values.shape[0], target - base), dtype=values.dtype)
    if role == "interface":
        return np.column_stack((values[:, :1], values[:, 1:], zeros))
    if role == "orientation":
        return np.column_stack((values, zeros))
    return np.column_stack((values, zeros))


def source_config() -> exp.ExperimentConfig:
    base = exp.ExperimentConfig()
    config = replace(
        base,
        surface_csv="data/case1/surface.csv",
        orientation_csv="data/case1/orientation.csv",
        fault_resolution=(60, 30, 30),
        grid_resolution=(100, 100, 100),
        fault_hidden_dim=256,
        fault_hidden_layers=2,
        fault_beta=(10.0, 10.0),
        fault_epochs=500,
        fault_lr=0.001,
        fault_eikonal_weight=0.0,
        fault_eikonal_samples=0,
        fault_above_below=True,
        strat_hidden_dim=256,
        strat_hidden_layers=2,
        strat_beta=1.0,
        strat_epochs=1000,
        strat_lr=0.001,
        orientation_weight=0.1,
        terminal_holdout_per_layer=5,
        max_cross_fault_samples=2000,
        prediction_batch_size=131072,
        seeds=SEEDS,
        variants=VARIANTS,
    )
    spacing = float(exp.cell_spacing(config).min())
    return replace(
        config,
        terminal_window_cells=80.0 / spacing,
        cross_fault_epsilon_cells=5.0 / spacing,
        boundary_exclusion_epsilon=2.0,
    )


def load_raw_meshes() -> Tuple[List[pv.PolyData], List[Optional[pv.PolyData]]]:
    raw = [
        pv.read(GEOMETRY_ROOT / f"{name}_raw.vtp")
        for name in FAULT_NAMES
    ]
    clipped = []
    for name in FAULT_NAMES:
        path = GEOMETRY_ROOT / f"{name}_clipped.vtp"
        clipped.append(pv.read(path) if path.exists() else None)
    return raw, clipped


def preflight_inputs() -> Dict[str, Any]:
    """Check the frozen data bundle without initializing or training a model."""

    hashes = validate_asset_hashes(REPO_ROOT, R1B_ASSET_SHA256)
    config = source_config()
    surface, orientation = exp.load_case(config)
    raw_meshes, _ = load_raw_meshes()
    return {
        "status": "R1B_PREFLIGHT_PASS",
        "training_performed": False,
        "input_hashes": hashes,
        "surface_rows": int(len(surface)),
        "orientation_rows": int(len(orientation)),
        "fault_mesh_points": {
            name: int(mesh.n_points) for name, mesh in zip(FAULT_NAMES, raw_meshes)
        },
        "seeds": list(config.seeds),
        "variants": list(config.variants),
        "formal_training_units": len(config.seeds) * len(config.variants),
        "normalization_extent": list(config.extent),
        "eikonal_points_per_seed": EIKONAL_SAMPLES,
        "eikonal_weight_normalized_coordinates": EIKONAL_WEIGHT,
        "orientation_gradient": "physical gradient obtained by Jacobian transform",
    }


def build_config_record(config: exp.ExperimentConfig, device: torch.device) -> Dict[str, Any]:
    selected_surface = REPO_ROOT / config.surface_csv
    selected_orientation = REPO_ROOT / config.orientation_csv
    raw_meshes, clipped_meshes = load_raw_meshes()
    input_hashes = validate_asset_hashes(REPO_ROOT, R1B_ASSET_SHA256)
    return {
        "repository_root": ".",
        "runner": "ablation/r1b_experiment.py",
        "source_code": {
            "case_ablation": "ablation/case_ablation.py",
            "feature_variants": "ablation/feature_variants.py",
            "mlps": "mlps.py",
            "lossf": "lossf.py",
            "utils": "utils.py",
            "r1b_core": "ablation/r1b_core.py",
            "r1b_experiment": "ablation/r1b_experiment.py",
        },
        "source_hashes": {
            "case_ablation.py": sha256_file(REPO_ROOT / "ablation" / "case_ablation.py"),
            "feature_variants.py": sha256_file(REPO_ROOT / "ablation" / "feature_variants.py"),
            "mlps.py": sha256_file(REPO_ROOT / "mlps.py"),
            "lossf.py": sha256_file(REPO_ROOT / "lossf.py"),
            "utils.py": sha256_file(REPO_ROOT / "utils.py"),
            "r1b_core.py": sha256_file(REPO_ROOT / "ablation" / "r1b_core.py"),
            "r1b_experiment.py": sha256_file(REPO_ROOT / "ablation" / "r1b_experiment.py"),
        },
        "input_hashes": {
            "surface.csv": input_hashes["data/case1/surface.csv"],
            "orientation.csv": input_hashes["data/case1/orientation.csv"],
            "fault1_raw.vtp": input_hashes["data/case1/fault_meshes/fault1_raw.vtp"],
            "fault2_raw.vtp": input_hashes["data/case1/fault_meshes/fault2_raw.vtp"],
        },
        "dataset": {
            "surface": str(selected_surface),
            "orientation": str(selected_orientation),
            "surface_rows": int(len(pd.read_csv(selected_surface))),
            "orientation_rows": int(len(pd.read_csv(selected_orientation))),
        },
        "geometry": {
            "fault_names": FAULT_NAMES,
            "raw_points": [int(mesh.n_points) for mesh in raw_meshes],
            "clipped_points": [int(mesh.n_points) if mesh is not None else None for mesh in clipped_meshes],
            "clipped_meshes_available": all(mesh is not None for mesh in clipped_meshes),
            "same_raw_mesh_for_all_variants": True,
            "retained_side": RETAINED_SIDE,
            "rules": MASK_RULES,
        },
        "config": asdict(config),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyvista": pv.__version__,
        },
        "training_protocol": {
            "formal_stratigraphic_runs": 20,
            "fault_network_training_in_r1b": 0,
            "eikonal_samples_per_seed": EIKONAL_SAMPLES,
            "eikonal_weight_normalized_coordinate_space": EIKONAL_WEIGHT,
            "eikonal_points_fixed_and_reused_across_variants": True,
        },
        "device": str(device),
    }


def mesh_geometry_diagnostic(config: exp.ExperimentConfig) -> Dict[str, Any]:
    raw_meshes, clipped_meshes = load_raw_meshes()
    points = sample_eikonal_coordinates(20260727, config.extent, 10000)
    raw_features = exp.raw_domain_for_points(config, raw_meshes, points)[:, 3:].astype(np.int8)
    raw_effective = apply_hierarchical_masks(raw_features, FAULT_NAMES, MASK_RULES)
    per_fault = []
    for idx, name in enumerate(FAULT_NAMES):
        if clipped_meshes[idx] is None:
            per_fault.append(
                {
                    "fault": name,
                    "raw_points": int(raw_meshes[idx].n_points),
                    "clipped_points": None,
                    "different_encoding_points": None,
                    "different_fraction": None,
                    "status": "CLIPPED_MESH_NOT_RECOVERED",
                }
            )
            continue
        comparison_meshes = list(raw_meshes)
        comparison_meshes[idx] = clipped_meshes[idx]
        clipped_features = exp.raw_domain_for_points(config, comparison_meshes, points)[:, 3:].astype(np.int8)
        differences = raw_features[:, idx] != clipped_features[:, idx]
        per_fault.append(
            {
                "fault": name,
                "raw_points": int(raw_meshes[idx].n_points),
                "clipped_points": int(clipped_meshes[idx].n_points),
                "different_encoding_points": int(differences.sum()),
                "different_fraction": float(differences.mean()),
                "status": "COMPARED",
            }
        )
    states = enumerate_binary_states(FAULT_NAMES)
    effective_states = apply_hierarchical_masks(states, FAULT_NAMES, MASK_RULES)
    state_rows = [
        {"raw": row.tolist(), "effective": effective.tolist()}
        for row, effective in zip(states, effective_states)
    ]
    clipped_available = all(mesh is not None for mesh in clipped_meshes)
    clipped_effective_different_rows = None
    if clipped_available:
        clipped_features_all = exp.raw_domain_for_points(config, clipped_meshes, points)[:, 3:].astype(np.int8)
        clipped_effective = apply_hierarchical_masks(clipped_features_all, FAULT_NAMES, MASK_RULES)
        clipped_effective_different_rows = int(np.any(raw_effective != clipped_effective, axis=1).sum())
    return {
        "sample_seed": 20260727,
        "sample_count": 10000,
        "raw_unique_states": int(len(np.unique(raw_features, axis=0))),
        "effective_unique_states": int(len(np.unique(raw_effective, axis=0))),
        "clipped_meshes_available": clipped_available,
        "raw_final_different_rows": None if not clipped_available else int(
            np.any(raw_features != clipped_features_all, axis=1).sum()
        ),
        "effective_raw_final_different_rows": clipped_effective_different_rows,
        "historical_engineering_clipped_comparison": {
            "source": "outputs/CAGEO_R1A_MASK_ORACLE.md",
            "status": "RECORDED_SEPARATELY_NOT_USED_IN_R1B_FORMAL_TRAINING",
            "f4_different_fraction": 0.0256,
        },
        "per_fault": per_fault,
        "state_table": state_rows,
        "same_raw_mesh_used_for_all_variants": True,
    }


def train_corrected_stratigraphic(
    config: exp.ExperimentConfig,
    interface: np.ndarray,
    orientation: np.ndarray,
    eikonal_domain: np.ndarray,
    seed: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, List[Dict[str, float]], Dict[str, Any]]:
    train_x = interface[:, 1:].astype(np.float64)
    train_y = interface[:, 0].astype(np.float64)
    orientation_x = np.delete(orientation.copy(), [3, 4, 5], axis=1).astype(np.float64)
    orientation_y = orientation[:, 3:6].astype(np.float64)
    normalized_interface = exp.normalize_inputs(train_x, config.extent)
    normalized_orientation = exp.normalize_inputs(orientation_x, config.extent)
    normalized_eikonal = exp.normalize_inputs(eikonal_domain, config.extent)
    combined = np.vstack((normalized_interface, normalized_orientation))

    x_tensor = torch.tensor(combined, dtype=torch.float32, device=device, requires_grad=True)
    y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
    dy_tensor = torch.tensor(orientation_y, dtype=torch.float32, device=device)
    eikonal_tensor = torch.tensor(
        normalized_eikonal, dtype=torch.float32, device=device, requires_grad=True
    )
    n_interface = normalized_interface.shape[0]
    n_orientation = normalized_orientation.shape[0]
    ranges = torch.tensor(
        [config.extent[1] - config.extent[0], config.extent[3] - config.extent[2], config.extent[5] - config.extent[4]],
        dtype=torch.float32,
        device=device,
    )

    exp.set_seed(seed)
    model = exp.ConcatMLP(
        in_dim=combined.shape[1],
        hidden_dim=config.strat_hidden_dim,
        out_dim=1,
        n_hidden_layers=config.strat_hidden_layers,
        activation="Softplus",
        beta=config.strat_beta,
        concat=True,
    ).to(device)
    init_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        init_hash.update(name.encode("utf-8"))
        init_hash.update(value.detach().cpu().numpy().tobytes())

    optimizer = torch.optim.Adam(model.parameters(), lr=config.strat_lr)
    best_loss = float("inf")
    best_state = None
    best_terms: Dict[str, Any] = {}
    history: List[Dict[str, float]] = []
    started = time.time()

    for epoch in range(config.strat_epochs):
        optimizer.zero_grad(set_to_none=True)
        x_tensor.grad = None
        eikonal_tensor.grad = None
        prediction = model(x_tensor)
        interface_loss = F.mse_loss(prediction[:n_interface].squeeze(), y_tensor)
        gradients = torch.autograd.grad(
            outputs=prediction,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
        )[0]
        orientation_loss = orientation_cosine_loss(
            gradients[-n_orientation:, :3], dy_tensor, ranges
        )
        eikonal_prediction = model(eikonal_tensor)
        eikonal_gradient = torch.autograd.grad(
            outputs=eikonal_prediction,
            inputs=eikonal_tensor,
            grad_outputs=torch.ones_like(eikonal_prediction),
            create_graph=True,
            retain_graph=True,
        )[0][:, :3]
        eikonal_term = eikonal_loss_normalized(eikonal_gradient)
        loss = (
            interface_loss
            + config.orientation_weight * orientation_loss
            + EIKONAL_WEIGHT * eikonal_term
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at seed={seed}, epoch={epoch + 1}")
        row = {
            "epoch": epoch + 1,
            "total_loss": float(loss.detach().cpu()),
            "interface_loss": float(interface_loss.detach().cpu()),
            "orientation_loss": float(orientation_loss.detach().cpu()),
            "eikonal_loss": float(eikonal_term.detach().cpu()),
        }
        history.append(row)
        if row["total_loss"] < best_loss:
            best_loss = row["total_loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_terms = row.copy()
        loss.backward()
        optimizer.step()

    if best_state is None:
        raise RuntimeError("no finite checkpoint produced")
    model.load_state_dict(best_state)
    model.eval()
    best_terms.update(
        {
            "training_seconds": time.time() - started,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "initialization_sha256": init_hash.hexdigest(),
        }
    )
    return model, history, best_terms


def cross_fault_metric_fixed(
    config: exp.ExperimentConfig,
    model: torch.nn.Module,
    variant: str,
    meshes: Sequence[pv.DataSet],
    rules: Sequence[exp.TemporalRule],
    pairs: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate cross-fault responses using the same padded width as training."""

    metrics: Dict[str, float] = {}
    point_values: Dict[str, np.ndarray] = {}
    for region in ("effective", "invalid"):
        plus_raw = exp.raw_domain_for_points(config, meshes, pairs[f"{region}_plus"])
        minus_raw = exp.raw_domain_for_points(config, meshes, pairs[f"{region}_minus"])
        plus_x, _ = exp.transform_domain(plus_raw, variant, config.fault_names, rules)
        minus_x, _ = exp.transform_domain(minus_raw, variant, config.fault_names, rules)
        plus_x = pad_variant_features(plus_x, "domain", len(config.fault_names))
        minus_x = pad_variant_features(minus_x, "domain", len(config.fault_names))
        plus_scalar = exp.predict_in_batches(
            model, plus_x, config.extent, device, config.prediction_batch_size
        )
        minus_scalar = exp.predict_in_batches(
            model, minus_x, config.extent, device, config.prediction_batch_size
        )
        difference = np.abs(plus_scalar - minus_scalar)
        metrics[f"delta_{region}_mean"] = float(difference.mean())
        metrics[f"delta_{region}_std"] = float(difference.std(ddof=1))
        metrics[f"delta_{region}_n"] = int(difference.size)
        point_values[region] = difference
    return metrics, point_values


def write_history(path: Path, history: Sequence[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def write_summary(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_ROOT / "metrics_all_runs.csv", index=False)
    summary_rows: List[Dict[str, Any]] = []
    for variant, group in frame.groupby("variant", sort=False):
        row: Dict[str, Any] = {"variant": variant, "runs": int(len(group))}
        for column in (
            "delta_effective_mean",
            "delta_invalid_mean",
            "terminal_interface_rmse",
            "best_total_loss",
            "training_seconds",
        ):
            values = pd.to_numeric(group[column])
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_ROOT / "metrics_summary.csv", index=False)
    return summary


def write_paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot(index="seed", columns="variant", values=["delta_invalid_mean", "terminal_interface_rmse"])
    paired = pd.DataFrame(index=pivot.index)
    paired["invalid_global_minus_effective"] = (
        pivot["delta_invalid_mean"]["global_side"]
        - pivot["delta_invalid_mean"]["effective_domain"]
    )
    paired["invalid_reduction_vs_global"] = paired["invalid_global_minus_effective"] / pivot["delta_invalid_mean"]["global_side"]
    paired["terminal_effective_minus_coord"] = (
        pivot["terminal_interface_rmse"]["effective_domain"]
        - pivot["terminal_interface_rmse"]["coord"]
    )
    paired.to_csv(RESULTS_ROOT / "paired_comparisons.csv")
    return paired


def dataframe_text(frame: pd.DataFrame, *, index: bool = False) -> str:
    """Render a dependency-free fixed-width table for the audit markdown."""

    return "```text\n" + frame.to_string(index=index) + "\n```"


def plot_metrics(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    colors = {"coord": "#0072B2", "coord_zero": "#56B4E9", "global_side": "#E69F00", "effective_domain": "#009E73"}
    columns = [
        ("delta_effective_mean_mean", "Effective-region response"),
        ("delta_invalid_mean_mean", "Invalid-region response"),
        ("terminal_interface_rmse_mean", "Terminal-interface RMSE (m)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, (column, title) in zip(axes, columns):
        values = [summary.loc[summary.variant == v, column].iloc[0] for v in VARIANTS]
        std_column = column.rsplit("_mean", 1)[0] + "_std"
        errors = [summary.loc[summary.variant == v, std_column].iloc[0] for v in VARIANTS]
        ax.bar(range(len(VARIANTS)), values, yerr=errors, capsize=3, color=[colors[v] for v in VARIANTS], edgecolor="black", linewidth=0.4)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(len(VARIANTS)), ["coord", "zero", "global", "effective"], rotation=35, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES_ROOT / "r1b_metrics_comparison.png", dpi=300)
    fig.savefig(FIGURES_ROOT / "r1b_metrics_comparison.pdf")
    fig.savefig(FIGURES_ROOT / "r1b_metrics_comparison.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 2.8), constrained_layout=True)
    x = np.arange(len(paired.index))
    ax.axhline(0.8, color="black", linestyle="--", linewidth=0.8, label="registered 80% threshold")
    ax.plot(x, paired["invalid_reduction_vs_global"], marker="o", color=colors["effective_domain"], linewidth=1.4)
    ax.set_xticks(x, [f"seed {s}" for s in paired.index], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Invalid-response reduction vs global side", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(FIGURES_ROOT / "r1b_invalid_reduction_by_seed.png", dpi=300)
    fig.savefig(FIGURES_ROOT / "r1b_invalid_reduction_by_seed.pdf")
    plt.close(fig)


def evaluate_go(summary: pd.DataFrame, paired: pd.DataFrame, geometry: Dict[str, Any]) -> Dict[str, Any]:
    effective = paired["invalid_reduction_vs_global"]
    effective_terminal = float(summary.loc[summary.variant == "effective_domain", "terminal_interface_rmse_mean"].iloc[0])
    coord_terminal = float(summary.loc[summary.variant == "coord", "terminal_interface_rmse_mean"].iloc[0])
    checks = {
        "invalid_lower_in_at_least_4_of_5_seeds": int((paired["invalid_global_minus_effective"] > 0).sum()) >= 4,
        "median_invalid_reduction_at_least_80_percent": float(effective.median()) >= 0.8,
        "terminal_rmse_not_more_than_10_percent_worse_than_coord": effective_terminal <= 1.10 * coord_terminal,
        "no_raw_final_geometry_difference_in_training_path": bool(geometry["same_raw_mesh_used_for_all_variants"]),
        "geometry_diagnostic_recorded": True,
        "all_twenty_cells_present": int(summary["runs"].sum()) == 20,
    }
    return {
        "status": "R1B_GO" if all(checks.values()) else "R1B_NO_GO",
        "checks": checks,
        "effective_terminal_rmse": effective_terminal,
        "coord_terminal_rmse": coord_terminal,
        "median_invalid_reduction": float(effective.median()),
        "positive_seed_count": int((paired["invalid_global_minus_effective"] > 0).sum()),
    }


def finalize_saved_results() -> None:
    """Finalize an already completed 20-cell run without retraining."""

    metric_paths = sorted(RESULTS_ROOT.glob("seed_*/**/metrics.json"))
    rows = [json.loads(path.read_text(encoding="utf-8"))["metrics"] for path in metric_paths]
    frame = pd.DataFrame(rows)
    if len(frame) != 20:
        raise RuntimeError(f"expected 20 saved metric files, got {len(frame)}")
    geometry = json.loads((RESULTS_ROOT / "geometry_diagnostic.json").read_text(encoding="utf-8"))
    summary = write_summary(rows)
    paired = write_paired_comparisons(frame)
    plot_metrics(summary, paired)
    decision = evaluate_go(summary, paired, geometry)
    decision["geometry_diagnostic"] = geometry
    decision["finalized_without_retraining"] = True
    (RESULTS_ROOT / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_ROOT / "numeric_comparison.md").write_text(
        "# CAGEO-R1B Numeric Comparison\n\n"
        + dataframe_text(summary)
        + "\n\n## Paired comparisons\n\n"
        + dataframe_text(paired, index=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    global RESULTS_ROOT, FIGURES_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--run-formal",
        action="store_true",
        help="explicitly start the 20-cell neural experiment",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    if sum((args.preflight_only, args.finalize_only, args.run_formal)) > 1:
        parser.error("choose only one of --preflight-only, --finalize-only, or --run-formal")
    try:
        training_requested = require_training_opt_in(
            run_formal=args.run_formal,
            preflight_only=args.preflight_only,
            finalize_only=args.finalize_only,
        )
    except ValueError as exc:
        parser.error(str(exc))

    RESULTS_ROOT = args.output_dir.resolve()
    FIGURES_ROOT = RESULTS_ROOT / "figures"
    if args.preflight_only:
        print(json.dumps(preflight_inputs(), ensure_ascii=False, indent=2))
        return
    if args.finalize_only:
        finalize_saved_results()
        return
    if not training_requested:
        parser.error("training was not explicitly requested")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = source_config()
    device = exp.resolve_device(args.device)
    record = build_config_record(config, device)
    (RESULTS_ROOT / "r1b_run_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    geometry = mesh_geometry_diagnostic(config)
    (RESULTS_ROOT / "geometry_diagnostic.json").write_text(json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8")
    if geometry["effective_raw_final_different_rows"] is not None and geometry["effective_raw_final_different_rows"] > 0:
        # This is recorded, not silently repaired: all formal variants use raw meshes.
        pass

    surface, orientation = exp.load_case(config)
    raw_meshes, _ = load_raw_meshes()
    domain_raw, interface_raw, orientation_raw = exp.feature_encode_all(
        config, raw_meshes, surface, orientation
    )
    holdout_mask, holdout_metadata = exp.choose_terminal_holdout(config, interface_raw, raw_meshes)
    train_interface_raw = interface_raw[~holdout_mask]
    holdout_interface_raw = interface_raw[holdout_mask]
    pairs = exp.prepare_cross_fault_pairs(config, raw_meshes, RETAINED_SIDE)
    rules = (exp.TemporalRule(config.older_fault, config.younger_fault, RETAINED_SIDE),)

    (RESULTS_ROOT / "split_metadata.json").write_text(
        json.dumps(
            {
                "holdout": holdout_metadata,
                "train_interface_count": int(len(train_interface_raw)),
                "holdout_interface_count": int(len(holdout_interface_raw)),
                "pair_counts": {key: int(len(value)) for key, value in pairs.items() if key.endswith("_center")},
                "rule": {"older_fault": config.older_fault, "younger_fault": config.younger_fault, "retained_side": RETAINED_SIDE},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: List[Dict[str, Any]] = []
    for seed in config.seeds:
        eikonal_path = RESULTS_ROOT / f"eikonal_points_seed_{seed}.npy"
        if eikonal_path.exists():
            eikonal_coordinates = np.load(eikonal_path)
        else:
            eikonal_coordinates = sample_eikonal_coordinates(seed, config.extent, EIKONAL_SAMPLES)
            np.save(eikonal_path, eikonal_coordinates)
        if eikonal_coordinates.shape != (EIKONAL_SAMPLES, 3):
            raise RuntimeError(f"invalid Eikonal point shape for seed {seed}")
        eikonal_raw = exp.raw_domain_for_points(config, raw_meshes, eikonal_coordinates)
        eikonal_hash = sha256_file(eikonal_path)
        for variant in config.variants:
            run_dir = RESULTS_ROOT / f"seed_{seed}" / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            domain, interface, orientation_variant, feature_metadata = exp.build_variant(
                domain_raw,
                train_interface_raw,
                orientation_raw,
                variant,
                config.fault_names,
                rules,
            )
            domain = pad_variant_features(domain, "domain", len(config.fault_names))
            interface = pad_variant_features(interface, "interface", len(config.fault_names))
            orientation_variant = pad_variant_features(
                orientation_variant, "orientation", len(config.fault_names)
            )
            holdout_domain = np.column_stack((holdout_interface_raw[:, 1:4], holdout_interface_raw[:, 4:]))
            holdout_x, _ = exp.transform_domain(holdout_domain, variant, config.fault_names, rules)
            holdout_x = pad_variant_features(holdout_x, "domain", len(config.fault_names))
            holdout_interface = np.column_stack((holdout_interface_raw[:, 0], holdout_x))
            eikonal_domain, eikonal_metadata = exp.transform_domain(
                eikonal_raw, variant, config.fault_names, rules
            )
            eikonal_domain = pad_variant_features(eikonal_domain, "domain", len(config.fault_names))
            model, history, best = train_corrected_stratigraphic(
                config,
                interface,
                orientation_variant,
                eikonal_domain,
                seed,
                device,
            )
            write_history(run_dir / "loss_history.csv", history)
            torch.save(model.state_dict(), run_dir / "model.pt")
            metric_values, sample_values = cross_fault_metric_fixed(
                config, model, variant, raw_meshes, rules, pairs, device
            )
            grid_prediction = exp.predict_in_batches(
                model, domain, config.extent, device, config.prediction_batch_size
            )
            grid = exp.scalar_grid(config, grid_prediction)
            grid.save(run_dir / "scalar_field.vti")
            terminal_rmse, terminal_by_label = exp.terminal_geometry_rmse(grid, holdout_interface)
            exp.save_sample_points(run_dir / "cross_fault_samples.vtp", pairs, sample_values)
            row = {
                "seed": int(seed),
                "variant": variant,
                "input_dim": int(domain.shape[1]),
                 "eikonal_points": EIKONAL_SAMPLES,
                "eikonal_sha256": eikonal_hash,
                "terminal_holdout_n": int(len(holdout_interface)),
                "terminal_interface_rmse": float(terminal_rmse),
                "best_total_loss": float(best["total_loss"]),
                "best_interface_loss": float(best["interface_loss"]),
                "best_orientation_loss": float(best["orientation_loss"]),
                "best_eikonal_loss": float(best["eikonal_loss"]),
                "training_seconds": float(best["training_seconds"]),
                "parameter_count": int(best["parameter_count"]),
                "initialization_sha256": best["initialization_sha256"],
                **metric_values,
            }
            (run_dir / "metrics.json").write_text(
                json.dumps({"metrics": row, "terminal_rmse_by_label": terminal_by_label, "feature_metadata": feature_metadata, "eikonal_metadata": eikonal_metadata}, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    if len(frame) != 20:
        raise RuntimeError(f"expected exactly 20 formal cells, got {len(frame)}")
    summary = write_summary(rows)
    paired = write_paired_comparisons(frame)
    plot_metrics(summary, paired)
    decision = evaluate_go(summary, paired, geometry)
    decision["geometry_diagnostic"] = geometry
    (RESULTS_ROOT / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_ROOT / "numeric_comparison.md").write_text(
        "# CAGEO-R1B Numeric Comparison\n\n"
        + dataframe_text(summary)
        + "\n\n## Paired comparisons\n\n"
        + dataframe_text(paired, index=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
