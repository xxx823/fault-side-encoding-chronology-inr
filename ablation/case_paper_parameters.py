"""Paper-parameter ablation-case runner.

This wrapper keeps the original ablation data preparation and evaluation protocol,
but replaces the training configuration with the values reported in Table 1 of
the manuscript.  It also enables the fault above/below constraint and adds an
Eikonal term to the stratigraphic-field loss.

The Eikonal term is evaluated on the existing interface/orientation query
points because the manuscript does not specify a separate N_E sampling rule.
The run metadata records this limitation explicitly.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pyvista as pv
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ablation.case_ablation as exp  # noqa: E402


def paper_config() -> exp.ExperimentConfig:
    """Return the known manuscript settings for the truncated two-fault case."""

    config = replace(
        exp.ExperimentConfig(),
        # Table 1: fault and stratigraphic query grids.
        fault_resolution=(60, 30, 30),
        grid_resolution=(100, 100, 100),
        # Table 1: both networks are 2 x 256.
        fault_hidden_dim=256,
        fault_hidden_layers=2,
        strat_hidden_dim=256,
        strat_hidden_layers=2,
        # Table 1: Softplus beta values.
        fault_beta=(10.0, 10.0),
        strat_beta=1.0,
        # Table 1: epochs and learning rate.
        fault_epochs=500,
        fault_lr=0.001,
        strat_epochs=1000,
        strat_lr=0.001,
        # Manuscript loss weights: interface=1, attitude=0.1, Eikonal=0.01.
        orientation_weight=0.1,
    )
    # The manuscript reports physical sampling distances rather than grid-cell
    # counts. Convert them to the legacy case fields after setting the 100^3
    # grid, so the actual run uses exactly 5 m, 10 m, and 80--320 m.
    minimum_spacing = float(exp.cell_spacing(config).min())
    return replace(
        config,
        terminal_window_cells=80.0 / minimum_spacing,
        cross_fault_epsilon_cells=5.0 / minimum_spacing,
        boundary_exclusion_epsilon=2.0,
    )


def eikonal_loss(model: torch.nn.Module, x: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Mean squared deviation of the spatial gradient norm from one."""

    gradient = torch.autograd.grad(
        outputs=prediction,
        inputs=x,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0][:, :3]
    return ((torch.linalg.vector_norm(gradient, dim=1) - 1.0) ** 2).mean()


def train_stratigraphic_model_paper(
    config: exp.ExperimentConfig,
    interface: np.ndarray,
    orientation: np.ndarray,
    seed: int,
    device: torch.device,
):
    """Train the stratigraphic network with Adam and the paper loss weights."""

    train_x = interface[:, 1:].astype(np.float64)
    train_y = interface[:, 0].astype(np.float64)
    orientation_x = np.delete(orientation.copy(), [3, 4, 5], axis=1).astype(np.float64)
    orientation_y = orientation[:, 3:6].astype(np.float64)

    normalized_interface = exp.normalize_inputs(train_x, config.extent)
    normalized_orientation = exp.normalize_inputs(orientation_x, config.extent)
    combined = np.vstack((normalized_interface, normalized_orientation))

    x_tensor = torch.tensor(combined, dtype=torch.float32, device=device, requires_grad=True)
    y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
    dy_tensor = torch.tensor(orientation_y, dtype=torch.float32, device=device)
    n_interface = normalized_interface.shape[0]
    n_orientation = normalized_orientation.shape[0]

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
    initialization_hash = exp.hashlib.sha256()
    for name, value in model.state_dict().items():
        initialization_hash.update(name.encode("utf-8"))
        initialization_hash.update(value.detach().cpu().numpy().tobytes())

    optimizer = torch.optim.Adam(model.parameters(), lr=config.strat_lr)
    best_loss = float("inf")
    best_state = None
    best_terms: Dict[str, object] = {}
    history: List[Dict[str, float]] = []
    started = time.time()

    for epoch in range(config.strat_epochs):
        optimizer.zero_grad(set_to_none=True)
        x_tensor.grad = None
        prediction = model(x_tensor)
        interface_term = exp.lossf.loss_intf_sum(prediction[:n_interface].squeeze(), y_tensor)
        eikonal_term = eikonal_loss(model, x_tensor, prediction)
        orientation_term = exp.lossf.loss_grad_with_fault_features(
            x_tensor, prediction, dy_tensor, n_orientation
        )
        loss = interface_term + config.orientation_weight * orientation_term + 0.01 * eikonal_term
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at seed={seed}, epoch={epoch + 1}")

        record = {
            "epoch": epoch + 1,
            "total_loss": float(loss.detach().cpu()),
            "interface_loss": float(interface_term.detach().cpu()),
            "orientation_loss": float(orientation_term.detach().cpu()),
            "eikonal_loss": float(eikonal_term.detach().cpu()),
        }
        history.append(record)
        if record["total_loss"] < best_loss:
            best_loss = record["total_loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_terms = record.copy()

        loss.backward()
        optimizer.step()

    if best_state is None:
        raise RuntimeError("No finite model checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()
    best_terms["training_seconds"] = time.time() - started
    best_terms["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    best_terms["initialization_sha256"] = initialization_hash.hexdigest()
    return model, history, best_terms


def run(args: argparse.Namespace) -> Path:
    config = paper_config()
    if args.seeds:
        config = replace(config, seeds=tuple(args.seeds))
    if args.variants:
        config = replace(config, variants=tuple(args.variants))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = exp.resolve_device(args.device)

    # The manuscript states that the fault-stage spatial relation constraint is enabled.
    original_fault_builder = exp.models.fault_ConcatMLP

    def paper_fault_builder(*builder_args, **builder_kwargs):
        builder_kwargs["above_below"] = True
        return original_fault_builder(*builder_args, **builder_kwargs)

    exp.models.fault_ConcatMLP = paper_fault_builder
    surface, orientation = exp.load_case(config)
    meshes = exp.train_or_load_faults(
        config,
        surface,
        orientation,
        output_dir,
        device,
        rebuild=args.rebuild_faults,
    )

    inferred_side, side_metadata = exp.infer_retained_side(config, surface, meshes)
    retained_side = args.retained_younger_side if args.retained_younger_side is not None else inferred_side
    rules = (
        exp.TemporalRule(
            older_fault=config.older_fault,
            younger_fault=config.younger_fault,
            retained_younger_side=retained_side,
        ),
    )
    domain_raw, interface_raw, orientation_raw = exp.feature_encode_all(
        config, meshes, surface, orientation
    )
    holdout_mask, holdout_metadata = exp.choose_terminal_holdout(config, interface_raw, meshes)
    train_interface_raw = interface_raw[~holdout_mask]
    holdout_interface_raw = interface_raw[holdout_mask]
    pairs = exp.prepare_cross_fault_pairs(config, meshes, retained_side)

    metadata = {
        "config": asdict(config),
        "device": str(device),
        "requested_device": args.device,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "paper_config": True,
        "fault_above_below_constraint": True,
        "eikonal_weight": 0.01,
        "eikonal_sampling": "existing interface and orientation query points; separate N_E is not specified in the manuscript",
        "retained_younger_side": retained_side,
        "retained_side_inference": side_metadata,
        "terminal_holdout": holdout_metadata,
        "cross_fault_epsilon_physical_units": float(pairs["epsilon"][0]),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "numpy_version": np.__version__,
        "pyvista_version": pv.__version__,
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(output_dir / "terminal_holdout_mask.npy", holdout_mask)

    all_rows = []
    for seed in config.seeds:
        for variant in config.variants:
            print(f"\n=== seed={seed} variant={variant} ===", flush=True)
            run_dir = output_dir / f"seed_{seed}" / variant
            run_dir.mkdir(parents=True, exist_ok=True)
            domain, interface, orientation_variant, feature_metadata = exp.build_variant(
                domain_raw,
                train_interface_raw,
                orientation_raw,
                variant,
                config.fault_names,
                rules,
            )
            holdout_domain = np.column_stack((holdout_interface_raw[:, 1:4], holdout_interface_raw[:, 4:]))
            holdout_x, _ = exp.transform_domain(holdout_domain, variant, config.fault_names, rules)
            holdout_interface = np.column_stack((holdout_interface_raw[:, 0], holdout_x))

            model, history, best = train_stratigraphic_model_paper(
                config, interface, orientation_variant, seed, device
            )
            exp.write_history(run_dir / "loss_history.csv", history)
            if args.save_checkpoints:
                torch.save(model.state_dict(), run_dir / "model.pt")

            grid_prediction = exp.predict_in_batches(model, domain, config.extent, device, config.prediction_batch_size)
            grid = exp.scalar_grid(config, grid_prediction)
            grid.save(run_dir / "scalar_field.vti")
            interface_mesh = grid.contour(isosurfaces=np.unique(interface_raw[:, 0]).tolist(), scalars="scalar")
            interface_mesh.save(run_dir / "interfaces.vtp")
            cross_metrics, sample_values = exp.cross_fault_metric(config, model, variant, meshes, rules, pairs, device)
            terminal_rmse, terminal_by_label = exp.terminal_geometry_rmse(grid, holdout_interface)
            exp.save_sample_points(run_dir / "cross_fault_samples.vtp", pairs, sample_values)

            row = {
                "seed": seed,
                "variant": variant,
                "input_dim": int(interface.shape[1] - 1),
                "best_epoch": int(best["epoch"]),
                "best_total_loss": best["total_loss"],
                "best_interface_loss": best["interface_loss"],
                "best_orientation_loss": best["orientation_loss"],
                "best_eikonal_loss": best["eikonal_loss"],
                "training_seconds": best["training_seconds"],
                "parameter_count": int(best["parameter_count"]),
                "initialization_sha256": best["initialization_sha256"],
                "terminal_interface_rmse": terminal_rmse,
                "terminal_holdout_n": int(len(holdout_interface)),
                **cross_metrics,
            }
            all_rows.append(row)
            (run_dir / "metrics.json").write_text(
                json.dumps({"metrics": row, "terminal_rmse_by_label": terminal_by_label, "feature_metadata": feature_metadata}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            del model, grid
            if device.type == "cuda":
                torch.cuda.empty_cache()

    exp.summarize(all_rows, output_dir)
    print(f"\nFinished. Summary: {output_dir / 'metrics_summary.csv'}", flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "ablation_outputs" / "paper_case")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rebuild-faults", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--variants", nargs="+", choices=exp.VALID_VARIANTS)
    parser.add_argument("--retained-younger-side", type=int, choices=(0, 1))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
