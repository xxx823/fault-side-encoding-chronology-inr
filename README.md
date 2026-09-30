# Fault-Side Encoding and Chronology-Constrained INR

## Associated manuscript

**Title:** Implicit Neural Representation for 3D Geological Modeling Using Fault Side Encoding and Chronology Constraints

**Authors:** Yingxian Chen, Jianzheng Xin, Pu Wang

This repository contains the reproducible implementation of the controlled
truncation ablation case associated with the manuscript on implicit neural
representation for 3D geological modelling.

The main mechanism is:

```text
fault surface -> signed side state -> chronology mask -> masked combination code -> stratigraphic INR
```

The repository is intentionally small. It includes the ablation case entry
point, the model and loss modules it imports, and the two CSV files required by
the controlled two-fault example. Engineering-area data are not included.

## Reported experimental environment

The experiments reported in the manuscript were performed on Windows 11 using
an Intel Core i7-13790F processor and an NVIDIA GeForce RTX 5060 Ti GPU. The
software environment was Python 3.12, PyTorch 2.6.0, and CUDA 12.8.

The paper-parameter runner is configured to use CUDA for strict reproduction
of the reported experiments. CPU execution and other CUDA devices remain
possible for portability, but their runtime and numerical results may differ.

## Requirements

- Windows 11
- Python 3.12
- PyTorch 2.6.0
- CUDA 12.8
- NumPy, pandas, Matplotlib, PyVista, and VTK

Install the pinned paper environment with:

```bash
python -m pip install -r requirements-paper.txt
```

For a CUDA installation, install the PyTorch wheel appropriate for the local
GPU first, then install the remaining packages from the requirements file.

## Input data

The controlled two-fault ablation example uses two input files in `data/case1/`:

- `surface.csv`: stratigraphic interface points and fault-surface points with spatial coordinates and geological labels.
- `orientation.csv`: stratigraphic and fault orientation samples with spatial coordinates and direction vectors.

These files contain only the controlled experimental data used for the ablation example and do not include data from the confidential engineering case.

## Quick smoke test

Run one seed and one configuration to verify the installation:

```bash
python ablation/case_paper_parameters.py --device cuda --seeds 0 --variants effective_domain
```

This command writes the outputs to `ablation_outputs/paper_case`.

## Full ablation case

The paper-parameter run uses four configurations and five random seeds. Use
`--device auto` only when intentionally running on a different machine and
accepting a CPU fallback:

```bash
python ablation/case_paper_parameters.py --device cuda --rebuild-faults
```

The four configurations are:

1. spatial coordinates only;
2. coordinates with zero placeholders;
3. coordinates with global fault-side encoding;
4. coordinates with fault-side encoding and chronology-constrained valid-domain masking.

The reported protocol uses 2 hidden layers with 256 units, Adam with a
learning rate of 0.001, fault Softplus beta 10, stratigraphic Softplus beta 1,
500 fault epochs, 1000 stratigraphic epochs, attitude weight 0.1, and Eikonal
weight 0.01. The evaluation uses 5 m cross-fault offsets, a 10 m boundary
exclusion, a fixed 80--320 m terminal-region window, and 15 fixed holdout
interface points.

## Outputs

Each seed/configuration directory contains:

- `scalar_field.vti`: queried implicit scalar field;
- `interfaces.vtp`: extracted stratigraphic interfaces;
- `cross_fault_samples.vtp`: paired evaluation samples;
- `loss_history.csv`: training history;
- `metrics.json`: per-run metrics.

The output root also contains `metrics_summary.csv`, `metrics_all_runs.csv`,
`run_config.json`, and the ablation summary plot.

## Expected behaviour

The four ablation configurations are expected to show distinct structural responses.

Global fault-side encoding should produce a clear scalar difference across the fault within its valid region. After the chronology-constrained valid-domain mask is introduced, the residual scalar response beyond the fault termination should be strongly suppressed, while the interface reconstruction near the termination should improve.

For the full five-seed experiment, the results should reproduce the same trends as those reported in the manuscript.

The reference results reported in the manuscript are:

| Configuration | Valid-region scalar difference | Invalid-region residual | Interface RMSE (m) |
| --- | ---: | ---: | ---: |
| Spatial coordinates only | 0.00542 ± 0.00010 | 0.00195 ± 0.00010 | 25.57 ± 1.00 |
| Zero-vector placeholder | 0.00553 ± 0.00012 | 0.00196 ± 0.00016 | 26.01 ± 1.35 |
| Global fault-side encoding | 0.44571 ± 0.01194 | 0.35995 ± 0.00218 | 29.31 ± 0.35 |
| Fault-side encoding + valid-domain constraint | 0.29998 ± 0.01216 | 0.00118 ± 0.00005 | 24.43 ± 0.86 |

## Reproducibility notes

The manuscript does not specify a separate Eikonal sample count `N_E`. The
paper-parameter runner therefore evaluates the Eikonal term on the existing
interface and orientation query points and records this choice in
`run_config.json`. The fault above/below option is enabled in the runner; the
included data currently produce a zero numerical above/below term.

The included CSV files are the redistributable input data used for the
controlled two-fault ablation experiment described in the manuscript. They do
not contain data from the confidential engineering case.

## Citation and contact

Please cite the associated manuscript when using this code. Include the
repository URL and author contact in the final citation information.
