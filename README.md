# Fault-Side Encoding and Chronology-Conditioned INR

## Associated manuscript

**Title:** Implicit Neural Representation for 3D Geological Modeling Using Fault Side Encoding and Chronology Constraints
**Authors:** Yingxian Chen, Jianzheng Xin, Pu Wang

This repository contains the public two-fault controlled experiment associated
with the manuscript. The reported R1B experiment uses fixed interpreted fault
meshes and trains only the stratigraphic INR. The four variants share one
network width and compare coordinate-only inputs, global side features, and
chronology-conditioned feature gating.

The engineering case is separate. Its processed observations and models are
not included because the project data are confidential. It is reported as a
five-fold spatial holdout on 220 processed interface points, not as a
borehole-level validation.

## Frozen R1B result

The official reference values below come from the 20 frozen R1B runs (four
variants × five seeds). Values are mean ± sample standard deviation across the
five seeds. Scalar response is the mean absolute difference across paired
points on either side of the older fault. The terminal interface RMSE is the
three-dimensional point-to-predicted-isosurface distance for 15 held-out
interface points.

| Variant | Valid-domain response | Invalid-domain response | Terminal interface RMSE (m) |
| --- | ---: | ---: | ---: |
| `coord` | 0.005431 ± 0.000077 | 0.003201 ± 0.000391 | 26.072 ± 1.718 |
| `coord_zero` | 0.005431 ± 0.000077 | 0.003201 ± 0.000391 | 26.072 ± 1.718 |
| `global_side` | 0.405550 ± 0.013729 | 0.341252 ± 0.008452 | 29.898 ± 0.893 |
| `effective_domain` | 0.285234 ± 0.027746 | 0.002309 ± 0.000339 | 22.262 ± 2.340 |

The paired invalid-response reduction for `effective_domain` relative to
`global_side` has a five-seed median of 99.36%. This is a controlled
two-fault mechanism result under fixed geometry and a prescribed retained-side
rule; it is not a physical fault-displacement estimate or a claim of general
topology recovery. Per-seed results and their provenance are in
[`paper_results/r1b`](paper_results/r1b/README.md).

## R1B data and code

The experiment reads four hash-checked inputs:

- `data/case1/surface.csv`
- `data/case1/orientation.csv`
- `data/case1/fault_meshes/fault1_raw.vtp`
- `data/case1/fault_meshes/fault2_raw.vtp`

The feature and training implementation is in `ablation/`. The corrected
orientation loss maps normalized-coordinate gradients into physical space with
the normalization Jacobian. The Eikonal term is evaluated in normalized
coordinate space on 4096 fixed points per seed, shared across the four
variants. Fault-side features are derived from the fixed input meshes; the R1B
run does not train a fault network.

Run a read-only preflight to validate the inputs and inspect the locked run
configuration:

```bash
python -m ablation.r1b_experiment --preflight-only
```

To intentionally run all 20 neural trainings, pass the explicit flag below.
Results are written to the ignored `ablation_outputs/r1b_formal` directory.

```bash
python -m ablation.r1b_experiment --run-formal --device cuda
```

Use `--device auto` for a CPU fallback. Numerical results can vary across
hardware and library builds. `--finalize-only` aggregates an already completed
run and does not train.

The existing `ablation/case_paper_parameters.py` is retained as a legacy
runner for earlier experiments. It is not the source of the frozen R1B values
above and should not be used to reproduce the current manuscript tables.

## Recorded environment

The frozen R1B and R1C runs used Python 3.11.17, PyTorch 2.12.1+cu130, CUDA
13.0, NumPy 1.26.4, pandas 2.2.3, Matplotlib 3.11.2, PyVista 0.46.4, and VTK
9.5.2. Install the matching PyTorch CUDA wheel first, then the other pinned
packages:

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-paper.txt
```

## R1C engineering comparison scope

R1C used a fixed four-fault prior, five XY spatial folds, three seeds, and
three neural variants (45 training cells). The held-out targets were processed
interface observations; no borehole identifiers were available. Neural
geometry MAE averaged across the 15 fold-seed cells was 121.905 ± 22.249 m for
`coord`, 106.091 ± 26.511 m for `global_side`, and 102.777 ± 31.175 m for
`effective_domain`. The LoopStructural 1.6.27 FDI reference produced a
five-fold geometry MAE of 30.094 ± 23.743 m. The FDI reference did not receive
fault displacement or slip-direction priors, so it is not an equivalent
fault-kinematic comparison. Engineering data, models, and fold-level files
remain outside this public repository.

## Citation and contact

Please cite the associated manuscript when using this code. The repository
contains the redistributable controlled-case inputs; it does not contain the
confidential engineering data.
