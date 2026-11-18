# Fault-Side Encoding and Chronology-Constrained INR

**Reproducible code for fault-conditioned implicit neural 3D geological modeling with chronology-constrained valid-domain masks.**

**Associated manuscript:** *Implicit Neural Representation for 3D Geological Modeling Using Fault Side Encoding and Chronology Constraints*<br>
**Authors:** Yingxian Chen, Jianzheng Xin, and Pu Wang<br>
**Corresponding author:** Jianzheng Xin (xinj_zheng@163.com)

This repository accompanies the manuscript and provides the controlled two-fault truncation experiment used to evaluate fault-side encoding and chronology-conditioned valid-domain constraints. It contains the experimental inputs, neural modeling code, and a reproducible four-configuration ablation workflow. The confidential open-pit coal mine dataset is **not** included.

## Overview

Implicit neural representations (INRs) can model stratigraphic scalar fields, but an unconstrained fault-side feature may produce an artificial discontinuity beyond the interpreted termination of a truncated fault. This study uses **prescribed fault chronology and truncation relationships** to determine where each fault feature is active.

The main steps are:

1. Represent interpreted fault geometry using implicit fault fields and derive binary fault-side features.
2. Construct valid-domain masks from the retained-side conditions of younger, controlling faults.
3. Combine the masked fault-side features with normalized spatial coordinates.
4. Train a stratigraphic INR using interface, orientation, and Eikonal regularization constraints.
5. Query the predicted scalar field and extract 3D stratigraphic interfaces.

**Important:** Fault chronology and truncation relationships are supplied as geological interpretations; they are not automatically inferred from observations by this implementation.

## Workflow diagram

![Workflow of the chronology-conditioned fault-side encoding and 3D geological modeling framework](images/workflow.png)

*Figure. Overview of the proposed framework. Geological interface, orientation, and fault observations are used to construct fault representations. Binary fault-side states are combined with chronology-based valid-domain constraints. The resulting features condition an implicit neural stratigraphic field, from which 3D geological interfaces are extracted.*

## Repository contents

```text
fault-side-encoding-chronology-inr/
|-- ablation/
|   |-- case_ablation.py
|   |-- feature_variants.py
|   |-- r1b_bundle.py
|   |-- r1b_core.py
|   `-- r1b_experiment.py
|-- data/case1/
|   |-- surface.csv
|   |-- orientation.csv
|   `-- fault_meshes/
|       |-- fault1_raw.vtp
|       `-- fault2_raw.vtp
|-- feature_encoding/
|   `-- fault_side_encoding.py
|-- losses.py
|-- networks.py
|-- geological_models.py
|-- geological_utils.py
|-- requirements.txt
|-- docs/EXPERIMENT_SCOPE.md
|-- paper_results/r1b/
|   |-- RESULTS_AND_PROVENANCE.md
|   |-- REPRODUCTION_CHECK.md
|   |-- metrics_all_runs.csv
|   `-- metrics_summary.csv
|-- tests/
|-- images/workflow.png
|-- LICENSE
`-- README.md
```

Only the most relevant files are shown above. Some existing modules and scripts have been omitted for readability.

## Software environment

The frozen R1B runs used Python 3.11.17, PyTorch 2.12.1+cu130, CUDA 13.0, and an NVIDIA GeForce RTX 5060 Laptop GPU. The repository pins the following Python packages in `requirements.txt`:

- `torch==2.12.1`
- `numpy==1.26.4`
- `pandas==2.2.3`
- `matplotlib==3.11.2`
- `pyvista==0.46.4`
- `vtk==9.5.2`

From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU execution, ensure that the installed PyTorch build supports the local CUDA-capable GPU. Different PyTorch/CUDA installations and hardware may lead to different runtimes or small numerical differences.

## Reproducing the controlled ablation

The controlled experiment entry point is `ablation.r1b_experiment`. It uses the frozen raw fault meshes as fixed inputs and trains the stratigraphic network only; the historical paper-parameter runner is excluded from the public code path.

### 1. Check the provided inputs (no training)

```bash
python -m ablation.r1b_experiment --preflight-only
```

The R1B runner checks the required input assets, including the two CSV files and the frozen raw fault meshes. Input hashes are recorded in `ablation/r1b_bundle.py`. If the check fails, restore the original repository assets before attempting to reproduce the experiment.

### 2. Run the full controlled experiment

```bash
python -m ablation.r1b_experiment --run-formal --device cuda
```

This **explicitly launches training** for all four ablation configurations and five random seeds (20 training runs). On a machine without a supported CUDA installation, `--device auto` can be used instead, with possible performance and numerical differences.

The four configurations are:

1. Spatial coordinates only (`coord`).
2. Spatial coordinates with zero-valued feature placeholders (`coord_zero`).
3. Global fault-side encoding (`global_side`).
4. Fault-side encoding with chronology-constrained valid-domain gating (`effective_domain`).

By default, results are written to `ablation_outputs/r1b_formal/`. To use a separate output directory, add `--output-dir <path>` to the command.

### 3. Regenerate summaries from an existing completed run

```bash
python -m ablation.r1b_experiment --finalize-only
```

This command re-generates the summary from previously saved outputs and **does not retrain** the models. It requires a complete set of 20 run metrics in the selected output directory.

## Outputs and visualization

The controlled experiment produces per-run artifacts such as:

- `scalar_field.vti` - a volumetric scalar field that can be opened in ParaView;
- `cross_fault_samples.vtp` - paired samples used for cross-fault evaluation;
- `model.pt` - the saved PyTorch model parameters;
- `loss_history.csv` - the optimization history;
- `metrics.json` - metrics for a single seed and configuration.

The output root also contains experiment configuration, integrity/geometry diagnostics, aggregate metrics, figures, and a numerical comparison report (for example, `numeric_comparison.md` and `decision.json`).

## Frozen R1B results

The controlled truncated-fault ablation compares the magnitude of the scalar jump inside the active fault domain, the residual response beyond the termination, and the near-termination interface RMSE. Values below are **mean +/- sample standard deviation across five random seeds**, from the frozen R1B experiment:

| Configuration | Valid-region scalar difference | Invalid-region residual | Interface RMSE (m) |
| --- | ---: | ---: | ---: |
| Coordinates only | 0.005431 +/- 0.000077 | 0.003201 +/- 0.000391 | 26.072 +/- 1.718 |
| Zero-vector placeholder | 0.005431 +/- 0.000077 | 0.003201 +/- 0.000391 | 26.072 +/- 1.718 |
| Global fault-side encoding | 0.405550 +/- 0.013729 | 0.341252 +/- 0.008452 | 29.898 +/- 0.893 |
| Fault-side encoding + valid-domain gating | 0.285234 +/- 0.027746 | 0.002309 +/- 0.000339 | 22.262 +/- 2.340 |

Across the five seeds, valid-domain gating reduced the invalid-region response relative to global fault-side encoding by a median of **99.36%**, while preserving a cross-fault response in the valid region.

These values are the frozen R1B summary in [`metrics_summary.csv`](paper_results/r1b/metrics_summary.csv); individual seed results are available in [`metrics_all_runs.csv`](paper_results/r1b/metrics_all_runs.csv). Detailed provenance and reproduction notes are in [`RESULTS_AND_PROVENANCE.md`](paper_results/r1b/RESULTS_AND_PROVENANCE.md), [`REPRODUCTION_CHECK.md`](paper_results/r1b/REPRODUCTION_CHECK.md), and [`EXPERIMENT_SCOPE.md`](docs/EXPERIMENT_SCOPE.md).

## Data availability and scope

- **Included:** redistributable input data and raw fault meshes for the controlled two-fault truncation case.
- **Not included:** borehole, interpretation, and other geological records from the engineering case study in an open-pit coal mine, because of project confidentiality restrictions.
- The accompanying manuscript also describes additional controlled structural scenarios and an engineering application; this repository is focused on the reproducible truncation ablation and should not be interpreted as a complete public release of every study dataset or experiment.

## Related work

- Hillier, M., et al. (2023). *GeoINR 1.0: An implicit neural network approach to three-dimensional geological modelling.* [https://doi.org/10.5194/gmd-16-6987-2023](https://doi.org/10.5194/gmd-16-6987-2023)
- Gao, K., and Wellmann, F. (2025). *Fault representation in structural modelling with implicit neural representations.* [https://doi.org/10.1016/j.cageo.2025.105911](https://doi.org/10.1016/j.cageo.2025.105911)

## Citation

If you use this code, please reference the associated manuscript and link to this repository:

> Chen, Y., Xin, J., and Wang, P. *Implicit Neural Representation for 3D Geological Modeling Using Fault Side Encoding and Chronology Constraints*. Manuscript; publication details to be updated when available.

Repository: https://github.com/xxx823/fault-side-encoding-chronology-inr

## License

This repository is distributed under the [MIT License](LICENSE). See the license file and any notices in the source code for applicable terms.
