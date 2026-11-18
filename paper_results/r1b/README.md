# Frozen R1B controlled results

These files are the official 20-cell R1B result table copied from the frozen
experiment archive. The public repository includes the per-cell CSV and its
five-seed summary. A fresh run of the repository-local portable runner matched
the frozen deterministic metrics across all 20 cells; see
[`REPRODUCTION_CHECK.md`](REPRODUCTION_CHECK.md). The fresh run wrote outputs to
an ignored run directory and did not overwrite the frozen tables.

## Protocol

- Two interpreted fault meshes, held fixed for all four variants.
- Seeds: `0, 1, 2, 3, 4`.
- Variants: `coord`, `coord_zero`, `global_side`, `effective_domain`.
- Total: 20 stratigraphic-network trainings; fault-network trainings in R1B: 0.
- Stratigraphic network: two hidden layers × 256 units, Softplus β = 1,
  Adam, learning rate 0.001, 1000 epochs.
- Orientation weight: 0.1. The normalized-coordinate gradient is transformed
  into physical space with the coordinate-normalization Jacobian.
- Eikonal: weight 0.01 in normalized-coordinate space, 4096 fixed points per
  seed, shared across the four variants.
- Interface evaluation: 15 held-out points near the interpreted termination.
- Scalar response: 479 retained-side pairs and 682 non-retained-side pairs.

`delta_effective_mean` and `delta_invalid_mean` are mean absolute scalar
differences across the paired samples. They are not displacement or throw
estimates. `terminal_interface_rmse` is the point-to-predicted-isosurface
distance RMSE in metres.

## Official summary

| Variant | Valid response, mean ± SD | Invalid response, mean ± SD | Terminal RMSE (m), mean ± SD |
| --- | ---: | ---: | ---: |
| `coord` | 0.005431 ± 0.000077 | 0.003201 ± 0.000391 | 26.072 ± 1.718 |
| `coord_zero` | 0.005431 ± 0.000077 | 0.003201 ± 0.000391 | 26.072 ± 1.718 |
| `global_side` | 0.405550 ± 0.013729 | 0.341252 ± 0.008452 | 29.898 ± 0.893 |
| `effective_domain` | 0.285234 ± 0.027746 | 0.002309 ± 0.000339 | 22.262 ± 2.340 |

Across paired seeds, the median invalid-response reduction for
`effective_domain` relative to `global_side` is 99.36%. The result supports
feature gating in this controlled two-fault setting under the prescribed
chronology and retained-side rule.

## Provenance

The input hashes below are the ones recorded by the frozen R1B run and are
validated by `python -m ablation.r1b_experiment --preflight-only`:

| Input | SHA-256 |
| --- | --- |
| `data/case1/surface.csv` | `BF96D0D8139683CE8ADF6D43F259957459EBA258E04689497085B503C2ECE337` |
| `data/case1/orientation.csv` | `C7204DA2BE30626ED61EEC0A958072F72056A8A911BD903BD9998FA978CC3987` |
| `fault1_raw.vtp` | `B934827188A8BABBEE9F45064CD4DFD115DFDF62F3E1771283326DA380BFDB57` |
| `fault2_raw.vtp` | `1DA1223AC440B4FA96C77F0A67BA4B3A51FD4F6EE87FE5E2956B870758FE6026` |

Frozen environment: Python 3.11.17, PyTorch 2.12.1+cu130, CUDA 13.0, NVIDIA
GeForce RTX 5060 Laptop GPU, NumPy 1.26.4, pandas 2.2.3, PyVista 0.46.4,
Matplotlib 3.11.2, VTK 9.5.2.

Historical source hashes recorded for the frozen results:

```text
case_ablation.py   071E3FC41C5F14E0119836611605D8D28352BE1C49F085A8CB9A9C6878BBA768
feature_variants.py 03D8C03992F66A2E155F5C367598EBBA69FC4554F265EC0F0F46862BCF36FABB
mlps.py            2A16F9EE271692778E681571964F7CD0C0A49FAF82F535DE7D655F0C8932B91E
lossf.py           AABB1091CC0C678A256A75E67D6C3792E800B13A3A75AD6F1D49001E4B23916B
utils.py           222F34884377C40BE4AEC4E7CD6AFD1B865E582F6CD3384DC06025F6DEFD6772
r1b_core.py        5CA8947F166D259526B08AD66E7F39EAD15547526477A1565E615BB92362BEA5
r1b_experiment.py  A96F4747E403350B5FE77E65C194348914836A3C4D40B989759325EC9E3E29C6
```

The repository's portable runner preserves the R1B training and evaluation
logic while replacing machine-specific D-drive paths with repository-relative
inputs and requiring `--run-formal` to start training. Its current file hashes
therefore differ from the historical run hashes above. The recorded fresh
reproduction matched the frozen deterministic metrics; hardware-dependent
wall-clock durations are not expected to match.

The legacy project directories `case5_paper_config_exact`,
`case5_paper_config_full`, and `case5_formal_gpu` contain earlier experiments
with different inputs/configurations. Their summary values do not match the
frozen R1B table and are not used here.

## CSV hashes

```text
metrics_all_runs.csv  FDC0C9AB85108B7BB4C874BCA7B7C307A3A1C501464EC12847B546958E4B1EC0
metrics_summary.csv   8165505D2E98A470F5E1511BDEB645FD87300A59265C6560AAB4FB1110485566
```
