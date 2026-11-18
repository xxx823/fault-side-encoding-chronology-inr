# Manuscript and code alignment record

This repository publishes the redistributable controlled two-fault R1B
experiment. It does not contain the confidential four-fault engineering data
or the full R1C engineering runner.

## R1B controlled experiment

The paper's controlled two-fault results are the frozen 20-run R1B results in
[`paper_results/r1b`](../paper_results/r1b/README.md), not the earlier
`case5_paper_config_*` summaries found in the working project. The frozen
values are reported with five-seed mean and sample standard deviation.

The R1B run used the public `surface.csv` and `orientation.csv` plus two fixed
raw fault meshes. The feature variants share a five-channel input and the same
network parameter count. `coord` and `coord_zero` therefore produce identical
results for the frozen run. The effective-domain variant gates the older
fault-side feature using the prescribed younger-fault side. Its two-fault
result demonstrates the controlled invalid-domain response; it does not
establish general multi-fault topology recovery.

The portable entry point is `ablation/r1b_experiment.py`. Its training and
evaluation logic follows the frozen R1B implementation, with repository-local
asset paths and explicit `--run-formal` opt-in. It does not change the frozen
CSV results in `paper_results/r1b/`.

## Four-fault engineering experiment (R1C)

R1C is a separate protocol. It used 220 processed stratigraphic interface
points, five XY spatial folds, three seeds, and three neural variants, for 45
formal training cells. The input tables have no borehole identifiers, so the
results must be described as spatial holdout validation of processed
interface observations rather than borehole validation. The four-fault meshes
were fixed interpreted priors; the experiment did not retrain fault networks
inside each fold. The supplied stratigraphic orientation vectors were used
directly; PCA was not used.

Across 15 fold-seed neural cells, geometry MAE was 121.905 ± 22.249 m for
`coord`, 106.091 ± 26.511 m for `global_side`, and 102.777 ± 31.175 m for
`effective_domain`. The pooled point-level MAE values were 121.916 m,
105.369 m, and 100.458 m, respectively; the repeated point predictions are
not independent replicates. LoopStructural 1.6.27 FDI had a five-fold mean
geometry MAE of 30.094 ± 23.743 m. FDI did not use fault displacement or slip
direction priors and is a reference baseline, not an equivalent
fault-kinematic comparison.

These engineering figures are listed to keep the public repository's scope
consistent with the revised manuscript. Engineering inputs, fold-level
predictions, models, and execution scripts remain outside this repository due
to the project's data restrictions. No code here can reproduce the R1C
engineering numbers without those private artifacts.

## Superseded claims

The manuscript copy supplied for this revision still contained the earlier
Table 2 values and a 57-borehole 80/20-style floor-elevation result. Those
values are not the frozen R1B/R1C results and are not reproduced by this
repository. The D-drive `case5_paper_config_exact`, `case5_paper_config_full`,
and `case5_formal_gpu` output directories are historical runs with different
inputs or configurations; they are not substituted for the R1B/R1C evidence.

The old `ablation/case_paper_parameters.py` entry point is retained for
historical comparison only. It reconstructs fault meshes and does not use the
corrected physical-space orientation-gradient computation in the frozen R1B
protocol. The R1B README and runner identify it as legacy.
