# R1B reproduction check

The repository runner was executed on the bundled two-fault controlled case
with the registered five seeds and four variants (20 training units). This is
a fresh reproduction run, separate from the frozen reference tables. The
run used Python 3.11.17, PyTorch 2.12.1+cu130, CUDA 13.0, and an NVIDIA RTX
5060 Laptop GPU.

All 20 rerun rows matched the frozen R1B record exactly for the deterministic
reported quantities: effective-domain response, invalid-domain response,
terminal-interface RMSE, total loss, interface loss, orientation loss, and
Eikonal loss. Input hashes matched the hashes recorded by the frozen
experiment. Wall-clock training times varied and are not treated as a model
result.

The fresh run was written to the ignored directory
`ablation_outputs/r1b_reproduction_2026-11-18/`; it did not overwrite the
frozen result table in this repository. The reproducibility check did not
include the confidential four-fault engineering case.
