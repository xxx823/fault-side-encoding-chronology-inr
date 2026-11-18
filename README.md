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



## License

This repository is distributed under the [MIT License](LICENSE). See the license file and any notices in the source code for applicable terms.
