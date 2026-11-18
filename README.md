# Fault-Side Encoding and Chronology-Constrained INR

Code accompanying *Implicit Neural Representation for 3D Geological Modeling Using Fault Side Encoding and Chronology Constraints*.

This repository provides the neural implicit modeling components, fault-side feature encoding, and chronology-conditioned valid-domain gating used in the study. Fault geometry and relative chronology are supplied as structural inputs; the code does not infer unknown fault ages or truncation relationships.

## Workflow

![Workflow for chronology-constrained fault-side encoding and implicit geological modeling](workflow.png)

Geological interface and orientation observations, together with interpreted fault surfaces, are converted into network inputs. A chronology rule gates an older fault-side feature according to the retained side of its younger controlling fault. The conditioned inputs are used to learn a stratigraphic scalar field and extract geological interfaces.

## Repository layout

~~~text
fault-side-encoding-chronology-inr/
├── data/
│   └── case1/
│       ├── fault_meshes/
│       │   ├── fault1_raw.vtp
│       │   └── fault2_raw.vtp
│       ├── orientation.csv
│       └── surface.csv
├── feature_encoding/
│   ├── chronology_mask.py
│   └── fault_side_encoding.py
├── experiments/
│   ├── controlled_case.py
│   └── feature_variants.py
├── geological_models.py
├── geological_utils.py
├── losses.py
├── networks.py
├── requirements.txt
├── LICENSE
├── workflow.png
└── README.md
~~~

## Software

Install the listed dependencies from the repository root:

~~~bash
python -m pip install -r requirements.txt
~~~

The model and feature modules use PyTorch, NumPy, pandas, PyVista, and VTK. GPU use depends on the locally installed PyTorch build and CUDA-compatible hardware.

## Chronology-conditioned feature gating

The feature_encoding/chronology_mask.py module exposes TemporalRule, apply_chronology_mask, and gate_domain_features.

A rule specifies an older fault, a younger controlling fault, and the retained side of the younger fault. For example:

~~~python
from feature_encoding import TemporalRule, gate_domain_features

rules = [
    TemporalRule(
        older_fault="fault1",
        younger_fault="fault2",
        retained_younger_side=1,
    )
]
effective_domain, mask_summary = gate_domain_features(
    domain_features,
    fault_names=["fault1", "fault2"],
    rules=rules,
)
~~~

The input domain array has columns [X, Y, Z, fault_1, ..., fault_M]. Side features are binary encodings derived from interpreted fault surfaces; they are not exact signed-distance fields. The mask represents the supplied chronology rule and does not establish a unique geological interpretation.

## Data scope

The included data/case1/ files provide the two-fault example inputs distributed with this code. The controlled-case implementation is in experiments/controlled_case.py; its generated outputs are written under outputs/ and are not versioned. The engineering mine dataset is not included.

## Citation

Please cite the associated manuscript:

> Chen, Y., Xin, J., and Wang, P. *Implicit Neural Representation for 3D Geological Modeling Using Fault Side Encoding and Chronology Constraints*. Manuscript.

## License

This project is distributed under the MIT License. See LICENSE.
