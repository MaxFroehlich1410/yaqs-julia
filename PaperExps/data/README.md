# Paper Experiment Data

This directory contains the data artifacts used for the plots associated with the
paper experiments in `PaperExps/scripts/`.

The data was collected from the Desktop folder `cTJM data` and organized here so
the paper code and the plotted data live in the same repository. Runtime caches
and helper artifacts such as `.DS_Store`, `__pycache__`, `.pyc`, `.jls`, and
temporary TOML files were intentionally excluded.

## Contents

- `2_qubit_bitflip_variance/`: 2-qubit bitflip-noise variance experiment for
  `scripts/2sites_variance_exp_circuit_tjm.jl`.
- `25_qubit_simulation/`: 25-qubit XY-quench simulation data and plots for
  `scripts/25q_exp.jl`.
- `127_qubit_simulation/`: IBM 127-qubit kicked-Ising simulation data and plots
  for `scripts/ibm127_exp.jl`.

Each experiment folder contains:

- `data/`: serialized data artifacts used to generate the plot.
- `plots/`: one canonical reference plot PNG generated from the neighboring
  data.

## Notes

- `.pkl` files are Python pickle artifacts produced by the experiment scripts or
  associated aggregation scripts.
- `.csv` files are plain numeric exports used by the IBM 127-qubit comparison
  plots.
- `.png` files are included as visual references for the exact plot backed by
  the data in the neighboring folder.
