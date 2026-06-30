# 2-Qubit Bitflip Variance Data

This folder contains the data artifacts for
`PaperExps/scripts/2sites_variance_exp_circuit_tjm.jl`.

The experiment is the 2-qubit variance comparison under sparse Pauli-X
bitflip noise on `X on site 1`, `X on site 2`, and `X on both sites`
(`IX`, `XI`, and `XX` in the script comments). The serialized pickle files are
the `L=2` runs found in the Desktop source folder `cTJM data`, and the
`plots/` directory contains the corresponding plot PNGs.

## Contents

- `data/`: three serialized `variance_comparison_*L2*.pkl` runs.
- `plots/`: six variance and expectation comparison PNGs.
