# 25-Qubit Simulation Data

This folder contains the data artifacts for
`PaperExps/scripts/25q_exp.jl`.

The experiment is the 25-qubit XY-quench simulation at noise strengths
`0.001`, `0.01`, and `0.1`. The data subfolders preserve the source run names
and include aggregate `LargeSystem_*.pkl` files plus the per-method batch
pickle sources used for the paper plots.

## Contents

- `data/`: serialized 25-qubit simulation data and the all-noise run note.
- `plots/`: aggregate all-noise and per-noise comparison PNGs.
