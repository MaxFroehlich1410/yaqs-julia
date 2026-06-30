# Paper Experiment Data

This directory contains the data artifacts used for the plots associated with the
paper experiments in `PaperExps/scripts/`.

The data was collected from the Desktop folder `cTJM data` and organized here so
the paper code and the plotted data live in the same repository. Runtime caches
and helper artifacts such as `.DS_Store`, `__pycache__`, `.pyc`, `.jls`, and
temporary TOML files were intentionally excluded.

## Contents

- `two_site_bitflip_noise/`: serialized 2-qubit bitflip-noise variance
  comparison runs and final figure PNGs corresponding to
  `scripts/2sites_variance_exp_circuit_tjm.jl`. The script defines the
  IX/XI/XX Pauli-X noise model used for these `L=2` runs.
- `ctjm_paper_data/`: curated CTJM paper plot data.
  - `unraveling_eff_N25_L20_tau0p1_noise*_basisXY_*`: 25-site XY quench data
    for noise strengths `0.001`, `0.01`, and `0.1`. These folders include the
    aggregate `LargeSystem_*.pkl` files, per-method batch pickle sources, and
    the generated comparison PNGs.
  - `127_datapoints_L5/`: IBM 127-qubit kicked-Ising data. The
    `127_datapoints_L5_V2/` subtree contains standard-run CSV exports, while
    sibling L5 folders contain projector and aggregate pickle outputs. The L20
    aggregate folders are retained as legacy/reference outputs from the same
    data collection.

## Notes

- `.pkl` files are Python pickle artifacts produced by the experiment scripts or
  associated aggregation scripts.
- `.csv` files are plain numeric exports used by the IBM 127-qubit comparison
  plots.
- `.png` files are included as visual references for the exact plots backed by
  the data in the neighboring folders.
