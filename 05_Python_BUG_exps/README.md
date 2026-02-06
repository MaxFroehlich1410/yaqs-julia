# Python BUG experiments (mirror of 04_BUG_exps)

Same options and behaviour as `04_BUG_exps/exp_runs.jl`, but implemented in Python using **mqt-yaqs** TDVP and BUG.

## Requirements

- Python 3.9+
- `mqt-yaqs` (uses the copy in `../mqt-yaqs/src` if present, otherwise the installed package)
- `numpy`, `opt_einsum` (mqt-yaqs dependencies)
- For plotting: `matplotlib`
- For reference runs: `qutip` (`reference=qutip`)

## Run

From this directory:

```bash
python3 exp_runs.py [options]
```

Or from the repo root:

```bash
python3 05_Python_BUG_exps/exp_runs.py [options]
```

## Options (same as Julia)

| Option | Default | Description |
|--------|---------|-------------|
| `outdir` | auto `experimentN` | Output directory |
| `L` | 8 | Chain length |
| `model` | tfim | tfim, ising, heisenberg, xx, xy, xxz, general |
| `reference` | qutip | qutip \| none |
| `plot` | true | Whether to produce comparison plots |
| `J`, `g`, `Delta`, `gamma` | (varies) | Model parameters |
| `Jxx`, `Jyy`, `Jzz`, `hx`, `hy`, `hz` | 0 | For model=general |
| `dt` | 0.1 | Time step |
| `steps` | 40 | Number of steps |
| `initial_state` | x+ | zeros, ones, x+, Neel |
| `site` | mid | Site for ⟨Z⟩ |
| `max_bond_dim` | 128 | Max bond (adaptive methods) |
| `fixed_max_bond_dim` | same | Bond for 1TDVP + DOUBLEFIXED |
| `threshold` | 1e-12 | Truncation threshold |
| `numiter_lanczos` | 25 | Lanczos iterations |
| `methods` | DOUBLEFIXED,DOUBLEADAPTIVE,SINGLE_SITE_TDVP,TWO_SITE_TDVP | Comma-separated |

## Example

```bash
python3 exp_runs.py --L=8 --steps=40 --reference=qutip --methods=SINGLE_SITE_TDVP,TWO_SITE_TDVP,DOUBLEFIXED,DOUBLEADAPTIVE
```

Output: `experimentN/timeseries.csv`, `meta.txt`, and (if `plot=true`) `runs_comparison.png`.
