# Long-Range Non-Hermitian TDVP Benchmark (`05_arnoldi`)

This benchmark extends the nearest-neighbor non-Hermitian study to a structured long-range model:

\[
\frac{d}{dt}\psi(t)=K_{\mathrm{lr}}\psi(t)
\]

with

\[
K_{\mathrm{lr}}
=-i\sum_{1\le i<j\le N}J_{ij}Z_iZ_j
-ig\sum_{j=1}^{N}X_j
-\gamma\sum_{j=1}^{N}\frac{I-Z_j}{2},
\quad
J_{ij}=J_0\,\lambda^{(|i-j|-1)}.
\]

## Files

- `longrange_nonhermitian_tfim.jl`
  - Builds the long-range MPO for `K_lr`.
  - Uses repository MPO building blocks and exact MPO summation.
  - Provides initial states `|+>^{⊗N}` (MPS and dense).
- `longrange_exact_reference.jl`
  - Builds dense `K_lr` directly in full Hilbert space (no tensor trains).
  - Evolves exactly with `U = exp(dt * K_lr)`, then repeated `psi <- U*psi`.
  - Computes unnormalized and normalized local `⟨Z_j⟩`.
- `run_longrange_nonhermitian_tdvp.jl`
  - Main benchmark runner with CLI options.
  - Runs exact dense reference and TDVP for each requested `lambda` and `D`.
  - Writes CSV time series and summary to `results_longrange/`.
- `plot_longrange_results.jl`
  - Reads `results_longrange/timeseries_lambda*_D*.csv`.
  - Produces per-lambda comparison plots and aggregate final-error plots.

## MPO Construction Strategy

The long-range MPO is built exactly using existing repository utilities:

1. For each pair `(i,j)`, build an MPO for `Z_i Z_j` using
   `Yaqs.MPOModule.mpo_from_two_qubit_gate_matrix`.
2. Weight each pair MPO by `(-i * J_ij)` and add all pair terms.
3. Build one-site MPOs for `X_j`, `I_j`, `Z_j`, weight them by
   `(-i g)`, `(-gamma/2)`, `(gamma/2)`, and sum them.

This prioritizes correctness and minimal intrusion over compact analytic MPO bond dimensions.

## TDVP Routine and Krylov Switch Reused

This benchmark reuses the same existing TDVP path:

- `Yaqs.Algorithms.two_site_tdvp!`
- `Yaqs.Algorithms.set_krylov_ishermitian_mode!` (`:arnoldi` or `:lanczos`)

Default is Arnoldi (`--krylov arnoldi`) for this non-Hermitian problem.

## Important Convention (`A = iK_lr`)

Repository TDVP internals apply local propagators of the form:

\[
\exp(-i\,dt\,A).
\]

To simulate

\[
\frac{d}{dt}\psi=K_{\mathrm{lr}}\psi,
\]

the runner passes

\[
A=iK_{\mathrm{lr}},
\]

so `exp(-i dt A) = exp(dt K_lr)`.

## Normalization Behavior

From inspection of the used TDVP path in `src/Algorithms.jl`, there is no explicit
`normalize!` call inside each TDVP step in this benchmark path. The benchmark tracks
unnormalized dynamics accordingly.

## Defaults

- `N = 10`
- `J0 = 1.0`
- `lambda = 0.5`
- `g = 0.7`
- `gamma = 0.2`
- `dt = 0.05`
- `tmax = 1.0`
- initial state `|+>^{⊗N}`
- bond dimensions `[4, 8, 16]`
- `krylov = arnoldi`

Optional lambda sweep:
- `--lambda-list 0.25,0.5,0.75`

## Run Commands

Default single-lambda run:

```bash
julia --project=. 05_arnoldi/run_longrange_nonhermitian_tdvp.jl
```

Explicit parameters:

```bash
julia --project=. 05_arnoldi/run_longrange_nonhermitian_tdvp.jl \
  --N 10 --J0 1.0 --lambda 0.5 \
  --g 0.7 --gamma 0.2 \
  --dt 0.05 --tmax 1.0 \
  --D 4,8,16 \
  --krylov arnoldi
```

Lambda sweep:

```bash
julia --project=. 05_arnoldi/run_longrange_nonhermitian_tdvp.jl \
  --lambda-list 0.25,0.5,0.75
```

Generate plots:

```bash
julia --project=. 05_arnoldi/plot_longrange_results.jl
```

## Outputs

Results are written to `05_arnoldi/results_longrange/`, including:

- `summary.txt`
- `timeseries_lambda0.25_D4.csv` (and analogous files for other lambda/D choices)
- `comparison_norms_lambda*.png`
- `comparison_state_error_lambda*.png`
- `comparison_z_center_norm_lambda*.png`
- `comparison_max_z_error_lambda*.png`
- `final_error_vs_lambda.png`
- `final_error_vs_D.png`

Each time-series CSV stores, per time step:
- `t`
- `||psi_ref||`, `||psi_tdvp||`
- `||psi_tdvp - psi_ref||_2`
- `max_j |<Z_j>_tdvp - <Z_j>_ref|`
- current MPS max bond dimension
- all unnormalized `⟨Z_j⟩` values (exact/TDVP)
- all normalized `⟨Z_j⟩ / ||psi||^2` values (exact/TDVP)
