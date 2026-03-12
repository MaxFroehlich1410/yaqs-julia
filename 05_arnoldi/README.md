# Non-Hermitian TDVP Benchmark (`05_arnoldi`)

This folder adds a compact benchmark for non-Hermitian nearest-neighbor dynamics:

\[
\frac{d}{dt}\psi(t)=K\psi(t)
\]

with

\[
K=-iJ\sum_{j=1}^{N-1} Z_j Z_{j+1} - ig\sum_{j=1}^{N}X_j - \gamma\sum_{j=1}^{N}\frac{I-Z_j}{2}.
\]

The benchmark compares:
- exact dense full-Hilbert-space evolution (`2^N` vector), and
- MPS-TDVP evolution using the repository's existing TDVP implementation.

## Files Added

- `nonhermitian_tfim.jl`
  - Builds the non-Hermitian MPO for `K` with bond dimension `3` (open boundaries) using the requested block form.
  - Provides Pauli/identity helpers.
  - Provides initial states: MPS `|+>^{⊗N}` and dense `|+>^{⊗N}`.
- `exact_reference.jl`
  - Builds dense `K_dense` in full space (`2^N x 2^N`) directly (no tensor-train methods).
  - Evolves exactly via `U = exp(dt * K_dense)` once, then repeated `psi <- U * psi`.
  - Computes exact local `⟨Z_j⟩ = psi† Z_j psi` (unnormalized).
- `run_nonhermitian_tdvp.jl`
  - End-to-end benchmark runner with CLI flags.
  - Runs exact reference and TDVP on the same time grid.
  - Writes CSV time series and a text summary to `05_arnoldi/results/`.
- `plot_results.jl`
  - Reads `timeseries_D*.csv` and generates comparison plots (exact vs Arnoldi-TDVP).
  - Writes PNG files to `05_arnoldi/results/`.

## Existing TDVP Routine Used

The benchmark uses:
- `Yaqs.Algorithms.two_site_tdvp!` as the TDVP stepper, and
- `Yaqs.Algorithms.set_krylov_ishermitian_mode!` to choose Krylov variant (`:arnoldi` or `:lanczos`).

Default is Arnoldi (`--krylov arnoldi`) because the generator is non-Hermitian.

## Important Convention for `dψ/dt = Kψ`

Repository TDVP internally applies local propagators in the form `exp(-im * dt * A)`.
Therefore, to realize the target equation `dψ/dt = Kψ`, the benchmark passes:

\[
A = iK
\]

so that `exp(-im*dt*A) = exp(dt*K)`.

`nonhermitian_tfim.jl` still constructs and exposes the requested MPO for `K` directly.

## Normalization Caveat

From inspection of `src/Algorithms.jl` TDVP paths used here, there is no explicit `normalize!` call in each time step.
So this benchmark tracks unnormalized dynamics as requested (state norms are reported over time).

## Default Parameters

- `N = 10`
- `J = 1.0`
- `g = 0.7`
- `gamma = 0.2`
- `dt = 0.05`
- `tmax = 1.0`
- initial state `|+>^{⊗N}`
- Krylov `arnoldi`
- bond dimensions `[4, 8, 16]`

These defaults are intended to run in a few seconds on Apple M1-class laptops.

## How To Run

From repository root:

```bash
julia --project=. 05_arnoldi/run_nonhermitian_tdvp.jl
```

Optional flags:

```bash
julia --project=. 05_arnoldi/run_nonhermitian_tdvp.jl \
  --N 10 --J 1.0 --g 0.7 --gamma 0.2 \
  --dt 0.05 --tmax 1.0 \
  --D 4,8,16 \
  --krylov arnoldi
```

`--krylov` accepts `arnoldi` or `lanczos`.

Generate plots from existing CSV results:

```bash
julia --project=. 05_arnoldi/plot_results.jl
```

## Outputs

Written to `05_arnoldi/results/`:

- `summary.txt`
- `timeseries_D4.csv`
- `timeseries_D8.csv`
- `timeseries_D16.csv`
- `comparison_norms.png`
- `comparison_state_error.png`
- `comparison_z_center.png`
- `comparison_max_z_error.png`

Each CSV row is one time step and includes at least:
- `t`
- exact norm `||psi_ref||`
- TDVP norm `||psi_tdvp||`
- state error `||psi_tdvp - psi_ref||_2`
- `max_j |<Z_j>_tdvp - <Z_j>_ref|`
- current MPS max bond dimension
- all per-site `⟨Z_j⟩` values for both exact and TDVP.
