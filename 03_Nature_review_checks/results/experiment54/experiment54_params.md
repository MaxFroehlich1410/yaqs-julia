# Run parameters

**Experiment**: experiment54
**Timestamp**: 2026-01-31 01:21:29
**Output directory**: `03_Nature_review_checks/results/experiment54`

## Command

```bash
julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl --circuit=Heisenberg --L=6 --steps=1 --dt=0.05 --periodic=false --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.0 --sites=1,3,6 --state_jl=Neel --state_py=neel --chi_max=32 --trunc=1e-12 --trunc_mode=relative --jl_local_mode=TDVP --jl_longrange_mode=TDVP --jl_tdvp_gate_sweeps=1 --warmup=false --jl_run_tebd=false --jl_run_src=false --jl_run_zipup=false --jl_compare_bug=false --tag_var=none --tag_exact=none --krylov_tol=1e-10 --outdir=03_Nature_review_checks/results
```

## Parsed flags (from command)

| key | value |
|---|---|
| `Jx` | `1.0` |
| `Jy` | `1.0` |
| `Jz` | `1.0` |
| `L` | `6` |
| `chi_max` | `32` |
| `circuit` | `Heisenberg` |
| `dt` | `0.05` |
| `h` | `0.0` |
| `jl_compare_bug` | `false` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_run_src` | `false` |
| `jl_run_tebd` | `false` |
| `jl_run_zipup` | `false` |
| `jl_tdvp_gate_sweeps` | `1` |
| `krylov_tol` | `1e-10` |
| `outdir` | `03_Nature_review_checks/results` |
| `periodic` | `false` |
| `sites` | `1,3,6` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `steps` | `1` |
| `tag_exact` | `none` |
| `tag_var` | `none` |
| `trunc` | `1e-12` |
| `trunc_mode` | `relative` |
| `warmup` | `false` |

## Resolved parameters used

| key | value |
|---|---|
| `L` | `6` |
| `base_outdir` | `03_Nature_review_checks/results` |
| `chi_max` | `32` |
| `circuit` | `Heisenberg` |
| `jl_bug_truncation_granularity` | `after_sweep` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_tdvp_gate_sweeps` | `1` |
| `jl_tdvp_truncation` | `during` |
| `outdir` | `03_Nature_review_checks/results/experiment54` |
| `sites` | `1,3,6` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `trunc` | `1.0e-12` |
| `trunc_julia_internal` | `1.0e-12` |
| `trunc_mode` | `relative` |
| `warmup` | `false` |

