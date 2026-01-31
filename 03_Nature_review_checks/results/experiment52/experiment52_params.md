# Run parameters

**Experiment**: experiment52
**Timestamp**: 2026-01-31 01:15:55
**Output directory**: `03_Nature_review_checks/results/experiment52`

## Command

```bash
julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl --circuit=Heisenberg --L=8 --steps=1 --dt=0.05 --periodic=true --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.5 --sites=1,4,8 --state_jl=Neel --state_py=neel --chi_max=64 --trunc=1e-12 --trunc_mode=relative --jl_local_mode=TDVP --jl_longrange_mode=TDVP --jl_tdvp_gate_sweeps=1 --warmup=false --jl_run_tebd=false --jl_run_src=false --jl_run_zipup=false --jl_compare_bug=false --tag_var=none --tag_exact=none --outdir=03_Nature_review_checks/results
```

## Parsed flags (from command)

| key | value |
|---|---|
| `Jx` | `1.0` |
| `Jy` | `1.0` |
| `Jz` | `1.0` |
| `L` | `8` |
| `chi_max` | `64` |
| `circuit` | `Heisenberg` |
| `dt` | `0.05` |
| `h` | `0.5` |
| `jl_compare_bug` | `false` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_run_src` | `false` |
| `jl_run_tebd` | `false` |
| `jl_run_zipup` | `false` |
| `jl_tdvp_gate_sweeps` | `1` |
| `outdir` | `03_Nature_review_checks/results` |
| `periodic` | `true` |
| `sites` | `1,4,8` |
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
| `L` | `8` |
| `base_outdir` | `03_Nature_review_checks/results` |
| `chi_max` | `64` |
| `circuit` | `Heisenberg` |
| `jl_bug_truncation_granularity` | `after_sweep` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_tdvp_gate_sweeps` | `1` |
| `jl_tdvp_truncation` | `during` |
| `outdir` | `03_Nature_review_checks/results/experiment52` |
| `sites` | `1,4,8` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `trunc` | `1.0e-12` |
| `trunc_julia_internal` | `1.0e-12` |
| `trunc_mode` | `relative` |
| `warmup` | `false` |

