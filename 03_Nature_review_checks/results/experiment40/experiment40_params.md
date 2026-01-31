# Run parameters

**Experiment**: experiment40
**Timestamp**: 2026-01-30 23:56:32
**Output directory**: `03_Nature_review_checks/results/experiment40`

## Command

```bash
julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl --circuit=Heisenberg --L=4 --steps=1 --dt=0.05 --periodic=false --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.5 --sites=1,2 --state_jl=Neel --state_py=neel --chi_max=8 --trunc=0 --trunc_mode=relative --jl_local_mode=TDVP --jl_longrange_mode=TDVP --jl_tdvp_truncation=during --jl_tdvp_gate_sweeps=2 --warmup=false --jl_run_tebd=false --jl_run_src=false --jl_run_zipup=false --jl_compare_bug=false --tag_var=none --tag_exact=none --outdir=03_Nature_review_checks/results
```

## Parsed flags (from command)

| key | value |
|---|---|
| `Jx` | `1.0` |
| `Jy` | `1.0` |
| `Jz` | `1.0` |
| `L` | `4` |
| `chi_max` | `8` |
| `circuit` | `Heisenberg` |
| `dt` | `0.05` |
| `h` | `0.5` |
| `jl_compare_bug` | `false` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_run_src` | `false` |
| `jl_run_tebd` | `false` |
| `jl_run_zipup` | `false` |
| `jl_tdvp_gate_sweeps` | `2` |
| `jl_tdvp_truncation` | `during` |
| `outdir` | `03_Nature_review_checks/results` |
| `periodic` | `false` |
| `sites` | `1,2` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `steps` | `1` |
| `tag_exact` | `none` |
| `tag_var` | `none` |
| `trunc` | `0` |
| `trunc_mode` | `relative` |
| `warmup` | `false` |

## Resolved parameters used

| key | value |
|---|---|
| `L` | `4` |
| `base_outdir` | `03_Nature_review_checks/results` |
| `chi_max` | `8` |
| `circuit` | `Heisenberg` |
| `jl_bug_truncation_granularity` | `after_sweep` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_tdvp_gate_sweeps` | `2` |
| `jl_tdvp_truncation` | `during` |
| `outdir` | `03_Nature_review_checks/results/experiment40` |
| `sites` | `1,2` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `trunc` | `0.0` |
| `trunc_julia_internal` | `0.0` |
| `trunc_mode` | `relative` |
| `warmup` | `false` |

