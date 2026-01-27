# Run parameters

**Experiment**: experiment17
**Timestamp**: 2026-01-27 20:40:10
**Output directory**: `03_Nature_review_checks/results/experiment17`

## Command

```bash
julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl --circuit=Heisenberg --L=10 --steps=8 --dt=0.1 --periodic=true --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.5 --sites=1,4,5,6 --state_jl=Neel --state_py=neel --chi_max=256 --trunc=1e-16 --trunc_mode=relative --jl_tdvp_truncation=after_window --jl_local_mode=TDVP --jl_longrange_mode=TDVP --jl_run_tebd=true --jl_run_src=true --jl_compare_bug=true --tag_jl=circuitTDVP --tag_jl_tebd=circuitTEBD --tag_jl_src=circuitSRC --tag_jl_bug=circuitBUG --outdir=03_Nature_review_checks/results
```

## Parsed flags (from command)

| key | value |
|---|---|
| `Jx` | `1.0` |
| `Jy` | `1.0` |
| `Jz` | `1.0` |
| `L` | `10` |
| `chi_max` | `256` |
| `circuit` | `Heisenberg` |
| `dt` | `0.1` |
| `h` | `0.5` |
| `jl_compare_bug` | `true` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_run_src` | `true` |
| `jl_run_tebd` | `true` |
| `jl_tdvp_truncation` | `after_window` |
| `outdir` | `03_Nature_review_checks/results` |
| `periodic` | `true` |
| `sites` | `1,4,5,6` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `steps` | `8` |
| `tag_jl` | `circuitTDVP` |
| `tag_jl_bug` | `circuitBUG` |
| `tag_jl_src` | `circuitSRC` |
| `tag_jl_tebd` | `circuitTEBD` |
| `trunc` | `1e-16` |
| `trunc_mode` | `relative` |

## Resolved parameters used

| key | value |
|---|---|
| `L` | `10` |
| `base_outdir` | `03_Nature_review_checks/results` |
| `chi_max` | `256` |
| `circuit` | `Heisenberg` |
| `jl_bug_truncation_granularity` | `after_sweep` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_tdvp_truncation` | `after_window` |
| `outdir` | `03_Nature_review_checks/results/experiment17` |
| `sites` | `1,4,5,6` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `trunc` | `1.0e-16` |
| `trunc_julia_internal` | `1.0e-16` |
| `trunc_mode` | `relative` |
| `warmup` | `true` |

