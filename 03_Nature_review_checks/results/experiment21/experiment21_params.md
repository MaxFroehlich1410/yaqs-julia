# Run parameters

**Experiment**: experiment21
**Timestamp**: 2026-01-27 23:39:52
**Output directory**: `03_Nature_review_checks/results/experiment21`

## Command

```bash
julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl --circuit=Heisenberg --L=16 --steps=30 --dt=0.05 --periodic=true --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.0 --sites=1,4,8,15,16 --state_jl=Neel --state_py=neel --chi_max=100 --trunc=1e-16 --trunc_mode=relative --jl_local_mode=TDVP --jl_longrange_mode=TDVP --warmup=true --jl_run_tebd=true --tag_jl_tebd=circuitTEBD --jl_run_zipup=false --tag_jl_zipup=circuitZIPUP --jl_tdvp_truncation=after_window --min_sweeps=2 --max_sweeps=10 --tag_jl=circuitTDVP --tag_var=none --tag_exact=none --outdir=03_Nature_review_checks/results
```

## Parsed flags (from command)

| key | value |
|---|---|
| `Jx` | `1.0` |
| `Jy` | `1.0` |
| `Jz` | `1.0` |
| `L` | `16` |
| `chi_max` | `100` |
| `circuit` | `Heisenberg` |
| `dt` | `0.05` |
| `h` | `0.0` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_run_tebd` | `true` |
| `jl_run_zipup` | `false` |
| `jl_tdvp_truncation` | `after_window` |
| `max_sweeps` | `10` |
| `min_sweeps` | `2` |
| `outdir` | `03_Nature_review_checks/results` |
| `periodic` | `true` |
| `sites` | `1,4,8,15,16` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `steps` | `30` |
| `tag_exact` | `none` |
| `tag_jl` | `circuitTDVP` |
| `tag_jl_tebd` | `circuitTEBD` |
| `tag_jl_zipup` | `circuitZIPUP` |
| `tag_var` | `none` |
| `trunc` | `1e-16` |
| `trunc_mode` | `relative` |
| `warmup` | `true` |

## Resolved parameters used

| key | value |
|---|---|
| `L` | `16` |
| `base_outdir` | `03_Nature_review_checks/results` |
| `chi_max` | `100` |
| `circuit` | `Heisenberg` |
| `jl_bug_truncation_granularity` | `after_sweep` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_tdvp_truncation` | `after_window` |
| `outdir` | `03_Nature_review_checks/results/experiment21` |
| `sites` | `1,4,8,15,16` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `trunc` | `1.0e-16` |
| `trunc_julia_internal` | `1.0e-16` |
| `trunc_mode` | `relative` |
| `warmup` | `true` |

