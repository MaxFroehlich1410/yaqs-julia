# Run parameters

**Experiment**: experiment29
**Timestamp**: 2026-01-28 00:40:28
**Output directory**: `03_Nature_review_checks/results/experiment29`

## Command

```bash
julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl --circuit=Heisenberg --L=25 --steps=4 --dt=0.05 --periodic=true --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.5 --sites=1,4,5,6,7,8,20,25 --state_jl=Neel --state_py=neel --chi_max=512 --trunc=0 --trunc_mode=relative --jl_local_mode=TDVP --jl_longrange_mode=TDVP --warmup=true --jl_run_tebd=true --tag_jl_tebd=circuitTEBD --jl_run_src=true --tag_jl_src=circuitSRC --jl_run_zipup=true --tag_jl_zipup=circuitZIPUP --jl_compare_bug=true --tag_jl_bug=circuitBUG --jl_tdvp_truncation=after_window --min_sweeps=2 --max_sweeps=10 --tag_jl=circuitTDVP --tag_var=none --tag_exact=none --outdir=03_Nature_review_checks/results
```

## Parsed flags (from command)

| key | value |
|---|---|
| `Jx` | `1.0` |
| `Jy` | `1.0` |
| `Jz` | `1.0` |
| `L` | `25` |
| `chi_max` | `512` |
| `circuit` | `Heisenberg` |
| `dt` | `0.05` |
| `h` | `0.5` |
| `jl_compare_bug` | `true` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_run_src` | `true` |
| `jl_run_tebd` | `true` |
| `jl_run_zipup` | `true` |
| `jl_tdvp_truncation` | `after_window` |
| `max_sweeps` | `10` |
| `min_sweeps` | `2` |
| `outdir` | `03_Nature_review_checks/results` |
| `periodic` | `true` |
| `sites` | `1,4,5,6,7,8,20,25` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `steps` | `4` |
| `tag_exact` | `none` |
| `tag_jl` | `circuitTDVP` |
| `tag_jl_bug` | `circuitBUG` |
| `tag_jl_src` | `circuitSRC` |
| `tag_jl_tebd` | `circuitTEBD` |
| `tag_jl_zipup` | `circuitZIPUP` |
| `tag_var` | `none` |
| `trunc` | `0` |
| `trunc_mode` | `relative` |
| `warmup` | `true` |

## Resolved parameters used

| key | value |
|---|---|
| `L` | `25` |
| `base_outdir` | `03_Nature_review_checks/results` |
| `chi_max` | `512` |
| `circuit` | `Heisenberg` |
| `jl_bug_truncation_granularity` | `after_sweep` |
| `jl_local_mode` | `TDVP` |
| `jl_longrange_mode` | `TDVP` |
| `jl_tdvp_truncation` | `after_window` |
| `outdir` | `03_Nature_review_checks/results/experiment29` |
| `sites` | `1,4,5,6,7,8,20,25` |
| `state_jl` | `Neel` |
| `state_py` | `neel` |
| `trunc` | `0.0` |
| `trunc_julia_internal` | `0.0` |
| `trunc_mode` | `relative` |
| `warmup` | `true` |

