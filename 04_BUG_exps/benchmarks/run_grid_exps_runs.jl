"""
Run `04_BUG_exps/exp_runs.jl` on a parameter grid and save each run
into its own folder.

This is a thin driver that just spawns Julia subprocesses.

Example:
  julia --project=. 04_BUG_exps/run_grid_exp_runs.jl --outbase=04_BUG_exps/grid_results

Notes:
- `exp_runs.jl` expects scalar `dt`, `max_bond_dim`, `threshold`; this script loops.
- `exp_runs.jl` does NOT support `--truncation_timing`. Use `--truncation_mode=after_sweep`
  to defer truncation consistently across BUG/TDVP in the harness.
"""

using Printf

function _parse_kv_args(args::Vector{String})
    out = Dict{String,String}()
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        if occursin("=", s)
            k, v = split(s, "=", limit=2)
            out[k] = v
        else
            out[s] = "true"
        end
    end
    return out
end

@inline function _slug(x::Real)
    # Stable-ish directory token:
    # 0.05 -> "0p05", 1e-16 -> "1e-16", 0 -> "0"
    if x == 0
        return "0"
    end
    s = @sprintf("%.16g", float(x))
    s = replace(s, "." => "p")
    s = replace(s, "-" => "m")
    return s
end

function main(args=ARGS)
    kv = _parse_kv_args(args)
    outbase = get(kv, "outbase", joinpath(@__DIR__, "grid_results"))
    mkpath(outbase)

    # Grid (edit here if you want different values)
    thresholds = (0)
    dts        = (0.005)
    chis       = (32,)

    # Common settings (mirrors your command, but uses supported flags/names)
    common = [
        "--measure_runtime=true",
        "--truncation_mode=during",
        "--model=general",
        "--reference=exact",
        "--adaptive_pad=4",
        "--plot=true",
        "--L=10",
        "--site=5",
        "--steps=200",
        "--initial_state=Neel",
        "--methods=DOUBLEFIXED,DOUBLEADAPTIVE,SINGLE_SITE_TDVP,TWO_SITE_TDVP",
        "--fixed_max_bond_dim=16",
        "--numiter_lanczos=25",
        # Couplings / fields: use the fully general Hamiltonian:
        "--Jxx=1.0",
        "--Jyy=1.0",
        "--Jzz=0.5",
        "--hx=0.0",
        "--hy=0.0",
        "--hz=0.0",
    ]

    total = length(thresholds) * length(dts) * length(chis)
    n = 0
    for thr in thresholds, dt in dts, chi in chis
        n += 1
        dtslug = _slug(dt)
        thrslug = _slug(thr)
        outdir = joinpath(outbase, "dt$(dtslug)_chi$(chi)_thr$(thrslug)")
        mkpath(outdir)

        # Skip if already completed (timeseries exists and non-empty)
        csvfile = joinpath(outdir, "timeseries.csv")
        if isfile(csvfile) && filesize(csvfile) > 0
            @printf("[grid] (%d/%d) skip existing %s\n", n, total, outdir)
            continue
        end

        tag = "grid_dt=$(dt)_chi=$(chi)_thr=$(thr)"
        cmd = `julia --project=. 04_BUG_exps/exp_runs.jl --outdir=$(outdir) --tag=$(tag) --dt=$(dt) --max_bond_dim=$(chi) --threshold=$(thr)`
        for a in common
            cmd = `$cmd $a`
        end
        @printf("[grid] (%d/%d) run dt=%.4g chi=%d thr=%.3g -> %s\n", n, total, dt, chi, thr, outdir)
        flush(stdout)
        run(cmd)
    end

    @printf("[grid] done. outbase=%s\n", outbase)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end