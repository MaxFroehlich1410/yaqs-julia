module BUGBatchExperiments

using Printf

include("exp_runs.jl")
using .BUGExpRuns

function _existing_tags(basedir::AbstractString)
    tags = Set{String}()
    for name in readdir(basedir)
        m = match(r"^experiment(\d+)$", name)
        m === nothing && continue
        d = joinpath(basedir, name)
        isdir(d) || continue
        meta = joinpath(d, "meta.txt")
        isfile(meta) || continue
        for line in eachline(meta)
            startswith(line, "tag=") || continue
            tag = strip(line[5:end])
            isempty(tag) || push!(tags, tag)
            break
        end
    end
    return tags
end

"""
Run 15 curated experiments (L=8..14) across several models.

Each call relies on `exp_runs.jl` auto-outdir logic, so results go into:
`04_BUG_exps/experimentX/`.
"""
function main()
    basedir = @__DIR__
    logfile = joinpath(basedir, "batch_15.log")

    exps = [
        # All experiments run:
        # - exact reference: qutip
        # - all methods (default exp_runs list): 4 BUG + 2 TDVP
        # - plotting enabled (runs_comparison.png per experiment)
        #
        # --- TFIM ---
        (; tag="tfim_L8_easy",          model="tfim", L=8,  J=1.0, g=0.5, dt=0.1,  steps=100, initial_state="x+", max_bond_dim=128, fixed_max_bond_dim=128, threshold=1e-12, numiter_lanczos=25, reference="qutip"),
        (; tag="tfim_L9_near_critical", model="tfim", L=9,  J=1.0, g=1.0, dt=0.05, steps=200, initial_state="x+", max_bond_dim=192, fixed_max_bond_dim=192, threshold=1e-12, numiter_lanczos=30, reference="qutip"),
        (; tag="tfim_L10_strong_field", model="tfim", L=10, J=1.0, g=1.6, dt=0.05, steps=200, initial_state="zeros", max_bond_dim=192, fixed_max_bond_dim=192, threshold=1e-12, numiter_lanczos=30, reference="qutip"),
        (; tag="tfim_L12_weak_field",   model="tfim", L=12, J=1.0, g=0.3, dt=0.1,  steps=150, initial_state="x+", max_bond_dim=256, fixed_max_bond_dim=256, threshold=1e-12, numiter_lanczos=30, reference="qutip"),

        # --- XX / XY ---
        (; tag="xx_L8",                 model="xx",   L=8,  J=1.0, hx=0.05, hz=0.00, dt=0.05, steps=200, initial_state="x+", max_bond_dim=192, fixed_max_bond_dim=192, threshold=1e-12, numiter_lanczos=30, reference="qutip"),
        (; tag="xx_L10_with_hz",        model="xx",   L=10, J=1.0, hx=0.00, hz=0.20, dt=0.05, steps=200, initial_state="zeros", max_bond_dim=256, fixed_max_bond_dim=256, threshold=1e-12, numiter_lanczos=30, reference="qutip"),
        (; tag="xy_L9_gamma0p5",        model="xy",   L=9,  J=1.0, gamma=0.5, hx=0.00, hz=0.00, dt=0.05, steps=200, initial_state="x+", max_bond_dim=256, fixed_max_bond_dim=256, threshold=1e-12, numiter_lanczos=30, reference="qutip"),
        (; tag="xy_L11_gamma0p8_hx",    model="xy",   L=11, J=1.0, gamma=0.8, hx=0.10, hz=0.00, dt=0.05, steps=200, initial_state="ones", max_bond_dim=320, fixed_max_bond_dim=320, threshold=1e-12, numiter_lanczos=35, reference="qutip"),

        # --- XXZ ---
        (; tag="xxz_L10_Delta0p5",      model="xxz",  L=10, J=1.0, Delta=0.5, hx=0.00, hz=0.00, dt=0.05, steps=200, initial_state="x+", max_bond_dim=256, fixed_max_bond_dim=256, threshold=1e-12, numiter_lanczos=35, reference="qutip"),
        (; tag="xxz_L12_Delta1p5_hz",   model="xxz",  L=12, J=1.0, Delta=1.5, hx=0.00, hz=0.10, dt=0.05, steps=200, initial_state="zeros", max_bond_dim=384, fixed_max_bond_dim=384, threshold=1e-12, numiter_lanczos=35, reference="qutip"),

        # --- Heisenberg ---
        (; tag="heis_L8",              model="heisenberg", L=8,  J=1.0, hx=0.05, hz=0.00, dt=0.05, steps=200, initial_state="x+", max_bond_dim=256, fixed_max_bond_dim=256, threshold=1e-12, numiter_lanczos=35, reference="qutip"),
        (; tag="heis_L10_with_hz",     model="heisenberg", L=10, J=1.0, hx=0.00, hz=0.20, dt=0.05, steps=200, initial_state="zeros", max_bond_dim=384, fixed_max_bond_dim=384, threshold=1e-12, numiter_lanczos=35, reference="qutip"),

        # --- Long-time performance (t = 50) with exact reference (keep L moderate so qutip remains feasible) ---
        (; tag="tfim_L8_long_t50",      model="tfim", L=8,  J=1.0, g=0.8, dt=0.1, steps=500, initial_state="x+", max_bond_dim=256, fixed_max_bond_dim=256, threshold=1e-12, numiter_lanczos=35, reference="qutip"),
        (; tag="xxz_L10_long_t50",      model="xxz",  L=10, J=1.0, Delta=1.2, hx=0.00, hz=0.10, dt=0.1, steps=500, initial_state="zeros", max_bond_dim=384, fixed_max_bond_dim=384, threshold=1e-12, numiter_lanczos=35, reference="qutip"),
        (; tag="heis_L10_long_t50",     model="heisenberg", L=10, J=1.0, hx=0.05, hz=0.00, dt=0.1, steps=500, initial_state="x+", max_bond_dim=384, fixed_max_bond_dim=384, threshold=1e-12, numiter_lanczos=35, reference="qutip"),
    ]

    open(logfile, "a") do io
        function log(msg::AbstractString)
            println(msg)
            println(io, msg)
            flush(io)
        end

        log(@sprintf("[batch] launching %d experiments", length(exps)))

        for (i, e) in enumerate(exps)
            log(@sprintf("\n[batch] (%d/%d) tag=%s model=%s L=%d dt=%.3g steps=%d ref=%s",
                         i, length(exps), e.tag, e.model, e.L, e.dt, e.steps, e.reference))

            args = String[
                "--tag=$(e.tag)",
                "--model=$(e.model)",
                "--reference=$(e.reference)",
                "--plot=true",
                "--L=$(e.L)",
                "--dt=$(e.dt)",
                "--steps=$(e.steps)",
                "--initial_state=$(e.initial_state)",
                "--max_bond_dim=$(e.max_bond_dim)",
                "--fixed_max_bond_dim=$(e.fixed_max_bond_dim)",
                "--threshold=$(e.threshold)",
                "--numiter_lanczos=$(e.numiter_lanczos)",
            ]

            # Common params
            if hasproperty(e, :J);     push!(args, "--J=$(getproperty(e, :J))"); end
            if hasproperty(e, :g);     push!(args, "--g=$(getproperty(e, :g))"); end
            if hasproperty(e, :Delta); push!(args, "--Delta=$(getproperty(e, :Delta))"); end
            if hasproperty(e, :gamma); push!(args, "--gamma=$(getproperty(e, :gamma))"); end
            if hasproperty(e, :hx);    push!(args, "--hx=$(getproperty(e, :hx))"); end
            if hasproperty(e, :hy);    push!(args, "--hy=$(getproperty(e, :hy))"); end
            if hasproperty(e, :hz);    push!(args, "--hz=$(getproperty(e, :hz))"); end

            BUGExpRuns.main(args)
            log(@sprintf("[batch] finished tag=%s", e.tag))
        end

        log("\\n[batch] done")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module BUGBatchExperiments

