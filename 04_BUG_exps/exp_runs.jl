module BUGExpRuns

using Printf
using DelimitedFiles

# ------------------------------------------------------------------------------
# Python / QuTiP startup note (macOS / OpenMP):
# Importing QuTiP can hang or take a very long time if OpenMP / BLAS tries to
# spawn many threads during library initialization. We defensively set the common
# thread env vars to 1 *unless the user already set them*.
# This must happen before the first Python/QuTiP import.
# ------------------------------------------------------------------------------
if !haskey(ENV, "OMP_NUM_THREADS")
    ENV["OMP_NUM_THREADS"] = "1"
end
if !haskey(ENV, "OPENBLAS_NUM_THREADS")
    ENV["OPENBLAS_NUM_THREADS"] = "1"
end
if !haskey(ENV, "MKL_NUM_THREADS")
    ENV["MKL_NUM_THREADS"] = "1"
end
if !haskey(ENV, "VECLIB_MAXIMUM_THREADS")
    ENV["VECLIB_MAXIMUM_THREADS"] = "1"
end

using PythonCall

include("exp_util.jl")
using .BUGExpUtil

export main

function _next_experiment_outdir(basedir::AbstractString)
    max_n = 0
    for name in readdir(basedir)
        path = joinpath(basedir, name)
        isdir(path) || continue
        m = match(r"^experiment(\d+)$", name)
        m === nothing && continue
        n = tryparse(Int, m.captures[1])
        n === nothing && continue
        max_n = max(max_n, n)
    end

    n_new = max_n + 1
    outdir = joinpath(basedir, "experiment$(n_new)")
    mkpath(outdir)
    return outdir, n_new
end

function main(args=ARGS)
    kv = parse_kv_args(args)

    outdir = if haskey(kv, "outdir")
        kv["outdir"]
    else
        dir, n = _next_experiment_outdir(@__DIR__)
        @printf("[exp_runs] using auto outdir %s (experiment %d)\n", dir, n)
        flush(stdout)
        dir
    end
    mkpath(outdir) # no-op if already exists

    L = parse(Int, get(kv, "L", "8"))
    tag = get(kv, "tag", "")
    model = get(kv, "model", "tfim")
    reference = lowercase(get(kv, "reference", "qutip")) # qutip|none
    plot = lowercase(get(kv, "plot", "true")) in ("true", "1", "yes", "y")

    # Common / model-specific parameters
    J = parse(Float64, get(kv, "J", "1.0"))          # TFIM + many general models
    g = parse(Float64, get(kv, "g", "0.5"))          # TFIM transverse field strength
    Delta = parse(Float64, get(kv, "Delta", "1.0"))  # XXZ anisotropy
    gamma = parse(Float64, get(kv, "gamma", "0.0"))  # XY anisotropy

    # Fully general couplings (only used for model=general)
    Jxx = parse(Float64, get(kv, "Jxx", "0.0"))
    Jyy = parse(Float64, get(kv, "Jyy", "0.0"))
    Jzz = parse(Float64, get(kv, "Jzz", "0.0"))
    hx = parse(Float64, get(kv, "hx", "0.0"))
    hy = parse(Float64, get(kv, "hy", "0.0"))
    hz = parse(Float64, get(kv, "hz", "0.0"))

    dt = parse(Float64, get(kv, "dt", "0.1"))
    steps = parse(Int, get(kv, "steps", "40"))
    initial_state = get(kv, "initial_state", "x+")
    max_bond_dim = parse(Int, get(kv, "max_bond_dim", "128"))
    adaptive_pad = parse(Int, get(kv, "adaptive_pad", "4"))
    truncation_mode = Symbol(lowercase(get(kv, "truncation_mode", "during")))
    # Fixed-method fairness: enforce a single shared padded bond dimension
    # across {1TDVP, fixed BUG 2nd}.
    fixed_max_bond_dim = parse(Int, get(kv, "fixed_max_bond_dim", string(max_bond_dim)))
    threshold = parse(Float64, get(kv, "threshold", "1e-12"))
    numiter_lanczos = parse(Int, get(kv, "numiter_lanczos", "25"))
    site = parse(Int, get(kv, "site", string(mid_site(L))))
    measure_runtime = lowercase(get(kv, "measure_runtime", "false")) in ("true", "1", "yes", "y")

    # Default: compare only the methods we care about:
    # - TDVP baselines (single-site + two-site)
    # - BUG 2nd order (fixed-bond + adaptive-bond)
    # (All optionally compared to qutip for accuracy.)
    methods_str = get(kv, "methods", "DOUBLEFIXED,DOUBLEADAPTIVE,SINGLE_SITE_TDVP,TWO_SITE_TDVP")
    methods = split(methods_str, ",")

    runtimes = Dict{String, Float64}()
    times_ref = collect(0:dt:(steps * dt))
    z_ref = nothing
    if reference == "qutip"
        t_ref0 = time()
        times_ref, zq = qutip_expect_z_site(L;
            model=model,
            J=J, g=g, Delta=Delta, gamma=gamma,
            Jxx=Jxx, Jyy=Jyy, Jzz=Jzz, hx=hx, hy=hy, hz=hz,
            dt=dt, steps=steps, initial_state=initial_state, site=site
        )
        z_ref = zq
        runtimes["qutip"] = time() - t_ref0
    elseif reference == "none"
        # no reference
    else
        error("Unsupported reference=$reference (supported: qutip, none)")
    end

    # Run each method and collect results.
    results = Dict{String, Vector{Float64}}()
    bond_dims = Dict{String, Vector{Int}}()

    fixed_methods = ["SINGLE_SITE_TDVP", "DOUBLEFIXED"]
    adaptive_methods = ["TWO_SITE_TDVP", "DOUBLEADAPTIVE"]
    track_bond_dims(m::AbstractString) = (m in adaptive_methods)
    for m in methods
        is_fixed_family = (m == "SINGLE_SITE_TDVP") || (m == "DOUBLEFIXED")
        chi = is_fixed_family ? fixed_max_bond_dim : max_bond_dim
        if track_bond_dims(m)
            times, z, bd, wall = run_method_expect_z_site(m;
                L=L,
                model=model,
                J=J, g=g, Delta=Delta, gamma=gamma,
                Jxx=Jxx, Jyy=Jyy, Jzz=Jzz, hx=hx, hy=hy, hz=hz,
                dt=dt, steps=steps, initial_state=initial_state, site=site,
                max_bond_dim=chi, threshold=threshold, numiter_lanczos=numiter_lanczos,
                adaptive_pad=adaptive_pad,
                truncation_mode=truncation_mode,
                track_bond_dims=true,
                measure_runtime=measure_runtime,
            )
            bond_dims[m] = bd
        else
            times, z, wall = run_method_expect_z_site(m;
                L=L,
                model=model,
                J=J, g=g, Delta=Delta, gamma=gamma,
                Jxx=Jxx, Jyy=Jyy, Jzz=Jzz, hx=hx, hy=hy, hz=hz,
                dt=dt, steps=steps, initial_state=initial_state, site=site,
                max_bond_dim=chi, threshold=threshold, numiter_lanczos=numiter_lanczos,
                adaptive_pad=adaptive_pad,
                truncation_mode=truncation_mode,
                measure_runtime=measure_runtime,
            )
        end
        @assert times == times_ref
        results[m] = z
        runtimes[m] = wall
        @printf("[exp_runs] %s runtime: %.3fs\n", m, wall) # informational only
        flush(stdout)
    end

    label_with_time(k::AbstractString) = @sprintf("%s %.2f s", k, runtimes[k])
    qutip_label = (reference == "qutip") ? @sprintf("exact (qutip) %.2f s", runtimes["qutip"]) : ""

    if plot
        # Force non-interactive backend (avoids GUI/event-loop hangs on macOS/headless runs).
        mpl = pyimport("matplotlib")
        mpl.use("Agg")
        plt = pyimport("matplotlib.pyplot")
        builtins = pyimport("builtins")
        nrows = (reference == "qutip") ? 4 : 3
        fig, axes = plt.subplots(nrows, 1, figsize=(12, (nrows == 4 ? 14 : 11)))

        function maybe_add_legend(ax)
            handles, labels = ax.get_legend_handles_labels()
            if pyconvert(Int, builtins.len(handles)) > 0
                ax.legend(loc="best")
            end
            return nothing
        end

        # 1) ⟨Z⟩ vs time: fixed family (1TDVP + BUG2nd fixed) + exact
        ax = axes[0]
        if reference == "qutip"
            ax.plot(times_ref, z_ref, "-", color="black", linewidth=2.0, alpha=0.85, label=qutip_label)
        end
        for m in fixed_methods
            haskey(results, m) || continue
            ax.plot(times_ref, results[m], linewidth=1.5, alpha=0.8, label=label_with_time(m))
        end
        ax.set_xlabel("Time")
        ax.set_ylabel("⟨Z⟩")
        tagstr = isempty(tag) ? "" : "  tag=$(tag)"
        ax.set_title("⟨Z⟩ vs time (fixed family: 1TDVP + BUG2nd fixed)  (site=$(site))  model=$(model)$(tagstr)  L=$(L), dt=$(dt), steps=$(steps), χpad=$(fixed_max_bond_dim)")
        ax.grid(true, alpha=0.3)
        maybe_add_legend(ax)

        # 2) ⟨Z⟩ vs time: adaptive family (2TDVP + BUG2nd adaptive) + exact
        ax2 = axes[1]
        if reference == "qutip"
            ax2.plot(times_ref, z_ref, "-", color="black", linewidth=2.0, alpha=0.85, label=qutip_label)
        end
        for m in adaptive_methods
            haskey(results, m) || continue
            ax2.plot(times_ref, results[m], linewidth=1.5, alpha=0.8, label=label_with_time(m))
        end
        ax2.set_xlabel("Time")
        ax2.set_ylabel("⟨Z⟩")
        ax2.set_title("⟨Z⟩ vs time (adaptive family: 2TDVP + BUG2nd adaptive)  (site=$(site))  model=$(model)$(tagstr)  L=$(L), dt=$(dt), steps=$(steps), χpad=$(adaptive_pad)")
        ax2.grid(true, alpha=0.3)
        maybe_add_legend(ax2)

        # 3) Bond dimension growth vs time: adaptive / 2TDVP methods
        ax_bd = axes[2]
        for m in adaptive_methods
            haskey(bond_dims, m) || continue
            ax_bd.plot(times_ref, bond_dims[m], linewidth=1.8, alpha=0.85, label=label_with_time(m))
        end
        ax_bd.set_xlabel("Time")
        ax_bd.set_ylabel("max χ")
        ax_bd.set_title("Bond dimension growth (adaptive / 2TDVP)  model=$(model)$(tagstr)  L=$(L), dt=$(dt), steps=$(steps), χmax=$(max_bond_dim), trunc=$(threshold)")
        ax_bd.grid(true, alpha=0.3)
        maybe_add_legend(ax_bd)

        if reference == "qutip"
            # 4) Squared error vs exact (qutip): all methods
            ax3 = axes[3]
            for m in methods
                haskey(results, m) || continue
                err_sq = abs2.(results[m] .- z_ref)
                ax3.plot(times_ref, err_sq, linewidth=1.5, alpha=0.8, label=label_with_time(m))
            end
            ax3.set_xlabel("Time")
            ax3.set_ylabel("|Δ⟨Z⟩|²")
            ax3.set_title("Squared error vs exact (qutip): all methods")
            ax3.set_yscale("log")
            ax3.grid(true, alpha=0.3)
            maybe_add_legend(ax3)
        end

        fig.tight_layout()
        outfile = joinpath(outdir, "runs_comparison.png")
        fig.savefig(outfile, dpi=160, bbox_inches="tight")
        plt.close(fig)
        @printf("[exp_runs] wrote %s\n", outfile)
        flush(stdout)
    end

    # Save timeseries CSV (and a small metadata file) for post-analysis.
    csvfile = joinpath(outdir, "timeseries.csv")
    keys_sorted = sort!(collect(keys(results)))
    open(csvfile, "w") do io
        header = String[]
        push!(header, "t")
        if reference == "qutip"
            push!(header, "z_ref_qutip")
        end
        append!(header, keys_sorted)
        println(io, join(header, ","))

        for idx in eachindex(times_ref)
            vals = String[]
            push!(vals, @sprintf("%.16g", times_ref[idx]))
            if reference == "qutip"
                push!(vals, @sprintf("%.16g", z_ref[idx]))
            end
            for k in keys_sorted
                push!(vals, @sprintf("%.16g", results[k][idx]))
            end
            println(io, join(vals, ","))
        end
    end
    @printf("[exp_runs] wrote %s\n", csvfile)
    flush(stdout)

    metafile = joinpath(outdir, "meta.txt")
    open(metafile, "w") do io
        println(io, "tag=$(tag)")
        println(io, "model=$(model)")
        println(io, "reference=$(reference)")
        println(io, "L=$(L)")
        println(io, "site=$(site)")
        println(io, "dt=$(dt)")
        println(io, "steps=$(steps)")
        println(io, "initial_state=$(initial_state)")
        println(io, "max_bond_dim=$(max_bond_dim)")
        println(io, "fixed_max_bond_dim=$(fixed_max_bond_dim)")
        println(io, "adaptive_pad=$(adaptive_pad)")
        println(io, "truncation_mode=$(truncation_mode)")
        println(io, "threshold=$(threshold)")
        println(io, "numiter_lanczos=$(numiter_lanczos)")
        println(io, "J=$(J)")
        println(io, "g=$(g)")
        println(io, "Delta=$(Delta)")
        println(io, "gamma=$(gamma)")
        println(io, "Jxx=$(Jxx)")
        println(io, "Jyy=$(Jyy)")
        println(io, "Jzz=$(Jzz)")
        println(io, "hx=$(hx)")
        println(io, "hy=$(hy)")
        println(io, "hz=$(hz)")
        println(io, "")
        println(io, "[runtimes_s]")
        for k in sort!(collect(keys(runtimes)))
            @printf(io, "%s=%.6f\n", k, runtimes[k])
        end
    end
    @printf("[exp_runs] wrote %s\n", metafile)
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module BUGExpRuns

