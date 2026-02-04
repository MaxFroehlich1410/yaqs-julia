module BUGExpRuns

using PythonCall
using Printf

include("exp_util.jl")
using .BUGExpUtil

export main

function main(args=ARGS)
    kv = parse_kv_args(args)

    outdir = get(kv, "outdir", joinpath(@__DIR__, "results"))
    mkpath(outdir)

    L = parse(Int, get(kv, "L", "8"))
    J = parse(Float64, get(kv, "J", "1.0"))
    g = parse(Float64, get(kv, "g", "0.5"))
    dt = parse(Float64, get(kv, "dt", "0.1"))
    steps = parse(Int, get(kv, "steps", "40"))
    initial_state = get(kv, "initial_state", "x+")
    max_bond_dim = parse(Int, get(kv, "max_bond_dim", "128"))
    # Fixed-method fairness: enforce a single shared padded bond dimension
    # across {1TDVP, fixed BUG 1st, fixed BUG 2nd}.
    fixed_max_bond_dim = parse(Int, get(kv, "fixed_max_bond_dim", string(max_bond_dim)))
    threshold = parse(Float64, get(kv, "threshold", "1e-12"))
    numiter_lanczos = parse(Int, get(kv, "numiter_lanczos", "25"))
    site = parse(Int, get(kv, "site", string(mid_site(L))))

    # Default: the 4 BUG variants + both TDVP baselines (all compared to qutip for accuracy).
    methods_str = get(kv, "methods", "FIXED,ADAPTIVE,DOUBLEFIXED,DOUBLEADAPTIVE,SINGLE_SITE_TDVP,TWO_SITE_TDVP")
    methods = split(methods_str, ",")

    # Exact reference: qutip
    runtimes = Dict{String, Float64}()
    t_ref0 = time()
    times_ref, z_ref = qutip_expect_z_site(L; J=J, g=g, dt=dt, steps=steps,
                                          initial_state=initial_state, site=site)
    runtimes["qutip"] = time() - t_ref0

    # Run each method and collect results.
    results = Dict{String, Vector{Float64}}()
    for m in methods
        is_fixed_family = (m == "SINGLE_SITE_TDVP") || (m == "FIXED") || (m == "DOUBLEFIXED")
        chi = is_fixed_family ? fixed_max_bond_dim : max_bond_dim
        times, z, wall = run_method_expect_z_site(m;
            L=L, J=J, g=g, dt=dt, steps=steps, initial_state=initial_state, site=site,
            max_bond_dim=chi, threshold=threshold, numiter_lanczos=numiter_lanczos,
        )
        @assert times == times_ref
        results[m] = z
        runtimes[m] = wall
        @printf("[exp_runs] %s runtime: %.3fs\n", m, wall) # informational only
    end

    label_with_time(k::AbstractString) = @sprintf("%s %.2f s", k, runtimes[k])
    qutip_label = @sprintf("exact (qutip) %.2f s", runtimes["qutip"])

    plt = pyimport("matplotlib.pyplot")
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    fixed_methods = ["SINGLE_SITE_TDVP", "FIXED", "DOUBLEFIXED"]
    adaptive_methods = ["TWO_SITE_TDVP", "ADAPTIVE", "DOUBLEADAPTIVE"]

    # 1) ⟨Z⟩ vs time: fixed family + 1TDVP + exact
    ax = axes[0]
    ax.plot(times_ref, z_ref, "-", color="black", linewidth=2.0, alpha=0.85, label=qutip_label)
    for m in fixed_methods
        haskey(results, m) || continue
        ax.plot(times_ref, results[m], linewidth=1.5, alpha=0.8, label=label_with_time(m))
    end
    ax.set_xlabel("Time")
    ax.set_ylabel("⟨Z⟩")
    ax.set_title("⟨Z⟩ vs time (fixed family + 1TDVP + exact)  (site=$(site))  L=$(L), J=$(J), g=$(g), dt=$(dt), steps=$(steps), χpad=$(fixed_max_bond_dim)")
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best")

    # 2) ⟨Z⟩ vs time: adaptive family + 2TDVP + exact
    ax2 = axes[1]
    ax2.plot(times_ref, z_ref, "-", color="black", linewidth=2.0, alpha=0.85, label=qutip_label)
    for m in adaptive_methods
        haskey(results, m) || continue
        ax2.plot(times_ref, results[m], linewidth=1.5, alpha=0.8, label=label_with_time(m))
    end
    ax2.set_xlabel("Time")
    ax2.set_ylabel("⟨Z⟩")
    ax2.set_title("⟨Z⟩ vs time (adaptive family + 2TDVP + exact)  (site=$(site))  L=$(L), J=$(J), g=$(g), dt=$(dt), steps=$(steps)")
    ax2.grid(true, alpha=0.3)
    ax2.legend(loc="best")

    # 3) Squared error vs exact (qutip): all methods
    ax3 = axes[2]
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
    ax3.legend(loc="best")

    fig.tight_layout()
    outfile = joinpath(outdir, "runs_comparison.png")
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)
    @printf("[exp_runs] wrote %s\n", outfile)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module BUGExpRuns

