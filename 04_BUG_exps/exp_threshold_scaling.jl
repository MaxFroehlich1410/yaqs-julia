module BUGExpThresholdScaling

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
    g = parse(Float64, get(kv, "g", "0.7"))
    dt = parse(Float64, get(kv, "dt", "0.1"))
    steps = parse(Int, get(kv, "steps", "40"))
    initial_state = get(kv, "initial_state", "x+")
    max_bond_dim = parse(Int, get(kv, "max_bond_dim", "64"))
    numiter_lanczos = parse(Int, get(kv, "numiter_lanczos", "25"))
    site = parse(Int, get(kv, "site", string(mid_site(L))))
    truncation_mode = Symbol(lowercase(get(kv, "truncation_mode", "during")))

    # Default: 4 BUG variants + TDVP baselines (all compared to qutip for accuracy).
    methods_str = get(kv, "methods", "FIXED,ADAPTIVE,DOUBLEFIXED,DOUBLEADAPTIVE,SINGLE_SITE_TDVP,TWO_SITE_TDVP")
    methods = split(methods_str, ",")

    thr_str = get(kv, "thresholds", "1e-3,1e-6,1e-9,1e-12,1e-14")
    thresholds = parse.(Float64, split(thr_str, ","))

    times_ref, z_ref = qutip_expect_z_site(L; J=J, g=g, dt=dt, steps=steps,
                                          initial_state=initial_state, site=site)

    errs = Dict{String, Vector{Float64}}()
    ymin = Dict{String, Vector{Float64}}()
    ymax = Dict{String, Vector{Float64}}()
    for m in methods
        errs[m] = Float64[]
        ymin[m] = Float64[]
        ymax[m] = Float64[]
    end

    for thr in thresholds
        @printf("[exp_threshold_scaling] threshold=%.1e\n", thr)
        for m in methods
            _, z, _ = run_method_expect_z_site(m;
                L=L, J=J, g=g, dt=dt, steps=steps, initial_state=initial_state, site=site,
                max_bond_dim=max_bond_dim, threshold=thr, numiter_lanczos=numiter_lanczos,
                truncation_mode=truncation_mode,
            )
            push!(errs[m], rms_error(z, z_ref; skip=2))
            push!(ymin[m], min_abs_error(z, z_ref; skip=2))
            push!(ymax[m], max_abs_error(z, z_ref; skip=2))
        end
    end

    np = pyimport("numpy")
    plt = pyimport("matplotlib.pyplot")
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for m in methods
        y = errs[m]
        lo = ymin[m]
        hi = ymax[m]
        yerr = np.vstack((lo, hi)) # shape (2, n)
        ax.errorbar(thresholds, y, yerr=yerr, capsize=5, marker="o", linewidth=1.5, alpha=0.8, label=m)
    end
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Truncation threshold")
    ax.set_ylabel("RMS |Δ⟨Z⟩| vs exact (qutip)")
    ax.set_title("Scaling vs truncation_threshold  (site=$(site))  L=$(L), dt=$(dt), steps=$(steps), χmax=$(max_bond_dim)")
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    outfile = joinpath(outdir, "threshold_scaling.png")
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)
    @printf("[exp_threshold_scaling] wrote %s\n", outfile)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module BUGExpThresholdScaling

