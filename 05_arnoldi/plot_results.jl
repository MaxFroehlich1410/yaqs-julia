using CSV
using DataFrames
using Printf
using PythonCall

function _load_timeseries(results_dir::String)
    files = sort(filter(f -> occursin(r"^timeseries_D\d+\.csv$", f), readdir(results_dir)))
    isempty(files) && error("No timeseries_D*.csv files found in $results_dir. Run the benchmark first.")

    data = Dict{Int, DataFrame}()
    for f in files
        m = match(r"^timeseries_D(\d+)\.csv$", f)
        m === nothing && continue
        D = parse(Int, m.captures[1])
        df = DataFrame(CSV.File(joinpath(results_dir, f)))
        data[D] = df
    end
    return sort(collect(keys(data))), data
end

function _center_site_index(df::DataFrame)
    nsites = 0
    for cname in names(df)
        s = String(cname)
        if startswith(s, "z_ref_")
            nsites += 1
        end
    end
    nsites > 0 || error("Could not infer number of sites from z_ref_* columns.")
    return Int(cld(nsites, 2))
end

function _setup_matplotlib()
    mpl = pyimport("matplotlib")
    mpl.use("Agg")
    plt = pyimport("matplotlib.pyplot")
    return plt
end

function plot_nonhermitian_results(; results_dir::String=joinpath(@__DIR__, "results"))
    Ds, data = _load_timeseries(results_dir)
    plt = _setup_matplotlib()

    # Use smallest D file to get exact reference curves (identical across D files).
    df_ref = data[first(Ds)]
    t = Vector{Float64}(df_ref.t)

    # ---- Plot 1: norms ----
    fig1, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax1.plot(t, Vector{Float64}(df_ref.norm_ref), "k-", linewidth=2.0, label="Exact norm")
    for D in Ds
        df = data[D]
        ax1.plot(t, Vector{Float64}(df.norm_tdvp), linewidth=1.8, label="TDVP Arnoldi norm (D=$(D))")
    end
    ax1.set_xlabel("t")
    ax1.set_ylabel("||psi||")
    ax1.set_title("Non-Hermitian evolution: state norm")
    ax1.grid(true, alpha=0.3)
    ax1.legend(loc="best")
    fig1.tight_layout()
    f1 = joinpath(results_dir, "comparison_norms.png")
    fig1.savefig(f1, dpi=180)
    plt.close(fig1)

    # ---- Plot 2: errors ----
    fig2, ax2 = plt.subplots(figsize=(8.0, 5.0))
    for D in Ds
        df = data[D]
        ax2.plot(t, Vector{Float64}(df.err_state), linewidth=1.8, label="||psi_tdvp-psi_ref|| (D=$(D))")
    end
    ax2.set_yscale("log")
    ax2.set_xlabel("t")
    ax2.set_ylabel("state error")
    ax2.set_title("State-vector error vs exact")
    ax2.grid(true, alpha=0.3)
    ax2.legend(loc="best")
    fig2.tight_layout()
    f2 = joinpath(results_dir, "comparison_state_error.png")
    fig2.savefig(f2, dpi=180)
    plt.close(fig2)

    # ---- Plot 3: local observable comparison at center site ----
    center = _center_site_index(df_ref)
    zref_col = Symbol("z_ref_$(center)")
    ztdvp_col = Symbol("z_tdvp_$(center)")

    fig3, ax3 = plt.subplots(figsize=(8.0, 5.0))
    ax3.plot(t, Vector{Float64}(df_ref[!, zref_col]), "k-", linewidth=2.0, label="Exact <Z_$(center)>")
    for D in Ds
        df = data[D]
        ax3.plot(t, Vector{Float64}(df[!, ztdvp_col]), linewidth=1.8, label="TDVP Arnoldi <Z_$(center)> (D=$(D))")
    end
    ax3.set_xlabel("t")
    ax3.set_ylabel("<Z_center>")
    ax3.set_title("Center-site magnetization comparison")
    ax3.grid(true, alpha=0.3)
    ax3.legend(loc="best")
    fig3.tight_layout()
    f3 = joinpath(results_dir, "comparison_z_center.png")
    fig3.savefig(f3, dpi=180)
    plt.close(fig3)

    # ---- Plot 4: max local observable error ----
    fig4, ax4 = plt.subplots(figsize=(8.0, 5.0))
    for D in Ds
        df = data[D]
        ax4.plot(t, Vector{Float64}(df.max_abs_err_z), linewidth=1.8, label="max_j |Δ<Z_j>| (D=$(D))")
    end
    ax4.set_yscale("log")
    ax4.set_xlabel("t")
    ax4.set_ylabel("max local observable error")
    ax4.set_title("Observable error summary")
    ax4.grid(true, alpha=0.3)
    ax4.legend(loc="best")
    fig4.tight_layout()
    f4 = joinpath(results_dir, "comparison_max_z_error.png")
    fig4.savefig(f4, dpi=180)
    plt.close(fig4)

    println("Wrote plots:")
    println(" - ", f1)
    println(" - ", f2)
    println(" - ", f3)
    println(" - ", f4)
end

plot_nonhermitian_results()
