using CSV
using DataFrames
using PythonCall
using Printf

function _setup_matplotlib_lr()
    mpl = pyimport("matplotlib")
    mpl.use("Agg")
    return pyimport("matplotlib.pyplot")
end

function _parse_longrange_files(results_dir::String)
    pat = r"^timeseries_lambda([0-9eE\+\-\.]+)_D(\d+)\.csv$"
    files = readdir(results_dir)
    grouped = Dict{Float64, Dict{Int, DataFrame}}()
    for f in files
        m = match(pat, f)
        m === nothing && continue
        lam = parse(Float64, m.captures[1])
        D = parse(Int, m.captures[2])
        df = DataFrame(CSV.File(joinpath(results_dir, f)))
        if !haskey(grouped, lam)
            grouped[lam] = Dict{Int, DataFrame}()
        end
        grouped[lam][D] = df
    end
    isempty(grouped) && error("No timeseries_lambda*_D*.csv files found in $results_dir")
    return grouped
end

function _center_site(df::DataFrame)
    nsites = count(nm -> startswith(String(nm), "z_ref_"), names(df))
    nsites > 0 || error("Could not infer number of sites from z_ref_* columns.")
    return Int(cld(nsites, 2))
end

function _lambda_tag(lam::Float64)
    s = @sprintf("%.6f", lam)
    s = rstrip(rstrip(s, '0'), '.')
    isempty(s) && (s = "0")
    return s
end

function plot_longrange_results(; results_dir::String=joinpath(@__DIR__, "results_longrange"))
    grouped = _parse_longrange_files(results_dir)
    plt = _setup_matplotlib_lr()

    lambdas = sort(collect(keys(grouped)))
    final_err_by_lambda = Dict{Float64, Dict{Int, Float64}}()

    for lam in lambdas
        data = grouped[lam]
        Ds = sort(collect(keys(data)))
        df_ref = data[first(Ds)]
        t = Vector{Float64}(df_ref.t)
        center = _center_site(df_ref)
        tag = _lambda_tag(lam)

        # 1) Norms
        fig1, ax1 = plt.subplots(figsize=(8.0, 5.0))
        ax1.plot(t, Vector{Float64}(df_ref.norm_ref), "k-", linewidth=2.0, label="Exact norm")
        for D in Ds
            df = data[D]
            ax1.plot(t, Vector{Float64}(df.norm_tdvp), linewidth=1.8, label="TDVP norm (D=$(D))")
        end
        ax1.set_xlabel("t")
        ax1.set_ylabel("||psi||")
        ax1.set_title("Long-range model: norms (lambda=$(lam))")
        ax1.grid(true, alpha=0.3)
        ax1.legend(loc="best")
        fig1.tight_layout()
        f1 = joinpath(results_dir, "comparison_norms_lambda$(tag).png")
        fig1.savefig(f1, dpi=180)
        plt.close(fig1)

        # 2) State error
        fig2, ax2 = plt.subplots(figsize=(8.0, 5.0))
        for D in Ds
            df = data[D]
            ax2.plot(t, Vector{Float64}(df.err_state), linewidth=1.8, label="||psi_tdvp-psi_ref|| (D=$(D))")
        end
        ax2.set_yscale("log")
        ax2.set_xlabel("t")
        ax2.set_ylabel("state error")
        ax2.set_title("State error vs exact (lambda=$(lam))")
        ax2.grid(true, alpha=0.3)
        ax2.legend(loc="best")
        fig2.tight_layout()
        f2 = joinpath(results_dir, "comparison_state_error_lambda$(tag).png")
        fig2.savefig(f2, dpi=180)
        plt.close(fig2)

        # 3) Center-site normalized Z
        zref_col = Symbol("znorm_ref_$(center)")
        ztdvp_col = Symbol("znorm_tdvp_$(center)")
        fig3, ax3 = plt.subplots(figsize=(8.0, 5.0))
        ax3.plot(t, Vector{Float64}(df_ref[!, zref_col]), "k-", linewidth=2.0, label="Exact <Z_$(center)>_norm")
        for D in Ds
            df = data[D]
            ax3.plot(t, Vector{Float64}(df[!, ztdvp_col]), linewidth=1.8, label="TDVP <Z_$(center)>_norm (D=$(D))")
        end
        ax3.set_xlabel("t")
        ax3.set_ylabel("normalized <Z_center>")
        ax3.set_title("Center-site normalized magnetization (lambda=$(lam))")
        ax3.grid(true, alpha=0.3)
        ax3.legend(loc="best")
        fig3.tight_layout()
        f3 = joinpath(results_dir, "comparison_z_center_norm_lambda$(tag).png")
        fig3.savefig(f3, dpi=180)
        plt.close(fig3)

        # 4) Max local error
        fig4, ax4 = plt.subplots(figsize=(8.0, 5.0))
        for D in Ds
            df = data[D]
            ax4.plot(t, Vector{Float64}(df.max_abs_err_z), linewidth=1.8, label="max_j |Δ<Z_j>| (D=$(D))")
        end
        ax4.set_yscale("log")
        ax4.set_xlabel("t")
        ax4.set_ylabel("max local observable error")
        ax4.set_title("Observable error summary (lambda=$(lam))")
        ax4.grid(true, alpha=0.3)
        ax4.legend(loc="best")
        fig4.tight_layout()
        f4 = joinpath(results_dir, "comparison_max_z_error_lambda$(tag).png")
        fig4.savefig(f4, dpi=180)
        plt.close(fig4)

        # Collect final errors for aggregate plots
        final_err_by_lambda[lam] = Dict{Int, Float64}()
        for D in Ds
            final_err_by_lambda[lam][D] = Vector{Float64}(data[D].err_state)[end]
        end
    end

    # 5) Final error vs lambda for each D
    all_Ds = sort(unique(vcat([collect(keys(grouped[lam])) for lam in lambdas]...)))
    fig5, ax5 = plt.subplots(figsize=(8.0, 5.0))
    for D in all_Ds
        ys = Float64[]
        xs = Float64[]
        for lam in lambdas
            if haskey(final_err_by_lambda[lam], D)
                push!(xs, lam)
                push!(ys, final_err_by_lambda[lam][D])
            end
        end
        if !isempty(xs)
            ax5.plot(xs, ys, marker="o", linewidth=1.8, label="D=$(D)")
        end
    end
    ax5.set_yscale("log")
    ax5.set_xlabel("lambda")
    ax5.set_ylabel("final state error at t=tmax")
    ax5.set_title("Final error vs lambda")
    ax5.grid(true, alpha=0.3)
    ax5.legend(loc="best")
    fig5.tight_layout()
    f5 = joinpath(results_dir, "final_error_vs_lambda.png")
    fig5.savefig(f5, dpi=180)
    plt.close(fig5)

    # 6) Final error vs D for each lambda
    fig6, ax6 = plt.subplots(figsize=(8.0, 5.0))
    for lam in lambdas
        Ds = sort(collect(keys(final_err_by_lambda[lam])))
        ys = [final_err_by_lambda[lam][D] for D in Ds]
        ax6.plot(Ds, ys, marker="o", linewidth=1.8, label="lambda=$(lam)")
    end
    ax6.set_yscale("log")
    ax6.set_xlabel("bond dimension D")
    ax6.set_ylabel("final state error at t=tmax")
    ax6.set_title("Final error vs bond dimension")
    ax6.grid(true, alpha=0.3)
    ax6.legend(loc="best")
    fig6.tight_layout()
    f6 = joinpath(results_dir, "final_error_vs_D.png")
    fig6.savefig(f6, dpi=180)
    plt.close(fig6)

    println("Wrote long-range plots in ", results_dir)
end

plot_longrange_results()
