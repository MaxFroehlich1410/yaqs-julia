using CSV
using DataFrames
using PythonCall

function _setup_plt_bug()
    mpl = pyimport("matplotlib")
    mpl.use("Agg")
    return pyimport("matplotlib.pyplot")
end

function _collect_files_bug(results_dir::String)
    files = filter(f -> occursin(r"^timeseries_.*_D\d+\.csv$", f), readdir(results_dir))
    isempty(files) && error("No comparison CSV files found in $results_dir.")
    parsed = Tuple{String, Int, String}[]
    for f in files
        m = match(r"^timeseries_(.+)_D(\d+)\.csv$", f)
        m === nothing && continue
        push!(parsed, (m.captures[1], parse(Int, m.captures[2]), f))
    end
    return parsed
end

function plot_bug2_vs_tdvp(; results_dir::String=joinpath(@__DIR__, "results_bug2_compare"))
    parsed = _collect_files_bug(results_dir)
    plt = _setup_plt_bug()

    models = sort(unique([p[1] for p in parsed]))
    Ds = sort(unique([p[2] for p in parsed]))

    # Final error vs D per model
    fig1, ax1 = plt.subplots(figsize=(8.0, 5.0))
    fig2, ax2 = plt.subplots(figsize=(8.0, 5.0))
    for model in models
        xs = Int[]
        y_tdvp = Float64[]
        y_bug = Float64[]
        yn_tdvp = Float64[]
        yn_bug = Float64[]
        for D in Ds
            rec = findfirst(p -> p[1] == model && p[2] == D, parsed)
            rec === nothing && continue
            df = DataFrame(CSV.File(joinpath(results_dir, parsed[rec][3])))
            push!(xs, D)
            push!(y_tdvp, Vector{Float64}(df.err_state_tdvp)[end])
            push!(y_bug, Vector{Float64}(df.err_state_bug)[end])
            push!(yn_tdvp, Vector{Float64}(df.err_state_normed_tdvp)[end])
            push!(yn_bug, Vector{Float64}(df.err_state_normed_bug)[end])
        end
        ax1.plot(xs, y_tdvp, marker="o", linewidth=1.8, label="$(model) TDVP")
        ax1.plot(xs, y_bug, marker="s", linestyle="--", linewidth=1.8, label="$(model) BUG2")
        ax2.plot(xs, yn_tdvp, marker="o", linewidth=1.8, label="$(model) TDVP")
        ax2.plot(xs, yn_bug, marker="s", linestyle="--", linewidth=1.8, label="$(model) BUG2")
    end
    ax1.set_yscale("log")
    ax1.set_xlabel("bond dimension D")
    ax1.set_ylabel("final raw state error")
    ax1.set_title("Final raw state error: TDVP vs BUG2")
    ax1.grid(true, alpha=0.3)
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(joinpath(results_dir, "bug2_vs_tdvp_final_error_raw.png"), dpi=180)
    plt.close(fig1)

    ax2.set_yscale("log")
    ax2.set_xlabel("bond dimension D")
    ax2.set_ylabel("final normalized-direction error")
    ax2.set_title("Final normalized error: TDVP vs BUG2")
    ax2.grid(true, alpha=0.3)
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(joinpath(results_dir, "bug2_vs_tdvp_final_error_normed.png"), dpi=180)
    plt.close(fig2)

    # Time traces at largest D per model
    for model in models
        Dmax = maximum([p[2] for p in parsed if p[1] == model])
        rec = findfirst(p -> p[1] == model && p[2] == Dmax, parsed)
        rec === nothing && continue
        df = DataFrame(CSV.File(joinpath(results_dir, parsed[rec][3])))
        t = Vector{Float64}(df.t)
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.plot(t, Vector{Float64}(df.err_state_tdvp), linewidth=2.0, label="TDVP raw error")
        ax.plot(t, Vector{Float64}(df.err_state_bug), linewidth=2.0, linestyle="--", label="BUG2 raw error")
        ax.plot(t, Vector{Float64}(df.err_state_normed_tdvp), linewidth=1.6, label="TDVP normed error")
        ax.plot(t, Vector{Float64}(df.err_state_normed_bug), linewidth=1.6, linestyle="--", label="BUG2 normed error")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("state error")
        ax.set_title("Error vs time, model=$(model), D=$(Dmax)")
        ax.grid(true, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(joinpath(results_dir, "bug2_vs_tdvp_error_timeseries_$(model)_D$(Dmax).png"), dpi=180)
        plt.close(fig)
    end

    println("Wrote BUG2-vs-TDVP plots to ", results_dir)
end

plot_bug2_vs_tdvp()
