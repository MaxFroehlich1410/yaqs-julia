#!/usr/bin/env julia

using DelimitedFiles
using PythonCall

function main()
    # Load CSV (written by exp_runs.jl)
    csvfile = joinpath(@__DIR__, "timeseries.csv")
    header = split(readline(csvfile), ",")
    data = readdlm(csvfile, ',', Float64; skipstart=1)

    # Columns
    col = Dict{String,Int}(name => i for (i, name) in enumerate(header))
    @assert haskey(col, "t")
    @assert haskey(col, "z_ref_qutip") "Expected qutip reference column in timeseries.csv"
    @assert haskey(col, "DOUBLEADAPTIVE") "Expected DOUBLEADAPTIVE column in timeseries.csv"

    t = data[:, col["t"]]
    zref = data[:, col["z_ref_qutip"]]

    # Matplotlib non-interactive backend
    mpl = pyimport("matplotlib")
    mpl.use("Agg")
    plt = pyimport("matplotlib.pyplot")

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))

    # Plot squared errors for all methods, but:
    # - For DOUBLEADAPTIVE we "shift left by one timestep" by dropping its first entry:
    #   compare z_bug[k+1] against z_ref[k] and plot vs t[k] (k=1..end-1).
    for name in header
        if name in ("t", "z_ref_qutip")
            continue
        end
        z = data[:, col[name]]
        if name == "DOUBLEADAPTIVE"
            # drop first BUG point, drop last reference point to keep alignment
            err_sq = (z[2:end] .- zref[1:end-1]).^2
            ax.plot(t[1:end-1], err_sq, linewidth=1.6, alpha=0.85, label="$(name) (dropped first sample)")
        else
            err_sq = (z .- zref).^2
            ax.plot(t, err_sq, linewidth=1.2, alpha=0.65, label=name)
        end
    end

    ax.set_xlabel("Time")
    ax.set_ylabel("|Δ⟨Z⟩|²")
    ax.set_title("Squared error vs exact (qutip): DOUBLEADAPTIVE shifted left by 1 step")
    ax.set_yscale("log")
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    outfile = joinpath(@__DIR__, "se_drop_first_bug.png")
    fig.savefig(outfile, dpi=170, bbox_inches="tight")
    plt.close(fig)
    println("wrote ", outfile)
    return nothing
end

main()

