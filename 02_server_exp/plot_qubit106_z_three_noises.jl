"""
Plot ⟨Z⟩(t) for a single qubit across 3 IBM127 kicked-Ising simulations.

This script is designed for the output folders produced by `02_server_exp/ibm_127_exp.jl`,
which save Python `*.pkl` files via PythonCall. Those pickles typically require the
`juliacall` Python module at load time, so we load them through Julia's `PythonCall`.

### What it plots
- ⟨Z⟩ expectation value (from `local_expvals_by_method["Julia Projector"]`)
- for IBM qubit **106** by default (IBM-style 0-based indexing)
- for noise strengths: 0.1, 0.01, 0.001

### Usage
From repo root:

    julia 02_server_exp/plot_qubit106_z_three_noises.jl

Optional arguments:
    julia 02_server_exp/plot_qubit106_z_three_noises.jl 106 ibm0
    julia 02_server_exp/plot_qubit106_z_three_noises.jl 107 onebased

Indexing modes:
- `ibm0`: interpret qubit as 0-based (IBM labels 0..126)  => row = qubit + 1
- `onebased`: interpret qubit as 1-based (Julia row index) => row = qubit
"""

import Pkg
# Ensure we use this repo's environment (so `PythonCall`, `CSV`, `DataFrames`, ... resolve).
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using PythonCall
using CSV, DataFrames

const _pickle = pyimport("pickle")
const _pybuiltins = pyimport("builtins")

function _haskey_py(d, k::AbstractString)::Bool
    return pytruth(d.__contains__(k))
end

function _find_matching_dir(base::AbstractString, required_substrings::Vector{String})
    for entry in readdir(base)
        p = joinpath(base, entry)
        isdir(p) || continue
        ok = true
        @inbounds for s in required_substrings
            if !occursin(s, entry)
                ok = false
                break
            end
        end
        ok && return p
    end
    return nothing
end

function _find_main_pkl(dir::AbstractString)
    # Prefer the "main" pickle (not the per-batch pickles).
    candidates = String[]
    for fn in readdir(dir)
        endswith(fn, ".pkl") || continue
        occursin("_Julia_Projector_batch", fn) && continue
        push!(candidates, fn)
    end
    isempty(candidates) && return nothing
    # If there are multiple, pick the largest-name match deterministically.
    sort!(candidates)
    return joinpath(dir, last(candidates))
end

function _load_pickle_dict(path::AbstractString)
    pyf = _pybuiltins.open(path, "rb")
    d = _pickle.load(pyf)
    pyf.close()
    return d
end

function _extract_times_and_z(d; method_name::String="Julia Projector", qubit_row::Int)
    _haskey_py(d, "times") || error("pickle missing key \"times\"")
    _haskey_py(d, "local_expvals_by_method") || error("pickle missing key \"local_expvals_by_method\"")

    times = pyconvert(Vector{Float64}, d["times"])
    local_by_method = d["local_expvals_by_method"]

    pytruth(local_by_method.__contains__(method_name)) || error("method \"$method_name\" not found in local_expvals_by_method")
    mat = pyconvert(Matrix{Float64}, local_by_method[method_name])  # (qubits × time)

    1 ≤ qubit_row ≤ size(mat, 1) || error("qubit_row=$qubit_row out of bounds for matrix with size $(size(mat))")
    z = @view mat[qubit_row, :]
    return times, z
end

function _parse_qubit_args()
    qubit = isempty(ARGS) ? 106 : parse(Int, ARGS[1])
    mode = length(ARGS) ≥ 2 ? lowercase(ARGS[2]) : "ibm0"
    if mode == "ibm0"
        return qubit + 1, qubit, mode
    elseif mode == "onebased"
        return qubit, qubit, mode
    else
        error("Unknown indexing mode \"$mode\". Use \"ibm0\" or \"onebased\".")
    end
end

function main()
    base = joinpath(@__DIR__, "CTJM_interesting")
    isdir(base) || error("Base directory not found: $base")

    qubit_row, qubit_label, mode = _parse_qubit_args()

    # Auto-resolve directories by required substrings (robust to thetaH variations).
    dir_noise_0p1 = _find_matching_dir(base, [
        "unraveling_eff_N127_L5_tau0p1",
        "noise0p1_",
        "basisIBM127_kicked_ising",
        "obsZ_locTDVP_lrTDVP",
        "thetaJ5pi2",
        "ErrXYZ_traj200",
    ])
    dir_noise_0p01 = _find_matching_dir(base, [
        "unraveling_eff_N127_L5_tau0p1",
        "noise0p01_",
        "basisIBM127_kicked_ising",
        "obsZ_locTDVP_lrTDVP",
        "thetaJ5pi2",
        "ErrXYZ_traj200",
    ])
    dir_noise_0p001 = _find_matching_dir(base, [
        "unraveling_eff_N127_L5_tau0p1",
        "noise0p001_",
        "basisIBM127_kicked_ising",
        "obsZ_locTDVP_lrTDVP",
        "thetaJ5pi2",
        "ErrXYZ_traj200",
    ])

    for (noise, dpath) in [("0.1", dir_noise_0p1), ("0.01", dir_noise_0p01), ("0.001", dir_noise_0p001)]
        isnothing(dpath) && error("Could not find directory for noise=$noise under $base")
    end

    pkl_0p1 = _find_main_pkl(dir_noise_0p1)
    pkl_0p01 = _find_main_pkl(dir_noise_0p01)
    pkl_0p001 = _find_main_pkl(dir_noise_0p001)
    for (noise, p) in [("0.1", pkl_0p1), ("0.01", pkl_0p01), ("0.001", pkl_0p001)]
        isnothing(p) && error("Could not find main .pkl for noise=$noise")
    end

    d1 = _load_pickle_dict(pkl_0p1)
    d2 = _load_pickle_dict(pkl_0p01)
    d3 = _load_pickle_dict(pkl_0p001)

    t1, z1 = _extract_times_and_z(d1; qubit_row=qubit_row)
    t2, z2 = _extract_times_and_z(d2; qubit_row=qubit_row)
    t3, z3 = _extract_times_and_z(d3; qubit_row=qubit_row)

    # Plot with matplotlib (same plotting backend used by ibm_127_exp.jl).
    plt = try
        pyimport("matplotlib.pyplot")
    catch err
        # Fallback: always at least save the time series as CSV.
        @warn "Python matplotlib not available in this Julia PythonCall environment; writing CSV instead. Install matplotlib (in the same Python env used by PythonCall) to get a PNG plot." err
        out_csv = joinpath(base, "qubit$(qubit_label)_z_three_noises.csv")
        df = DataFrame(
            time = collect(t1),
            z_noise_0p1 = collect(z1),
            z_noise_0p01 = collect(z2),
            z_noise_0p001 = collect(z3),
        )
        CSV.write(out_csv, df)
        println("Saved: $out_csv")
        return
    end
    plt.close("all")

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(t1, collect(z1), label="noise=0.1", linewidth=2)
    ax.plot(t2, collect(z2), label="noise=0.01", linewidth=2)
    ax.plot(t3, collect(z3), label="noise=0.001", linewidth=2)

    ax.set_xlabel("time")
    ax.set_ylabel("<Z>")
    ax.set_title("IBM127 kicked Ising: <Z>(t) for qubit $(qubit_label) (mode=$(mode))")
    ax.grid(true)
    ax.legend(loc="best")

    out_png = joinpath(base, "qubit$(qubit_label)_z_three_noises.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    println("Saved: $out_png")
end

main()


