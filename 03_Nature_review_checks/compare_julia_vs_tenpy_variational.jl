using Printf
using DelimitedFiles
using TensorOperations

using Yaqs
using Yaqs.GateLibrary
using Yaqs.MPSModule
using Yaqs.MPOModule
using Yaqs.CircuitLibrary
using Yaqs.DigitalTJM: DigitalCircuit, add_gate!

include("gatelist_io.jl")
using .GateListIO: write_gatelist_csv

function _measure_z_sites(psi::MPS, sites::Vector{Int})
    # IMPORTANT: `evaluate_all_local_expectations` shifts the orthogonality center and
    # mutates the MPS tensors (gauge changes). For a fair circuit-evolution comparison
    # against TenPy (which measures without modifying `psi`), measure on a copy.
    psi_m = deepcopy(psi)
    Zop = Matrix(matrix(ZGate()))
    z = real.(MPSModule.evaluate_all_local_expectations(psi_m, [Zop for _ in 1:psi.length]))
    return z[sites]
end

function _apply_1q!(psi::MPS, op::GateLibrary.AbstractOperator, site::Int)
    U = matrix(op) # 2x2 StaticArray
    A = psi.tensors[site] # (χL, d, χR)
    χL, d, χR = size(A)
    @assert d == size(U, 1)
    # Apply on physical leg: newA[χL, dout, χR] = U[dout, din] * A[χL, din, χR]
    @tensor Anew[χL, dout, χR] := U[dout, din] * A[χL, din, χR]
    psi.tensors[site] = Array(Anew)
    return nothing
end

function _run_julia_variational!(circ::Yaqs.DigitalTJM.DigitalCircuit;
                                state::String,
                                sites_meas::Vector{Int},
                                chi_max::Int,
                                trunc::Float64,
                                min_sweeps::Int,
                                max_sweeps::Int,
                                tol_theta_diff::Float64)
    L = circ.num_qubits
    psi = MPS(L; state=state)
    MPSModule.normalize!(psi)

    obs_rows = Vector{Vector{Float64}}()
    step = -1

    for g in circ.gates
        if g.op isa GateLibrary.Barrier && uppercase(g.op.label) == "SAMPLE_OBSERVABLES"
            step += 1
            push!(obs_rows, [float(step); _measure_z_sites(psi, sites_meas)...])
            continue
        end

        if length(g.sites) == 1
            _apply_1q!(psi, g.op, g.sites[1])
        elseif length(g.sites) == 2
            s1, s2 = g.sites
            U4 = Matrix(matrix(g.op)) # 4x4
            mpo_gate = mpo_from_two_qubit_gate_matrix(U4, s1, s2, L; d=psi.phys_dims[1])
            apply_variational!(psi, mpo_gate;
                               chi_max=chi_max,
                               trunc=trunc,
                               svd_min=eps(Float64),
                               min_sweeps=min_sweeps,
                               max_sweeps=max_sweeps,
                               tol_theta_diff=tol_theta_diff)
        else
            error("Only 1q/2q gates are supported in this comparison script.")
        end
    end

    # If no SAMPLE_OBSERVABLES barriers exist, still return one measurement at step 0
    if step < 0
        step = 0
        push!(obs_rows, [0.0; _measure_z_sites(psi, sites_meas)...])
    end

    return reduce(vcat, (row' for row in obs_rows))
end

function _read_obs_csv(path::AbstractString)
    header = split(strip(readline(path)), ",")
    data = readdlm(path, ',', Float64; skipstart=1)
    return header, data
end

function _write_obs_csv(path::AbstractString, sites::Vector{Int}, data::AbstractMatrix{<:Real})
    open(path, "w") do io
        println(io, join(["step"; ["Z_site$(s)" for s in sites]...], ","))
        for r in 1:size(data, 1)
            println(io, join(string.(data[r, :]), ","))
        end
    end
    return path
end

function _plot_z_comparison!(; outdir::AbstractString,
                             sites::Vector{Int},
                             tag_jl::AbstractString,
                             tag_py::AbstractString)
    # Plot with system python (avoids PythonCall/CondaPkg init inside Julia plotting stacks).
    code = """
import os, sys, csv

outdir = sys.argv[1]
tag_jl = sys.argv[2]
tag_py = sys.argv[3]
sites = [int(x) for x in sys.argv[4].split(',') if x.strip()]

def read_csv(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        rows = [[float(x) for x in row] for row in r]
    return header, rows

def col_index(cols, name):
    try:
        return cols.index(name)
    except ValueError:
        raise RuntimeError(f'Missing column {name} in {cols}')

h_jl, rows_jl = read_csv(os.path.join(outdir, f'{tag_jl}_obs.csv'))
h_py, rows_py = read_csv(os.path.join(outdir, f'{tag_py}_obs.csv'))

n = min(len(rows_jl), len(rows_py))
rows_jl = rows_jl[:n]
rows_py = rows_py[:n]
x = [int(r[0]) for r in rows_jl]

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError('matplotlib not available in system python') from e

fig, axes = plt.subplots(len(sites), 1, figsize=(10, 3*len(sites)), sharex=True)
if len(sites) == 1:
    axes = [axes]

for ax, s in zip(axes, sites):
    c = f'Z_site{s}'
    ij = col_index(h_jl, c)
    ip = col_index(h_py, c)
    yj = [r[ij] for r in rows_jl]
    yp = [r[ip] for r in rows_py]
    ax.plot(x, yj, label=f'{tag_jl} (Julia)', linewidth=2)
    ax.plot(x, yp, label=f'{tag_py} (TenPy)', linewidth=2, linestyle='--')
    ax.set_ylabel(c)
    ax.grid(True)
    ax.legend(loc='best')

axes[-1].set_xlabel('Layer / step')
fig.suptitle('Local ⟨Z⟩ comparison vs layers')
fig.tight_layout()
outpath = os.path.join(outdir, 'z_layers_comparison.png')
fig.savefig(outpath, dpi=160)
plt.close(fig)
print(outpath)
"""
    cmd = `python3 -c $code $outdir $tag_jl $tag_py $(join(sites, ","))`
    run(cmd)
    return nothing
end

function main()
    # --- Small, reproducible example ---
    L = 8
    steps = 10
    dt = 0.05
    periodic = true
    sites = [1, 4, 8] # 1-based measurement sites

    chi_max = 128
    trunc = 1e-12          # relative discarded weight tolerance (matches TenPy wrapper)
    min_sweeps = 2
    max_sweeps = 10
    tol_theta_diff = 1e-12

    outdir = joinpath(@__DIR__, "results", "variational_precision_example")
    mkpath(outdir)

    # Periodic Heisenberg Trotter circuit (as used throughout `03_Nature_review_checks/`)
    circ = create_heisenberg_circuit(L, 1.0, 1.0, 1.0, 0.0, dt, steps; periodic=periodic)
    gatelist_path = joinpath(outdir, "circuit_gatelist.csv")
    write_gatelist_csv(circ, gatelist_path)

    # --- Run TenPy variational (Python) ---
    py_tenpy = joinpath(@__DIR__, "tenpy_mpo_from_gatelist.py")
    tag_py = "tenpy_variational"
    cmd_py = `python3 $py_tenpy --gatelist=$gatelist_path --method=variational --chi-max=$chi_max --trunc=$trunc --sites=$(join(sites, ",")) --state=neel --outdir=$outdir --tag=$tag_py --min-sweeps=$min_sweeps --max-sweeps=$max_sweeps --tol-theta-diff=$tol_theta_diff --max-trunc-err=none`
    @printf("Running TenPy variational:\n  %s\n", string(cmd_py))
    run(cmd_py)

    # --- Run Julia variational ---
    @printf("Running Julia variational MPO-application...\n")
    data_jl = _run_julia_variational!(circ;
                                      state="Neel",
                                      sites_meas=sites,
                                      chi_max=chi_max,
                                      trunc=trunc,
                                      min_sweeps=min_sweeps,
                                      max_sweeps=max_sweeps,
                                      tol_theta_diff=tol_theta_diff)
    tag_jl = "julia_variational"
    _write_obs_csv(joinpath(outdir, "$(tag_jl)_obs.csv"), sites, data_jl)

    # --- Compare ---
    obs_py_path = joinpath(outdir, "$(tag_py)_obs.csv")
    _, data_py = _read_obs_csv(obs_py_path)

    n = min(size(data_jl, 1), size(data_py, 1))
    diff = abs.(data_jl[1:n, 2:end] .- data_py[1:n, 2:end])
    max_abs = maximum(diff)
    @printf("Max |Δ⟨Z⟩| between Julia and TenPy (over %d measurement points): %.3e\n", n, max_abs)
    _plot_z_comparison!(; outdir=outdir, sites=sites, tag_jl=tag_jl, tag_py=tag_py)
    @printf("Outputs in: %s\n", outdir)
end

main()

