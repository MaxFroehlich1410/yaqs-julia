using Printf
using DelimitedFiles
using TensorOperations

using Yaqs
using Yaqs.GateLibrary
using Yaqs.MPSModule
using Yaqs.MPOModule
using Yaqs.SimulationConfigs
using Yaqs.CircuitLibrary
using Yaqs.DigitalTJM

"""
Compare Julia TEBD vs Julia variational MPO-application on a periodic Heisenberg circuit.

- System: L=8 qubits, Neel initial state
- Circuit: `create_heisenberg_circuit(...; periodic=true)` with `timesteps=10`
- Measurements: ⟨Z⟩ on a chosen set of sites at each `SAMPLE_OBSERVABLES` barrier

Outputs:
- `OUTDIR/tebd_obs.csv`
- `OUTDIR/variational_obs.csv`
- `OUTDIR/z_comparison.png`

Run:
  julia --project 03_Nature_review_checks/compare_julia_tebd_vs_variational_heisenberg_periodic_L8_steps10.jl
"""

# -----------------------------
# Small config block (edit me)
# -----------------------------
const L = 8
const STEPS = 10
const DT = 0.05
const PERIODIC = true
const STATE = "Neel"

const SITES_MEAS = [1, 4, 8]           # 1-based indices
const CHI_MAX = 256
const TRUNC = 1e-12                    # relative discarded weight tolerance (repo convention)

# Variational MPO apply convergence controls
const MIN_SWEEPS = 2
const MAX_SWEEPS = 10
const TOL_THETA_DIFF = 1e-12

const OUTDIR = joinpath(@__DIR__, "results", "jl_tebd_vs_jl_variational_heisenberg_L8_steps10")


@inline function _apply_1q!(psi::MPSModule.MPS{T}, op::GateLibrary.AbstractOperator, site::Int) where {T<:Number}
    U = matrix(op) # StaticArray (d×d)
    A = psi.tensors[site] # (χL, d, χR)
    χL, d, χR = size(A)
    @assert d == size(U, 1)
    @tensor Anew[χL, dout, χR] := U[dout, din] * A[χL, din, χR]
    psi.tensors[site] = Array(Anew)
    return nothing
end

@inline function _measure_z_sites(psi::MPSModule.MPS, sites::Vector{Int})
    # `evaluate_all_local_expectations` shifts/changes gauge; measure on a copy.
    psi_m = deepcopy(psi)
    Zop = Matrix(matrix(ZGate()))
    z_all = real.(MPSModule.evaluate_all_local_expectations(psi_m, [Zop for _ in 1:psi.length]))
    return z_all[sites]
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

function _plot!(outdir::AbstractString, sites::Vector{Int})
    code = """
import os, sys, csv

outdir = sys.argv[1]
sites = [int(x) for x in sys.argv[2].split(',') if x.strip()]

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

h_tebd, rows_tebd = read_csv(os.path.join(outdir, 'tebd_obs.csv'))
h_var, rows_var = read_csv(os.path.join(outdir, 'variational_obs.csv'))

n = min(len(rows_tebd), len(rows_var))
rows_tebd = rows_tebd[:n]
rows_var = rows_var[:n]
x = [int(r[0]) for r in rows_tebd]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(sites), 1, figsize=(10, 3*len(sites)), sharex=True)
if len(sites) == 1:
    axes = [axes]

for ax, s in zip(axes, sites):
    c = f'Z_site{s}'
    it = col_index(h_tebd, c)
    iv = col_index(h_var, c)
    yt = [r[it] for r in rows_tebd]
    yv = [r[iv] for r in rows_var]
    ax.plot(x, yt, label='Julia TEBD', linewidth=2)
    ax.plot(x, yv, label='Julia variational (MPO apply)', linewidth=2, linestyle='--')
    ax.set_ylabel(c)
    ax.grid(True, alpha=0.4)
    ax.legend(loc='best')

axes[-1].set_xlabel('Trotter step')
fig.suptitle('Heisenberg periodic L=8: ⟨Z⟩ comparison')
fig.tight_layout()
outpath = os.path.join(outdir, 'z_comparison.png')
fig.savefig(outpath, dpi=160)
plt.close(fig)
print(outpath)
"""
    cmd = `python3 -c $code $outdir $(join(sites, ","))`
    run(cmd)
    return nothing
end


function run_tebd(circ::DigitalTJM.DigitalCircuit)
    obs = [Observable("Z_site$(s)", ZGate(), s) for s in SITES_MEAS]
    sim = StrongMeasurementConfig(obs; num_traj=1, max_bond_dim=CHI_MAX, truncation_threshold=TRUNC)
    psi0 = MPS(L; state=STATE)
    MPSModule.normalize!(psi0)
    alg = TJMOptions(local_method=:TEBD, long_range_method=:TEBD)

    _, results, _bond_dims = run_digital_tjm(psi0, circ, nothing, sim; alg_options=alg)
    # results is (num_obs, num_steps). Convert to rows: step + values.
    num_steps = size(results, 2)
    data = Matrix{Float64}(undef, num_steps, 1 + length(SITES_MEAS))
    for t in 1:num_steps
        data[t, 1] = t - 1
        @inbounds for (k, _) in enumerate(SITES_MEAS)
            data[t, 1 + k] = real(results[k, t])
        end
    end
    return data
end

function run_variational(circ::DigitalTJM.DigitalCircuit)
    psi = MPS(L; state=STATE)
    MPSModule.normalize!(psi)

    rows = Vector{Vector{Float64}}()
    step = -1

    for g in circ.gates
        if g.op isa GateLibrary.Barrier && uppercase(g.op.label) == "SAMPLE_OBSERVABLES"
            step += 1
            push!(rows, [float(step); _measure_z_sites(psi, SITES_MEAS)...])
            continue
        end

        if length(g.sites) == 1
            _apply_1q!(psi, g.op, g.sites[1])
        elseif length(g.sites) == 2
            s1, s2 = g.sites
            U4 = Matrix{ComplexF64}(matrix(g.op)) # 4x4
            mpo_gate = mpo_from_two_qubit_gate_matrix(U4, s1, s2, L; d=psi.phys_dims[1])
            apply_variational!(psi, mpo_gate;
                               chi_max=CHI_MAX,
                               trunc=TRUNC,
                               svd_min=eps(Float64),
                               min_sweeps=MIN_SWEEPS,
                               max_sweeps=MAX_SWEEPS,
                               tol_theta_diff=TOL_THETA_DIFF)
        else
            error("Only 1q/2q gates are supported.")
        end
    end

    if step < 0
        step = 0
        push!(rows, [0.0; _measure_z_sites(psi, SITES_MEAS)...])
    end

    return reduce(vcat, (r' for r in rows))
end


function main()
    mkpath(OUTDIR)
    @printf("Output dir: %s\n", OUTDIR)

    circ = create_heisenberg_circuit(L, 1.0, 1.0, 1.0, 0.0, DT, STEPS; periodic=PERIODIC)
    @printf("Circuit: L=%d steps=%d dt=%.4g periodic=%s gates=%d\n",
            L, STEPS, DT, string(PERIODIC), length(circ.gates))

    @printf("[tebd] running...\n")
    data_tebd = run_tebd(circ)
    _write_obs_csv(joinpath(OUTDIR, "tebd_obs.csv"), SITES_MEAS, data_tebd)

    @printf("[variational] running...\n")
    data_var = run_variational(circ)
    _write_obs_csv(joinpath(OUTDIR, "variational_obs.csv"), SITES_MEAS, data_var)

    # quick numeric summary
    n = min(size(data_tebd, 1), size(data_var, 1))
    max_abs = maximum(abs.(data_tebd[1:n, 2:end] .- data_var[1:n, 2:end]))
    @printf("Max |Δ⟨Z⟩| (over %d measurement points, %d sites): %.3e\n", n, length(SITES_MEAS), max_abs)

    _plot!(OUTDIR, SITES_MEAS)
    @printf("Wrote plot: %s\n", joinpath(OUTDIR, "z_comparison.png"))
end

main()

