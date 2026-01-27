using Printf
using DelimitedFiles
using TensorOperations
using Dates

using Yaqs
using Yaqs.GateLibrary
using Yaqs.MPSModule
using Yaqs.MPOModule
using Yaqs.CircuitLibrary

include("gatelist_io.jl")
using .GateListIO: write_gatelist_csv

function _measure_z_sites(psi::MPS, sites::Vector{Int})
    # IMPORTANT: `evaluate_all_local_expectations` shifts the orthogonality center and
    # mutates the MPS tensors (gauge changes). For a fair comparison against TenPy,
    # measure on a copy.
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
    @tensor Anew[χL, dout, χR] := U[dout, din] * A[χL, din, χR]
    psi.tensors[site] = Array(Anew)
    return nothing
end

function _run_julia_zipup!(circ::Yaqs.DigitalTJM.DigitalCircuit;
                           state::String,
                           sites_meas::Vector{Int},
                           chi_max::Int,
                           trunc::Float64,
                           svd_min::Float64,
                           m_temp::Int,
                           trunc_weight::Float64)
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
            apply_zipup!(psi, mpo_gate; chi_max=chi_max, svd_min=svd_min, m_temp=m_temp, trunc_weight=trunc_weight)
            # TenPy's `MPO.apply(..., compression_method="zip_up")` does:
            #   apply_zipup + psi.compress_svd(trunc_params)
            # We emulate the final compress step here (relative discarded-weight truncation).
            if trunc > 0
                MPSModule.truncate!(psi; threshold=trunc, max_bond_dim=chi_max)
                MPSModule.normalize!(psi)
            end
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

function _write_timing_csv(path::AbstractString, wall_s_sim::Real)
    open(path, "w") do io
        println(io, "wall_s_sim")
        println(io, string(float(wall_s_sim)))
    end
    return path
end

function _plot_z_comparison!(; outdir::AbstractString,
                             sites::Vector{Int},
                             tag_jl::AbstractString,
                             tag_py::AbstractString)
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

def read_timing(tag):
    p = os.path.join(outdir, f'{tag}_timing.csv')
    if not os.path.exists(p):
        return None
    h, rows = read_csv(p)
    if not rows or not rows[0]:
        return None
    return float(rows[0][0])

t_jl = read_timing(tag_jl)
t_py = read_timing(tag_py)

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
fig.suptitle('Local ⟨Z⟩ comparison vs layers (zip-up)')

# Inset: wallclock time comparison
ax0 = axes[0]
inset = ax0.inset_axes([0.62, 0.08, 0.34, 0.35])
labels = ['Julia', 'TenPy']
vals = [t_jl if t_jl is not None else float('nan'),
        t_py if t_py is not None else float('nan')]
xpos = [0, 1]
inset.bar(xpos, vals, color=['C0', 'C1'], alpha=0.8)
inset.set_xticks(xpos)
inset.set_xticklabels(labels)
inset.set_title('wall_s_sim', fontsize=9)
inset.tick_params(axis='both', labelsize=8)
inset.set_ylabel('s', fontsize=8)
for i, v in enumerate(vals):
    if v == v:  # not NaN
        inset.text(xpos[i], v, f'{v:.2f}s', ha='center', va='bottom', fontsize=8)

fig.tight_layout()
outpath = os.path.join(outdir, 'z_layers_zipup_comparison.png')
fig.savefig(outpath, dpi=160)
plt.close(fig)
print(outpath)
"""
    cmd = `python3 -c $code $outdir $tag_jl $tag_py $(join(sites, ","))`
    run(cmd)
    return nothing
end

function main()
    # --- 8-qubit periodic Heisenberg circuit, 10 timesteps ---
    L = 8
    steps = 100
    dt = 0.05
    periodic = true
    sites = [1, 4, 8]

    chi_max = 256
    trunc = 1e-12
    # TenPy wrapper uses `svd_min≈eps` for numerical safety; match that here.
    svd_min = eps(Float64)
    m_temp = 2
    trunc_weight = 1.0

    outdir = joinpath(@__DIR__, "results", "zipup_precision_example")
    mkpath(outdir)

    circ = create_heisenberg_circuit(L, 1.0, 1.0, 1.0, 0.0, dt, steps; periodic=periodic)
    gatelist_path = joinpath(outdir, "circuit_gatelist.csv")
    write_gatelist_csv(circ, gatelist_path)

    # --- Run TenPy zip_up (Python) ---
    py_tenpy = joinpath(@__DIR__, "tenpy_mpo_from_gatelist.py")
    tag_py = "tenpy_zipup"
    cmd_py = `python3 $py_tenpy --gatelist=$gatelist_path --method=zip_up --chi-max=$chi_max --trunc=$trunc --sites=$(join(sites, ",")) --state=neel --outdir=$outdir --tag=$tag_py --m-temp=$m_temp --trunc-weight=$trunc_weight`
    @printf("Running TenPy zip_up:\n  %s\n", string(cmd_py))
    run(cmd_py)

    # --- Run Julia zip-up ---
    @printf("Running Julia zip-up MPO-application...\n")
    t0 = time()
    data_jl = _run_julia_zipup!(circ;
                                state="Neel",
                                sites_meas=sites,
                                chi_max=chi_max,
                                trunc=trunc,
                                svd_min=svd_min,
                                m_temp=m_temp,
                                trunc_weight=trunc_weight)
    wall_jl = time() - t0
    tag_jl = "julia_zipup"
    _write_obs_csv(joinpath(outdir, "$(tag_jl)_obs.csv"), sites, data_jl)
    _write_timing_csv(joinpath(outdir, "$(tag_jl)_timing.csv"), wall_jl)

    # --- Compare ---
    obs_py_path = joinpath(outdir, "$(tag_py)_obs.csv")
    _, data_py = _read_obs_csv(obs_py_path)
    n = min(size(data_jl, 1), size(data_py, 1))
    diff = abs.(data_jl[1:n, 2:end] .- data_py[1:n, 2:end])
    max_abs = maximum(diff)
    @printf("Max |Δ⟨Z⟩| between Julia and TenPy zip-up (over %d measurement points): %.3e\n", n, max_abs)
    _plot_z_comparison!(; outdir=outdir, sites=sites, tag_jl=tag_jl, tag_py=tag_py)
    @printf("Outputs in: %s\n", outdir)
end

main()

