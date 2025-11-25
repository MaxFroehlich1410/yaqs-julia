using LinearAlgebra
using Printf
using PythonCall
using TensorOperations

# Ensure src modules are available
if !isdefined(Main, :GateLibrary)
    include("../src/GateLibrary.jl")
end
if !isdefined(Main, :Decompositions)
    include("../src/Decompositions.jl")
end
if !isdefined(Main, :MPSModule)
    include("../src/MPS.jl")
end
if !isdefined(Main, :MPOModule)
    include("../src/MPO.jl")
end
if !isdefined(Main, :SimulationConfigs)
    include("../src/SimulationConfigs.jl")
end
if !isdefined(Main, :Algorithms)
    include("../src/Algorithms.jl")
end

using .GateLibrary
using .MPSModule
using .MPOModule
using .SimulationConfigs
using .Algorithms

# --- Configuration ---
const L = 4
const dt = 0.05
const t_max = 4.0
const steps = Int(floor(t_max / dt))
const max_bond_dim = 8
const trunc_err = 1e-12

# Local Observable to Plot
const OBS_TYPE = "Z" # "X", "Y", "Z"
const SITES_TO_PLOT = [1, 2, 3, 4]

# Hamiltonian Parameters
# J terms (size L-1)
Jxx = 2.0 * ones(L-1)
Jyy = 1.8 * ones(L-1)
Jzz = 0.5 * ones(L-1)

# Local Fields (size L)
hx = 0.3 * ones(L)
hy = 2.0 * ones(L)
hz = 0.5 * ones(L)
# Add some inhomogeneity
hx[1] += 0.2
hz[L] -= 0.2

println("Configuration:")
println("  L = $L")
println("  dt = $dt, t_max = $t_max")
println("  max_bond = $max_bond_dim")
println("  Observable = $OBS_TYPE")
println("  Sites = $SITES_TO_PLOT")

# --- QuTiP Simulation (Exact) ---
println("\nRunning QuTiP Simulation...")
qt = pyimport("qutip")
np = pyimport("numpy")

function get_qutip_hamiltonian(L, Jxx, Jyy, Jzz, hx, hy, hz)
    # Explicitly convert to Python lists for QuTiP
    d_list = pylist(fill(2, L))
    dims = pylist([d_list, d_list])
    
    H = qt.Qobj(np.zeros((2^L, 2^L)), dims=dims)
    
    sx_list = [qt.tensor(pylist([i == j ? qt.sigmax() : qt.qeye(2) for j in 0:L-1])) for i in 0:L-1]
    sy_list = [qt.tensor(pylist([i == j ? qt.sigmay() : qt.qeye(2) for j in 0:L-1])) for i in 0:L-1]
    sz_list = [qt.tensor(pylist([i == j ? qt.sigmaz() : qt.qeye(2) for j in 0:L-1])) for i in 0:L-1]
    
    # Interaction terms
    for i in 1:(L-1)
        # Python 0-based indexing
        # But wait, sx_list is 0-based in my construction?
        # sx_list = [ ... for i in 0:L-1]. So sx_list[1] is site 0.
        # Julia 1-based indexing for Jxx[i].
        # Interaction between site i and i+1 (1-based) -> indices i-1 and i (0-based).
        # Julia arrays are 1-based.
        # sx_list is a Julia Vector of PyObjects.
        # sx_list[i] is the i-th element (site i-1).
        
        # Site 1 (Julia) -> index 1 in sx_list -> site 0 operator.
        # Site i (Julia) -> index i.
        # Bond (i, i+1) involves sx_list[i] and sx_list[i+1].
        
        H += Jxx[i] * sx_list[i] * sx_list[i+1]
        H += Jyy[i] * sy_list[i] * sy_list[i+1]
        H += Jzz[i] * sz_list[i] * sz_list[i+1]
    end
    
    # Field terms
    for i in 1:L
        H += hx[i] * sx_list[i]
        H += hy[i] * sy_list[i]
        H += hz[i] * sz_list[i]
    end
    
    return H
end

# Initial State: |00...0> (All Up in Z)
# Use pylist for qt.tensor
psi0_qutip = qt.tensor(pylist([qt.basis(2, 0) for _ in 1:L]))

H_qutip = get_qutip_hamiltonian(L, Jxx, Jyy, Jzz, hx, hy, hz)

times = collect(0:dt:t_max)
t_list_py = np.array(times)

# Evolution
result = qt.sesolve(H_qutip, psi0_qutip, t_list_py)

# Extract Expectation Values
qutip_data = Dict()
for site in SITES_TO_PLOT
    # Site 1 -> index 0
    op_py = qt.tensor(pylist([i == (site-1) ? (OBS_TYPE=="X" ? qt.sigmax() : (OBS_TYPE=="Y" ? qt.sigmay() : qt.sigmaz())) : qt.qeye(2) for i in 0:L-1]))
    # expect returns array
    qutip_data[site] = pyconvert(Vector{Float64}, qt.expect(op_py, result.states))
end
println("QuTiP Done.")

# --- Julia Simulation ---
println("\nRunning Julia TDVP...")

# 1. Initialize MPS |00...0>
mps_1 = MPS(L; state="zeros")
pad_bond_dimension!(mps_1, 4) # Helper to expand bond dim slightly for 1-site

mps_2 = MPS(L; state="zeros")

# 2. Construct MPO
H_mpo = init_general_hamiltonian(L, Jxx, Jyy, Jzz, hx, hy, hz)

# 3. Configs
config_1 = TimeEvolutionConfig(Observable[], t_max; dt=dt, max_bond_dim=max_bond_dim)
config_2 = TimeEvolutionConfig(Observable[], t_max; dt=dt, max_bond_dim=max_bond_dim, truncation_threshold=trunc_err)

# Data Storage
julia_1_data = Dict(s => Float64[] for s in SITES_TO_PLOT)
julia_2_data = Dict(s => Float64[] for s in SITES_TO_PLOT)

# Observable Matrix
obs_op = OBS_TYPE == "X" ? matrix(XGate()) : (OBS_TYPE == "Y" ? matrix(YGate()) : matrix(ZGate()))
obs_mat = Matrix(obs_op)

function record_obs!(mps, data_dict)
    for s in SITES_TO_PLOT
        val = real(local_expect(mps, obs_mat, s))
        push!(data_dict[s], val)
    end
end

# Initial Record
record_obs!(mps_1, julia_1_data)
record_obs!(mps_2, julia_2_data)

# Time Loop
for i in 1:steps
    # 1-Site
    single_site_tdvp!(mps_1, H_mpo, config_1)
    record_obs!(mps_1, julia_1_data)
    
    # 2-Site
    two_site_tdvp!(mps_2, H_mpo, config_2)
    record_obs!(mps_2, julia_2_data)
    
    if i % 10 == 0
        @printf("Step %d / %d\n", i, steps)
    end
end

println("Julia Simulation Done.")

# --- Plotting ---
println("\nPlotting Results...")
plt = pyimport("matplotlib.pyplot")

fig, axes = plt.subplots(length(SITES_TO_PLOT), 1, figsize=(10, 3*length(SITES_TO_PLOT)), sharex=true)

# Ensure consistent access (always treat as list/array)
axes_py = length(SITES_TO_PLOT) == 1 ? pylist([axes]) : axes

for (idx, site) in enumerate(SITES_TO_PLOT)
    ax = axes_py[idx-1] # 0-based indexing for python list access
    
    # QuTiP (Dashed)
    ax.plot(times, qutip_data[site], label="QuTiP (Exact)", linestyle="--", color="black", linewidth=2)
    
    # Julia 1-TDVP
    ax.plot(times, julia_1_data[site], label="Julia 1-TDVP", linestyle="-", marker="o", markersize=3, markevery=5)
    
    # Julia 2-TDVP
    ax.plot(times, julia_2_data[site], label="Julia 2-TDVP", linestyle="-", marker="x", markersize=3, markevery=5)
    
    ax.set_ylabel("ExpVal <$(OBS_TYPE)> Site $site")
    if idx == 1
        ax.legend()
    end
    ax.grid(true)
end

axes_py[length(SITES_TO_PLOT)-1].set_xlabel("Time")
plt.tight_layout()
plt.savefig("general_tdvp_comparison.png")
println("Plot saved to general_tdvp_comparison.png")

