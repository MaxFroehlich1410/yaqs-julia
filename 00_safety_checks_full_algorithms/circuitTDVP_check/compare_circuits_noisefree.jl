using LinearAlgebra
using PythonCall

# Include Yaqs source
include("../../src/Yaqs.jl")
using .Yaqs
using .Yaqs.CircuitLibrary
using .Yaqs.GateLibrary
using .Yaqs.MPSModule
using .Yaqs.SimulationConfigs
using .Yaqs.Simulator
using .Yaqs.DigitalTJM: DigitalCircuit, add_gate!, run_digital_tjm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Select Circuit: "Ising", "Heisenberg", "XY", "FermiHubbard", "QAOA", "HEA", "longrange_test"
CIRCUIT_NAME = "longrange_test" 
periodic = false

# System Parameters
L = 7
timesteps = 150
dt = 0.1
num_traj = 1 # Deterministic

# Flags for Execution
RUN_JULIA = true # New Default
RUN_PYTHON_YAQS = true
RUN_QISKIT_EXACT = true
SITES_TO_SAMPLE = [1, 2, 3, 4, 5, 6, 7]

MAX_BOND_DIM = 56

# Model Specific Params
# Ising
J = 1.0
g = 0.5

# Heisenberg
Jx, Jy, Jz = 1.0, 1.0, 1.0
h_field = 0.5

# XY
tau = 0.1 # "dt" for XY often called tau

# QAOA
beta_qaoa = 0.3
gamma_qaoa = 0.5

# HEA
phi_hea = 0.2
theta_hea = 0.4
lam_hea = 0.6
start_parity_hea = 0

# Longrange Test
longrange_theta = π/4  # Rotation angle for the RXX gate

# Initial State
# "zeros" for Ising/Heisenberg (if h != 0)
# "Neel" for XY/FermiHubbard (particle conserving)
INITIAL_STATE = "zeros" 

if CIRCUIT_NAME == "XY" || CIRCUIT_NAME == "FermiHubbard" || CIRCUIT_NAME == "Heisenberg"
    global INITIAL_STATE = "Neel"
end

println("Comparing Simulation: $CIRCUIT_NAME")
println("L=$L, dt=$dt, steps=$timesteps, Initial=$INITIAL_STATE")
println("Run Config: Julia=$RUN_JULIA, Python_YAQS=$RUN_PYTHON_YAQS, Qiskit_Exact=$RUN_QISKIT_EXACT")

# ==============================================================================
# 1. CIRCUIT SETUP
# ==============================================================================

circ_jl = DigitalCircuit(L)
# Initial Barrier
add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])

# Circuit Construction
if CIRCUIT_NAME == "Ising"
    circ_jl = ising_circuit(L, J, g, dt, timesteps, periodic=periodic)
    
elseif CIRCUIT_NAME == "XY"
    for _ in 1:timesteps
        layer = xy_trotter_layer(L, tau) 
        for g in layer.gates; add_gate!(circ_jl, g.op, g.sites); end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

elseif CIRCUIT_NAME == "Heisenberg"
    circ_jl = heisenberg_circuit(L, Jx, Jy, Jz, h_field, dt, timesteps, periodic=periodic)

elseif CIRCUIT_NAME == "QAOA"
    for _ in 1:timesteps
        layer = qaoa_ising_layer(L; beta=beta_qaoa, gamma=gamma_qaoa)
        for g in layer.gates; add_gate!(circ_jl, g.op, g.sites); end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

elseif CIRCUIT_NAME == "HEA"
    phis = fill(phi_hea, L)
    thetas = fill(theta_hea, L)
    lams = fill(lam_hea, L)
    for _ in 1:timesteps
        layer = hea_layer(L; phis=phis, thetas=thetas, lams=lams, start_parity=start_parity_hea)
        for g in layer.gates; add_gate!(circ_jl, g.op, g.sites); end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

elseif CIRCUIT_NAME == "longrange_test"
    # Longrange test circuit: H gates on all qubits, then one RXX gate between qubits L and 1
    for _ in 1:timesteps
        # Apply H gates to all qubits
        for q in 1:L
            add_gate!(circ_jl, HGate(), [q])
        end
        # Apply exactly ONE long-range two-qubit gate: RXX between qubits L and 1
        add_gate!(circ_jl, RxxGate(longrange_theta), [L, 1])
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end
end

# Observables (Sites 1, 3, 6)
obs_list = [Observable("Z_$s", ZGate(), s) for s in SITES_TO_SAMPLE]

# Results Containers
results_jl_v1 = nothing
results_jl = nothing
results_py_yaqs = nothing
results_py_exact = nothing
results_py_noisefree = nothing

time_jl_v1 = 0.0
time_jl = 0.0
time_py_yaqs = 0.0

# ==============================================================================
# 2. JULIA
# ==============================================================================
if RUN_JULIA
    println("\n--- Running Julia (Windowed) ---")
    
    # Warmup
    println("Warming up...")
    # Create a minimal warmup run using the ACTUAL circuit and observables
    # to trigger compilation of all relevant methods (gate application, measurement, etc.)
    warmup_L = L 
    warmup_psi = MPS(warmup_L; state="zeros") 
    
    # Use the actual circuit and observables for warmup
    # We use a reduced config (e.g. max_bond_dim=2) to keep it slightly faster if possible,
    # but using the real config is safer for compilation coverage.
    warmup_config = TimeEvolutionConfig(obs_list, 100.0; dt=1.0, max_bond_dim=4)
    
    # Run!
    run_digital_tjm(warmup_psi, circ_jl, nothing, warmup_config)
    println("Warmup complete.")
    
    println("Executing...")
    time_jl = @elapsed begin
        psi = MPS(L; state=INITIAL_STATE)
        
        # Reset observables
        obs = deepcopy(obs_list)
        sim_params = TimeEvolutionConfig(obs, 100.0; dt=1.0, num_traj=num_traj, sample_timesteps=true, max_bond_dim=MAX_BOND_DIM)
        
        _, res_matrix = run_digital_tjm(psi, circ_jl, nothing, sim_params)
    end
    println("Julia Time: $(round(time_jl, digits=4)) s")
    
    # Slice [2:end]
    results_jl = [res_matrix[i, 2:end] for i in 1:length(obs_list)]
end

# ==============================================================================
# 3. PYTHON SIMULATION SETUP
# ==============================================================================
if RUN_PYTHON_YAQS || RUN_QISKIT_EXACT
    sys = pyimport("sys")
    local_src = normpath(joinpath(@__DIR__, "../../src"))
    # Adjusted to ../../../ because mqt-yaqs is a sibling of yaqs-julia
    mqt_src = normpath(joinpath(@__DIR__, "../../../mqt-yaqs/src"))
    mqt_inner = normpath(joinpath(@__DIR__, "../../../mqt-yaqs/src/mqt/yaqs"))

    paths = pyconvert(Vector{String}, sys.path)
    if !(local_src in paths); sys.path.insert(0, local_src); end
    if !(mqt_src in paths); sys.path.insert(0, mqt_src); end
    if !(mqt_inner in paths); sys.path.insert(0, mqt_inner); end

    qiskit = pyimport("qiskit")
    mqt_circuit_lib = pyimport("mqt.yaqs.core.libraries.circuit_library")
    # Local Qiskit_simulator module for longrange_test_circuit
    local_circuit_lib = pyimport("Qiskit_simulator.circuit_library")
    mqt_simulators = pyimport("mqt.yaqs.codex_experiments.worker_functions.qiskit_simulators")
    mqt_sim = pyimport("mqt.yaqs.simulator")
    mqt_networks = pyimport("mqt.yaqs.core.data_structures.networks")
    mqt_params = pyimport("mqt.yaqs.core.data_structures.simulation_parameters")
    mqt_gates = pyimport("mqt.yaqs.core.libraries.gate_library")

    init_circuit = qiskit.QuantumCircuit(L)

    if INITIAL_STATE == "Neel"
        for i in 1:2:(L-1) 
            init_circuit.x(i) 
        end
    end

    # Construct Trotter Step
    trotter_step = nothing
    if CIRCUIT_NAME == "Ising"
        trotter_step = mqt_circuit_lib.create_ising_circuit(L, J, g, dt, 1, periodic=periodic)
    elseif CIRCUIT_NAME == "XY"
        trotter_step = mqt_circuit_lib.xy_trotter_layer(L, tau)
    elseif CIRCUIT_NAME == "Heisenberg"
        trotter_step = mqt_circuit_lib.create_heisenberg_circuit(L, Jx, Jy, Jz, h_field, dt, 1, periodic=periodic)
    elseif CIRCUIT_NAME == "QAOA"
        trotter_step = mqt_circuit_lib.qaoa_ising_layer(L, beta=beta_qaoa, gamma=gamma_qaoa)
    elseif CIRCUIT_NAME == "HEA"
        phis_list = [phi_hea for _ in 1:L]
        thetas_list = [theta_hea for _ in 1:L]
        lams_list = [lam_hea for _ in 1:L]
        trotter_step = mqt_circuit_lib.hea_layer(L, phis=phis_list, thetas=thetas_list, lams=lams_list, start_parity=start_parity_hea)
    elseif CIRCUIT_NAME == "longrange_test"
        trotter_step = local_circuit_lib.longrange_test_circuit(L, longrange_theta)
    end
end

# ==============================================================================
# 5. QISKIT EXACT
# ==============================================================================
if RUN_QISKIT_EXACT
    println("\n--- Running Qiskit Exact Density Matrix ---")
    reference_py = mqt_simulators.run_qiskit_exact(
        L, timesteps, init_circuit, trotter_step, nothing,
        method="density_matrix", observable_basis="Z"
    )
    reference = pyconvert(Array{Float64,2}, reference_py)
    
    target_sites_py = [s-1 for s in SITES_TO_SAMPLE] 
    results_py_exact = [reference[i+1, :] for i in target_sites_py]
end
# ==============================================================================
# 6. PYTHON YAQS
# ==============================================================================
if RUN_PYTHON_YAQS
    println("\n--- Running Python YAQS ---")
    
    full_yaqs_circ = init_circuit.copy()
    full_yaqs_circ.barrier(label="SAMPLE_OBSERVABLES")
    for _ in 1:timesteps
        full_yaqs_circ.compose(trotter_step, inplace=true)
        full_yaqs_circ.barrier(label="SAMPLE_OBSERVABLES")
    end
    
    obs_yaqs = [mqt_params.Observable(mqt_gates.Z(), s-1) for s in SITES_TO_SAMPLE]
    
    sim_params_yaqs = mqt_params.StrongSimParams(
        observables=obs_yaqs, num_traj=1, max_bond_dim=MAX_BOND_DIM,
        sample_layers=true, num_mid_measurements=timesteps
    )
    sim_params_yaqs.dt = 1.0
    
    psi_yaqs = mqt_networks.MPS(L, state="zeros", pad=2)
    
    println("Executing YAQS...")
    time_py_yaqs = @elapsed mqt_sim.run(psi_yaqs, full_yaqs_circ, sim_params_yaqs, nothing, parallel=false)
    println("YAQS Time: $(round(time_py_yaqs, digits=4)) s")
    
    res_raw = [pyconvert(Vector{Float64}, obs.results) for obs in obs_yaqs]
    results_py_yaqs = [r[3:end] for r in res_raw]
end

# ==============================================================================
# 7. COMPARISON & PLOTTING
# ==============================================================================

println("\n--- Performance Summary ---")
if RUN_JULIA; println("Julia: $(round(time_jl, digits=4)) s"); end
if RUN_PYTHON_YAQS;       println("Python YAQS:       $(round(time_py_yaqs, digits=4)) s"); end

if RUN_JULIA && RUN_PYTHON_YAQS
    sp = time_py_yaqs / time_jl
    println("Speedup Julia vs YAQS: $(round(sp, digits=2))x")
end

# Plotting
plt = pyimport("matplotlib.pyplot")

num_plots = length(SITES_TO_SAMPLE)
fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=true)

# Handle single vs multiple axes
axes_array = Vector{Any}(undef, num_plots)
if num_plots == 1
    axes_array[1] = axes
else
    for i in 1:num_plots
        axes_array[i] = axes[i-1] # 0-based indexing
    end
end

# Determine X axis length from available results
len_x = 0
if RUN_JULIA; len_x = length(results_jl[1]); end
if RUN_PYTHON_YAQS;       len_x = length(results_py_yaqs[1]); end
if RUN_QISKIT_EXACT;      len_x = length(results_py_exact[1]); end

# Truncate all to minimum length
min_len = len_x
if RUN_JULIA; min_len = min(min_len, length(results_jl[1])); end
if RUN_PYTHON_YAQS;       min_len = min(min_len, length(results_py_yaqs[1])); end
if RUN_QISKIT_EXACT;      min_len = min(min_len, length(results_py_exact[1])); end

x = 1:min_len
colors = ["r", "g", "b", "c", "m", "y", "k"]

for (i, s) in enumerate(SITES_TO_SAMPLE)
    ax = axes_array[i]
    
    if RUN_JULIA
        ax.plot(x, results_jl[i][1:min_len], label="Julia", color="r", linestyle="-", linewidth=2)
    end
    
    if RUN_PYTHON_YAQS
        ax.plot(x, results_py_yaqs[i][1:min_len], label="YAQS", color="g", linestyle=":", linewidth=3)
    end
    
    if RUN_QISKIT_EXACT
        exact = results_py_exact[i][1:min_len]
        ax.plot(x, exact, label="Qiskit Exact", color="b", linestyle="--", linewidth=1, alpha=0.6)
    end
    
    ax.set_title("Site $s Evolution")
    ax.set_ylabel("<Z>")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(true)
    
    if i == num_plots
        ax.set_xlabel("Step")
    end
end

fig.suptitle("$CIRCUIT_NAME (L=$L, $timesteps steps)")
plt.tight_layout()
results_dir = joinpath(@__DIR__, "results")
if !isdir(results_dir)
    mkpath(results_dir)
end
fname = joinpath(results_dir, "comparison_$(lowercase(CIRCUIT_NAME)).png")
plt.savefig(fname)
println("Plot saved to $fname")
