using LinearAlgebra
using PythonCall

# Include Yaqs source
include("../src/Yaqs.jl")
using .Yaqs
using .Yaqs.CircuitLibrary
using .Yaqs.GateLibrary
using .Yaqs.MPSModule
using .Yaqs.SimulationConfigs
using .Yaqs.Simulator
using .Yaqs.DigitalTJM: DigitalCircuit, add_gate!, run_digital_tjm
using .Yaqs.DigitalTJMFullLayerMPO: run_digital_tjm_full_layer

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Select Circuit: "Ising", "Heisenberg", "XY", "FermiHubbard", "QAOA", "HEA"
CIRCUIT_NAME = "Heisenberg" 

# System Parameters
L = 25
timesteps = 50
dt = 0.1
num_traj = 1 # Deterministic

# Flags for Execution
RUN_JULIA_V2_WINDOWED = true # New Default
RUN_JULIA_V1_FULL_MPO = false # Old V1
RUN_PYTHON_YAQS = true
RUN_QISKIT_EXACT = false

MAX_BOND_DIM = 128

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

# Initial State
# "zeros" for Ising/Heisenberg (if h != 0)
# "Neel" for XY/FermiHubbard (particle conserving)
INITIAL_STATE = "zeros" 

if CIRCUIT_NAME == "XY" || CIRCUIT_NAME == "FermiHubbard" || CIRCUIT_NAME == "Heisenberg"
    global INITIAL_STATE = "Neel"
end

println("Comparing Simulation: $CIRCUIT_NAME")
println("L=$L, dt=$dt, steps=$timesteps, Initial=$INITIAL_STATE")
println("Run Config: V2_Windowed=$RUN_JULIA_V2_WINDOWED, V1_FullMPO=$RUN_JULIA_V1_FULL_MPO, Python_YAQS=$RUN_PYTHON_YAQS, Qiskit_Exact=$RUN_QISKIT_EXACT")

# ==============================================================================
# 1. CIRCUIT SETUP
# ==============================================================================

circ_jl = DigitalCircuit(L)
# Initial Barrier
add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])

# Circuit Construction
if CIRCUIT_NAME == "Ising"
    circ_jl = ising_circuit(L, J, g, dt, timesteps)
    
elseif CIRCUIT_NAME == "XY"
    for _ in 1:timesteps
        layer = xy_trotter_layer(L, tau) 
        for g in layer.gates; add_gate!(circ_jl, g.op, g.sites); end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

elseif CIRCUIT_NAME == "Heisenberg"
    circ_jl = heisenberg_circuit(L, Jx, Jy, Jz, h_field, dt, timesteps)

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
end

# Observables (Sites 1, 3, 6)
obs_list = [
    Observable("Z_1", ZGate(), 1),
    Observable("Z_3", ZGate(), 3),
    Observable("Z_6", ZGate(), 6)
]

# Results Containers
results_jl_v1 = nothing
results_jl_v2 = nothing
results_py_yaqs = nothing
results_py_exact = nothing

time_jl_v1 = 0.0
time_jl_v2 = 0.0
time_py_yaqs = 0.0

# ==============================================================================
# 2. JULIA V2 (WINDOWED - NEW DEFAULT)
# ==============================================================================
if RUN_JULIA_V2_WINDOWED
    println("\n--- Running Julia V2 (Windowed) ---")
    
    # Warmup
    println("Warming up V2...")
    warmup_psi = MPS(L; state=INITIAL_STATE)
    # Need config
    warmup_config = TimeEvolutionConfig(Observable[], 1.0; dt=1.0, max_bond_dim=MAX_BOND_DIM)
    run_digital_tjm(warmup_psi, circ_jl, nothing, warmup_config)
    println("Warmup complete.")
    
    println("Executing V2...")
    time_jl_v2 = @elapsed begin
        psi_v2 = MPS(L; state=INITIAL_STATE)
        
        # Reset observables
        obs_v2 = deepcopy(obs_list)
        sim_params_v2 = TimeEvolutionConfig(obs_v2, 100.0; dt=1.0, num_traj=num_traj, sample_timesteps=true, max_bond_dim=MAX_BOND_DIM)
        
        _, res_matrix_v2 = run_digital_tjm(psi_v2, circ_jl, nothing, sim_params_v2)
    end
    println("V2 Time: $(round(time_jl_v2, digits=4)) s")
    
    # Slice [2:end]
    results_jl_v2 = [res_matrix_v2[i, 2:end] for i in 1:length(obs_list)]
end

# ==============================================================================
# 3. JULIA V1 (FULL LAYER MPO - OLD)
# ==============================================================================
if RUN_JULIA_V1_FULL_MPO
    println("\n--- Running Julia V1 (Full Layer MPO) ---")
    
    # Warmup
    println("Warming up V1...")
    warmup_psi = MPS(L; state=INITIAL_STATE)
    warmup_config = TimeEvolutionConfig(Observable[], 1.0; dt=1.0, max_bond_dim=MAX_BOND_DIM)
    run_digital_tjm_full_layer(warmup_psi, circ_jl, nothing, warmup_config)
    println("Warmup complete.")
    
    println("Executing V1...")
    time_jl_v1 = @elapsed begin
        psi_v1 = MPS(L; state=INITIAL_STATE)
        
        obs_v1 = deepcopy(obs_list)
        sim_params_v1 = TimeEvolutionConfig(obs_v1, 100.0; dt=1.0, num_traj=num_traj, sample_timesteps=true, max_bond_dim=MAX_BOND_DIM)
        
        _, res_matrix_v1 = run_digital_tjm_full_layer(psi_v1, circ_jl, nothing, sim_params_v1)
    end
    println("V1 Time: $(round(time_jl_v1, digits=4)) s")
    
    results_jl_v1 = [res_matrix_v1[i, 2:end] for i in 1:length(obs_list)]
end

# ==============================================================================
# 4. PYTHON SIMULATION SETUP
# ==============================================================================
if RUN_PYTHON_YAQS || RUN_QISKIT_EXACT
    sys = pyimport("sys")
    local_src = normpath(joinpath(@__DIR__, "../src"))
    mqt_src = normpath(joinpath(@__DIR__, "..", "..", "mqt-yaqs", "src"))

    paths = pyconvert(Vector{String}, sys.path)
    if !(local_src in paths); sys.path.insert(0, local_src); end
    if !(mqt_src in paths); sys.path.insert(0, mqt_src); end

    qiskit = pyimport("qiskit")
    mqt_circuit_lib = pyimport("mqt.yaqs.core.libraries.circuit_library")
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
        trotter_step = mqt_circuit_lib.create_ising_circuit(L, J, g, dt, 1, periodic=false)
    elseif CIRCUIT_NAME == "XY"
        trotter_step = mqt_circuit_lib.xy_trotter_layer(L, tau)
    elseif CIRCUIT_NAME == "Heisenberg"
        trotter_step = mqt_circuit_lib.create_heisenberg_circuit(L, Jx, Jy, Jz, h_field, dt, 1, periodic=false)
    elseif CIRCUIT_NAME == "QAOA"
        trotter_step = mqt_circuit_lib.qaoa_ising_layer(L, beta=beta_qaoa, gamma=gamma_qaoa)
    elseif CIRCUIT_NAME == "HEA"
        phis_list = [phi_hea for _ in 1:L]
        thetas_list = [theta_hea for _ in 1:L]
        lams_list = [lam_hea for _ in 1:L]
        trotter_step = mqt_circuit_lib.hea_layer(L, phis=phis_list, thetas=thetas_list, lams=lams_list, start_parity=start_parity_hea)
    end
end

# ==============================================================================
# 5. QISKIT EXACT
# ==============================================================================
if RUN_QISKIT_EXACT
    println("\n--- Running Qiskit Exact ---")
    reference_py = mqt_simulators.run_qiskit_exact(
        L, timesteps, init_circuit, trotter_step, nothing,
        method="density_matrix", observable_basis="Z"
    )
    reference = pyconvert(Array{Float64,2}, reference_py)
    
    target_sites_py = [0, 2, 5] 
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
    
    obs_yaqs = [
        mqt_params.Observable(mqt_gates.Z(), 0),
        mqt_params.Observable(mqt_gates.Z(), 2),
        mqt_params.Observable(mqt_gates.Z(), 5)
    ]
    
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
if RUN_JULIA_V2_WINDOWED; println("Julia V2 (Windowed): $(round(time_jl_v2, digits=4)) s"); end
if RUN_JULIA_V1_FULL_MPO; println("Julia V1 (Full MPO): $(round(time_jl_v1, digits=4)) s"); end
if RUN_PYTHON_YAQS;       println("Python YAQS:       $(round(time_py_yaqs, digits=4)) s"); end

if RUN_JULIA_V2_WINDOWED && RUN_PYTHON_YAQS
    sp = time_py_yaqs / time_jl_v2
    println("Speedup V2 vs YAQS: $(round(sp, digits=2))x")
end

# Plotting
plt = pyimport("matplotlib.pyplot")

if RUN_QISKIT_EXACT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=true)
else
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax2 = nothing
end

# Determine X axis length from available results
len_x = 0
if RUN_JULIA_V2_WINDOWED; len_x = length(results_jl_v2[1]); end
if RUN_JULIA_V1_FULL_MPO; len_x = length(results_jl_v1[1]); end
if RUN_PYTHON_YAQS;       len_x = length(results_py_yaqs[1]); end
if RUN_QISKIT_EXACT;      len_x = length(results_py_exact[1]); end

# Truncate all to minimum length
min_len = len_x
if RUN_JULIA_V2_WINDOWED; min_len = min(min_len, length(results_jl_v2[1])); end
if RUN_JULIA_V1_FULL_MPO; min_len = min(min_len, length(results_jl_v1[1])); end
if RUN_PYTHON_YAQS;       min_len = min(min_len, length(results_py_yaqs[1])); end
if RUN_QISKIT_EXACT;      min_len = min(min_len, length(results_py_exact[1])); end

x = 1:min_len
colors = ["r", "g", "b"]
sites = [1, 3, 6]

for i in 1:3
    c = colors[i]
    s = sites[i]
    
    if RUN_JULIA_V2_WINDOWED
        ax1.plot(x, results_jl_v2[i][1:min_len], label="JuliaV2 Z_$s", color=c, linestyle="-", linewidth=2)
    end
    
    if RUN_JULIA_V1_FULL_MPO
        ax1.plot(x, results_jl_v1[i][1:min_len], label="JuliaV1 Z_$s", color=c, linestyle="-.", linewidth=1)
    end
    
    if RUN_PYTHON_YAQS
        ax1.plot(x, results_py_yaqs[i][1:min_len], label="YAQS Z_$s", color=c, linestyle=":", linewidth=3)
    end
    
    if RUN_QISKIT_EXACT
        exact = results_py_exact[i][1:min_len]
        ax1.plot(x, exact, label="Qiskit Z_$s", color=c, linestyle="--", linewidth=1, alpha=0.6)
        
        # Errors
        if RUN_JULIA_V2_WINDOWED
            err = abs.(results_jl_v2[i][1:min_len] - exact).^2
            ax2.plot(x, err, label="|JlV2 - Qiskit|^2 Z_$s", color=c, linestyle="-")
        end
        if RUN_PYTHON_YAQS
            err = abs.(results_py_yaqs[i][1:min_len] - exact).^2
            ax2.plot(x, err, label="|YAQS - Qiskit|^2 Z_$s", color=c, linestyle=":")
        end
    end
end

ax1.set_title("$CIRCUIT_NAME (L=$L, $timesteps steps)")
ax1.set_ylabel("<Z>")
ax1.legend(loc="upper right", fontsize="small")
ax1.grid(true)

if RUN_QISKIT_EXACT
    ax2.set_title("Squared Error vs Exact")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Error")
    ax2.legend(loc="upper right", fontsize="small")
    ax2.grid(true)
    ax2.set_yscale("log")
else
    ax1.set_xlabel("Step")
end

plt.tight_layout()
plt.savefig("comparison_$(lowercase(CIRCUIT_NAME)).png")
println("Plot saved.")
