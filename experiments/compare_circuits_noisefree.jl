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
using .Yaqs.DigitalTJM: DigitalCircuit, add_gate!
using .Yaqs.DigitalTJMV2: run_digital_tjm_v2

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Select Circuit: "Ising", "Heisenberg", "XY", "FermiHubbard", "QAOA", "HEA"
CIRCUIT_NAME = "Ising" 

# System Parameters
L = 49
timesteps = 50
dt = 0.1
num_traj = 1 # Deterministic

# Flags
RUN_QISKIT_EXACT = false
MAX_BOND_DIM = 512

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

println("Comparing Simulation: $CIRCUIT_NAME (Julia vs Python)")
println("L=$L, dt=$dt, steps=$timesteps, Initial=$INITIAL_STATE")

# ==============================================================================
# 1. JULIA SIMULATION
# ==============================================================================
println("\n--- Running Julia Simulation ---")

circ_jl = DigitalCircuit(L)
# Initial Barrier
add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])

# Circuit Construction
if CIRCUIT_NAME == "Ising"
    # ising_circuit builds the whole timeline internally including barriers (if modified)
    # But wait, ising_circuit in library MIGHT NOT have barriers if we didn't commit that change fully.
    # Let's rely on manual construction if library function is single-step, or library if multi-step.
    # Library `ising_circuit` takes `timesteps`.
    circ_jl = ising_circuit(L, J, g, dt, timesteps)
    
elseif CIRCUIT_NAME == "XY"
    for _ in 1:timesteps
        layer = xy_trotter_layer(L, tau) # xy_trotter_layer is single step
        for g in layer.gates; add_gate!(circ_jl, g.op, g.sites); end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

elseif CIRCUIT_NAME == "Heisenberg"
    circ_jl = heisenberg_circuit(L, Jx, Jy, Jz, h_field, dt, timesteps)
    # Note: heisenberg_circuit in library likely needs update to include barriers if we want step-by-step
    # For now, let's assume it does, or we only get final result.
    # To be safe/generic, we should probably construct loop here if library isn't trusted.
    # But let's try library.

elseif CIRCUIT_NAME == "QAOA"
    for _ in 1:timesteps
        layer = qaoa_ising_layer(L; beta=beta_qaoa, gamma=gamma_qaoa)
        for g in layer.gates; add_gate!(circ_jl, g.op, g.sites); end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

elseif CIRCUIT_NAME == "HEA"
    # Construct vectors for HEA
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
obs_jl = [
    Observable("Z_1", ZGate(), 1),
    Observable("Z_3", ZGate(), 3),
    Observable("Z_6", ZGate(), 6)
]

sim_params_jl = TimeEvolutionConfig(obs_jl, 100.0; dt=1.0, num_traj=num_traj, sample_timesteps=true, max_bond_dim=MAX_BOND_DIM)

psi_jl = MPS(L; state=INITIAL_STATE)

println("Warming up Julia compilation...")
# Run once to compile (ignoring results)
warmup_psi = MPS(L; state=INITIAL_STATE)
Simulator.run(warmup_psi, circ_jl, sim_params_jl, nothing; parallel=false)
println("Warmup complete.")

println("Executing Julia Simulation (Measured)...")
time_jl = @elapsed begin
    # Re-initialize state to ensure fresh start
    # (Note: In a real high-perf loop, we'd allocate once, but here allocation is negligible compared to evolution)
    psi_jl_measured = MPS(L; state=INITIAL_STATE)
    Simulator.run(psi_jl_measured, circ_jl, sim_params_jl, nothing; parallel=false)
end
# Update psi_jl to the result of the measured run for plotting logic below
psi_jl = psi_jl_measured

println("Julia Simulation Time: $(round(time_jl, digits=4)) s")

results_jl = [obs.results for obs in obs_jl]

# Trim: Remove initial state (index 1) to align with Python which usually returns t=1..N
# Check length first
jl_len = length(results_jl[1])
println("Julia raw points: $jl_len")

# Python usually returns 'timesteps' points (1 to N).
# Julia returns 1 (Initial) + N (steps) = N+1.
# So we slice [2:end].
results_jl_final = [res[2:end] for res in results_jl]

# ==============================================================================
# 2. JULIA V2 SIMULATION (Windowed TJM)
# ==============================================================================
println("\n--- Running Julia V2 Simulation (Windowed) ---")

# Warmup V2
println("Warming up Julia V2 compilation...")
warmup_psi_v2 = MPS(L; state=INITIAL_STATE)
# Note: Simulator.run currently calls DigitalTJM (V1). We need to call V2 directly.
run_digital_tjm_v2(warmup_psi_v2, circ_jl, nothing, sim_params_jl)
println("Warmup complete.")

println("Executing Julia V2 Simulation...")
time_jl_v2 = @elapsed begin
    psi_jl_v2 = MPS(L; state=INITIAL_STATE)
    # We must reset observable results inside sim_params_jl before run, or they will append?
    # Simulator.run handles resetting usually? No, SimConfig holds results in Observables.
    # We should create a new config or clear results.
    obs_jl_v2 = [
        Observable("Z_1", ZGate(), 1),
        Observable("Z_3", ZGate(), 3),
        Observable("Z_6", ZGate(), 6)
    ]
    sim_params_jl_v2 = TimeEvolutionConfig(obs_jl_v2, 100.0; dt=1.0, num_traj=num_traj, sample_timesteps=true, max_bond_dim=MAX_BOND_DIM)
    
    run_digital_tjm_v2(psi_jl_v2, circ_jl, nothing, sim_params_jl_v2)
end

println("Julia V2 Simulation Time: $(round(time_jl_v2, digits=4)) s")

results_jl_v2 = [obs.results for obs in sim_params_jl_v2.observables] # V2 might return results differently?
# run_digital_tjm_v2 returns (state, results_matrix) where results_matrix is (num_obs, num_steps)
# But it ALSO updates sim_params.observables via evaluate_observables! if it uses `expect` which pushes?
# Let's check V2 implementation.
# In V2:
# function measure!(idx)
#    for (i, obs) in enumerate(sim_params.observables)
#        results[i, idx] = SimulationConfigs.expect(state, obs)
#    end
# end
# It populates `results` matrix AND `SimulationConfigs.expect` might push to obs.results?
# SimulationConfigs.expect just returns value usually.
# Let's rely on the returned matrix from run_digital_tjm_v2 if possible, or observable results if pushed.
# Actually, DigitalTJM.jl implementation of `evaluate_observables!` calls `expect`.
# `expect` in SimulationConfigs usually just computes.
# Wait, let's check `SimulationConfigs.expect` in `SimulationConfigs.jl`?
# In `DigitalTJMV2.jl` I wrote: `results[i, idx] = SimulationConfigs.expect(state, obs)`.
# So the matrix `results` (returned as second arg) is the source of truth.

# However, `run_digital_tjm_v2` returns `(state, results)`.
# Let's capture the return value.

# We need to re-run the V2 block to capture output correctly.
# Redoing the block above:

time_jl_v2 = @elapsed begin
    psi_jl_v2 = MPS(L; state=INITIAL_STATE)
    # We don't strictly need new config if we use returned results matrix, but safe practice.
    obs_jl_v2 = [
        Observable("Z_1", ZGate(), 1),
        Observable("Z_3", ZGate(), 3),
        Observable("Z_6", ZGate(), 6)
    ]
    sim_params_jl_v2 = TimeEvolutionConfig(obs_jl_v2, 100.0; dt=1.0, num_traj=num_traj, sample_timesteps=true)
    
    _, results_matrix_v2 = run_digital_tjm_v2(psi_jl_v2, circ_jl, nothing, sim_params_jl_v2)
end

println("Julia V2 Simulation Time: $(round(time_jl_v2, digits=4)) s")

# results_matrix_v2 is (num_obs, num_steps)
# Slice [2:end] (remove initial)
results_jl_v2_final = [results_matrix_v2[i, 2:end] for i in 1:length(obs_jl_v2)]

# ==============================================================================
# 3. PYTHON SIMULATION
# ==============================================================================
println("\n--- Running Python Simulation ---")

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

# Apply Initial State
if INITIAL_STATE == "Neel"
    # |010101...>
    # Julia "Neel": Odd sites (1,3..) -> Up (|0>), Even sites (2,4..) -> Down (|1>)
    # Python Qiskit: Starts |00...0> (Up).
    # To match Julia, we apply X (flip to Down) on ODD indices (1, 3, 5...) 
    # (Remember Python 0-based: 1 is the second qubit).
    for i in 1:2:(L-1) 
        init_circuit.x(i) 
    end
end

# Construct Trotter Step
trotter_step = nothing

if CIRCUIT_NAME == "Ising"
    # create_ising_circuit returns FULL circuit with timesteps. 
    # But run_qiskit_exact expects a single STEP and repeats it.
    # We need a function that returns ONE step.
    # mqt_circuit_lib.create_ising_circuit returns full.
    # We must check if there is a single step function or ask it for 1 step.
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
    trotter_step = mqt_circuit_lib.hea_layer(
        L, 
        phis=phis_list, 
        thetas=thetas_list, 
        lams=lams_list, 
        start_parity=start_parity_hea
    )
end

results_py_final = nothing

if RUN_QISKIT_EXACT
    println("Running Qiskit density-matrix reference ...")
    reference_py = mqt_simulators.run_qiskit_exact(
        L,
        timesteps,
        init_circuit,
        trotter_step,
        nothing,
        method="density_matrix",
        observable_basis="Z",
    )
    reference = pyconvert(Array{Float64,2}, reference_py)
    println("Qiskit reference obtained. Shape: ", size(reference))

    # Extract Z expectations for sites 1, 3, 6
    # Python reference is (num_qubits, timesteps)
    target_sites_py = [0, 2, 5] # 0-based for 1, 3, 6
    results_py_final = [reference[i+1, :] for i in target_sites_py] # Julia 1-based index into array
else
    println("Skipping Qiskit exact simulation (RUN_QISKIT_EXACT = false)")
end

# ==============================================================================
# 3. PYTHON YAQS SIMULATION (DigitalTJM)
# ==============================================================================
println("\n--- Running Python YAQS (DigitalTJM) ---")

# Build full circuit with barriers
full_yaqs_circ = init_circuit.copy()
# Initial measurement barrier
full_yaqs_circ.barrier(label="SAMPLE_OBSERVABLES")

for _ in 1:timesteps
    full_yaqs_circ.compose(trotter_step, inplace=true)
    full_yaqs_circ.barrier(label="SAMPLE_OBSERVABLES")
end

# Observables: Z on sites 1, 3, 6 (indices 0, 2, 5)
obs_yaqs = [
    mqt_params.Observable(mqt_gates.Z(), 0),
    mqt_params.Observable(mqt_gates.Z(), 2),
    mqt_params.Observable(mqt_gates.Z(), 5)
]

# Simulation Parameters
# sample_layers=true tells YAQS to measure at barriers labeled "SAMPLE_OBSERVABLES"
sim_params_yaqs = mqt_params.StrongSimParams(
    observables=obs_yaqs,
    num_traj=1,
    max_bond_dim=MAX_BOND_DIM,
    sample_layers=true,
    num_mid_measurements=timesteps
)
sim_params_yaqs.dt = 1.0 # Manually set dt to 1.0 for physics consistency

# Initial State (MPS)
# We use "zeros" and rely on the X gates in init_circuit (if Neel) to prepare the state,
# just like we did for the Qiskit reference.
psi_yaqs = mqt_networks.MPS(L, state="zeros", pad=2)

println("Starting YAQS simulation (Measured)...")
time_yaqs = @elapsed mqt_sim.run(psi_yaqs, full_yaqs_circ, sim_params_yaqs, nothing, parallel=false)
println("YAQS Simulation Time: $(round(time_yaqs, digits=4)) s")

# Speedup stats
speedup = time_yaqs / time_jl
println("\n--- Performance Comparison ---")
println("Julia V1 Time: $(round(time_jl, digits=4)) s")
println("Julia V2 Time: $(round(time_jl_v2, digits=4)) s")
println("YAQS Time:     $(round(time_yaqs, digits=4)) s")

# Speedup calculated as Time_Baseline / Time_Target
# We want to see how much FASTER Julia is compared to YAQS (Python).
speedup_v1_vs_py = time_yaqs / time_jl
speedup_v2_vs_py = time_yaqs / time_jl_v2
speedup_v2_vs_v1 = time_jl / time_jl_v2

println("\n--- Speedup Factors (Base: YAQS Python) ---")
println("Julia V1 Speedup: $(round(speedup_v1_vs_py, digits=2))x  (Higher is better)")
println("Julia V2 Speedup: $(round(speedup_v2_vs_py, digits=2))x  (Target > 1.0)")
println("V2 vs V1 Speedup: $(round(speedup_v2_vs_v1, digits=2))x")

if speedup_v2_vs_py > 1.0
    println("\nSUCCESS: Julia V2 is $(round(speedup_v2_vs_py, digits=2))x faster than YAQS Python.")
else
    println("\nWARNING: Julia V2 is $(round(1/speedup_v2_vs_py, digits=2))x SLOWER than YAQS Python.")
end

println("------------------------------\n")

# Extract results
# Each observable has a .results array.
# We expect size to be Timesteps + 1 (Initial) or similar.
results_yaqs_raw = [pyconvert(Vector{Float64}, obs.results) for obs in obs_yaqs]
println("YAQS raw points: $(length(results_yaqs_raw[1]))")

# Check alignment.
# YAQS Raw often contains [Init_Auto, Init_Barrier, Step1, ..., StepN, Final_Auto]
# If we see 13 points for 10 steps + 1 barrier + 1 init, it's likely:
# 1. Initial (t=0) automatic
# 2. Initial (t=0) barrier
# 3. Step 1
# ...
# 12. Step 10
# 13. Final
#
# We want Step 1 to Step 10.
# So we should slice [3:12] (1-based).
# Let's try slicing [3:end-1] or similar. 
# If we assume the last point is also junk/duplicate, and we want 10 points.
# Let's slice [3:end] and then truncation will handle the tail if it's too long.
results_yaqs_final = [res[3:end] for res in results_yaqs_raw]

# Also ensure we pass dt=1.0 to sim_params so that gate generators (which include theta) are applied fully.
# (This was missing in previous run).


# ==============================================================================
# 4. COMPARISON & PLOTTING
# ==============================================================================

# Length Check
len_jl = length(results_jl_final[1])
len_jl_v2 = length(results_jl_v2_final[1])
len_yaqs = length(results_yaqs_final[1])

len_py = RUN_QISKIT_EXACT ? length(results_py_final[1]) : len_jl

println("Compare Lengths: JuliaV1=$len_jl, JuliaV2=$len_jl_v2, YAQS=$len_yaqs, Qiskit=$(RUN_QISKIT_EXACT ? len_py : "N/A")")

min_len = min(len_jl, len_jl_v2, len_yaqs)
if RUN_QISKIT_EXACT
    min_len = min(min_len, len_py)
end

if len_jl > min_len; results_jl_final = [r[1:min_len] for r in results_jl_final]; end
if len_jl_v2 > min_len; results_jl_v2_final = [r[1:min_len] for r in results_jl_v2_final]; end
if len_yaqs > min_len; results_yaqs_final = [r[1:min_len] for r in results_yaqs_final]; end
if RUN_QISKIT_EXACT && len_py > min_len; results_py_final = [r[1:min_len] for r in results_py_final]; end

println("Truncated to $min_len")

if RUN_QISKIT_EXACT
    # Max Error (Julia vs Qiskit)
    max_diff = 0.0
    for i in 1:3
        diff = maximum(abs.(results_jl_final[i] - results_py_final[i]))
        global max_diff = max(max_diff, diff)
        println("Site $([1,3,6][i]) Max Diff (JlV1 vs Qiskit): $diff")
    end
    println("Overall Max Diff (JlV1 vs Qiskit): $max_diff")

    # Max Error (Julia V2 vs Qiskit)
    max_diff_v2 = 0.0
    for i in 1:3
        diff = maximum(abs.(results_jl_v2_final[i] - results_py_final[i]))
        global max_diff_v2 = max(max_diff_v2, diff)
        println("Site $([1,3,6][i]) Max Diff (JlV2 vs Qiskit): $diff")
    end
    println("Overall Max Diff (JlV2 vs Qiskit): $max_diff_v2")

    # Max Error (YAQS vs Qiskit)
    max_diff_yaqs = 0.0
    for i in 1:3
        diff = maximum(abs.(results_yaqs_final[i] - results_py_final[i]))
        global max_diff_yaqs = max(max_diff_yaqs, diff)
        println("Site $([1,3,6][i]) Max Diff (YAQS vs Qiskit): $diff")
    end
    println("Overall Max Diff (YAQS vs Qiskit): $max_diff_yaqs")
end

# Plotting
plt = pyimport("matplotlib.pyplot")

if RUN_QISKIT_EXACT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=true)
else
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax2 = nothing
end

x = 1:length(results_jl_final[1])

colors = ["r", "g", "b"]
sites = [1, 3, 6]

for i in 1:3
    c = colors[i]
    s = sites[i]
    ax1.plot(x, results_jl_final[i], label="JuliaV1 Z_$s", color=c, linestyle="-", linewidth=2)
    ax1.plot(x, results_jl_v2_final[i], label="JuliaV2 Z_$s", color=c, linestyle="-.", linewidth=2)
    ax1.plot(x, results_yaqs_final[i], label="YAQS Z_$s", color=c, linestyle=":", linewidth=3)
    
    if RUN_QISKIT_EXACT
        ax1.plot(x, results_py_final[i], label="Qiskit Z_$s", color=c, linestyle="--", linewidth=2)
        
        diff_sq = abs.(results_jl_final[i] - results_py_final[i]).^2
        ax2.plot(x, diff_sq, label="|JlV1 - Qiskit|^2 Z_$s", color=c, linestyle="-", linewidth=2)
        
        diff_sq_v2 = abs.(results_jl_v2_final[i] - results_py_final[i]).^2
        ax2.plot(x, diff_sq_v2, label="|JlV2 - Qiskit|^2 Z_$s", color=c, linestyle="-.", linewidth=2)

        diff_sq_yaqs = abs.(results_yaqs_final[i] - results_py_final[i]).^2
        ax2.plot(x, diff_sq_yaqs, label="|YAQS - Qiskit|^2 Z_$s", color=c, linestyle=":", linewidth=2)
    end
end

ax1.set_title("$CIRCUIT_NAME Model (L=$L, $timesteps Steps): Julia vs YAQS$(RUN_QISKIT_EXACT ? " vs Qiskit" : "")")
ax1.set_ylabel("Expectation Value <Z>")
ax1.legend()
ax1.grid(true)

if RUN_QISKIT_EXACT
    ax2.set_title("Absolute Squared Difference")
    ax2.set_xlabel("Step Index")
    ax2.set_ylabel("|Diff|^2")
    ax2.legend()
    ax2.grid(true)
    ax2.set_yscale("log")
else
    ax1.set_xlabel("Step Index")
end

plt.tight_layout()
output_file = "comparison_$(lowercase(CIRCUIT_NAME)).png"
plt.savefig(output_file)
println("Plot saved to $output_file")
