using LinearAlgebra
using PythonCall
using Random
using Printf
using Statistics
using Dates

# Include Yaqs source
include("../src/Yaqs.jl")
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.GateLibrary
using .Yaqs.NoiseModule
using .Yaqs.SimulationConfigs
using .Yaqs.DigitalTJMV2 
using .Yaqs.DigitalTJM: DigitalCircuit, add_gate!, DigitalGate
using .Yaqs.Simulator

# ==============================================================================
# CONFIGURATION
# ==============================================================================

NUM_QUBITS = 6
NUM_LAYERS = 20
TAU = 0.1
NOISE_STRENGTH = 0.01
NUM_TRAJECTORIES = 1000 # Smaller for periodic test as it's slower
THRESHOLD_MSE = 1e-4

# Enable Periodic BC (Long Range)
PERIODIC = true

# Observables
OBSERVABLE_BASIS = "Z"

# Flags
RUN_QISKIT_MPS = true
RUN_PYTHON_YAQS = false # Skip Python YAQS for this specific test if needed, or try
RUN_JULIA_V2 = true

# ==============================================================================
# PYTHON SETUP
# ==============================================================================

sys = pyimport("sys")
mqt_yaqs_src = abspath(joinpath(@__DIR__, "../../mqt-yaqs/src"))
mqt_yaqs_inner = abspath(joinpath(@__DIR__, "../../mqt-yaqs/src/mqt/yaqs"))

if !(mqt_yaqs_src in sys.path)
    sys.path.insert(0, mqt_yaqs_src)
end
if !(mqt_yaqs_inner in sys.path)
    sys.path.insert(0, mqt_yaqs_inner)
end

qiskit = pyimport("qiskit")
aer_noise = pyimport("qiskit_aer.noise")

# Import MQT YAQS modules
try
    global mqt_circuit_lib = pyimport("mqt.yaqs.core.libraries.circuit_library")
    global mqt_simulators = pyimport("codex_experiments.worker_functions.qiskit_simulators")
    global mqt_sim = pyimport("mqt.yaqs.simulator")
    global mqt_networks = pyimport("mqt.yaqs.core.data_structures.networks")
    global mqt_params = pyimport("mqt.yaqs.core.data_structures.simulation_parameters")
    global mqt_gates = pyimport("mqt.yaqs.core.libraries.gate_library")
    global mqt_noise_utils = pyimport("codex_experiments.worker_functions.yaqs_simulator")
catch e
    println("Error importing Python modules: ", e)
    println("Sys path: ", sys.path)
    rethrow(e)
end

# ==============================================================================
# HELPERS
# ==============================================================================

function staggered_magnetization(expvals::Vector{Float64}, L::Int)
    sum_val = 0.0
    for i in 1:L
        sum_val += (-1)^(i-1) * expvals[i]
    end
    return sum_val / L
end

function compute_mse(pred_series::Vector{Float64}, exact_series::Vector{Float64})
    if length(pred_series) != length(exact_series)
        min_len = min(length(pred_series), length(exact_series))
        return mean((pred_series[1:min_len] .- exact_series[1:min_len]).^2)
    end
    return mean((pred_series .- exact_series).^2)
end

# ==============================================================================
# TRAJECTORY FINDER
# ==============================================================================

function run_trajectories(runner_single_shot::Function, 
                          exact_stag::Union{Vector{Float64}, Nothing}, 
                          label::String)
    
    println("\n--- Running $label ---")
    cumulative_results = nothing
    bond_dims = Int[]
    final_stag = Float64[]
    final_mse = 0.0
    
    for n in 1:NUM_TRAJECTORIES
        res_mat, bond = runner_single_shot()
        if isnothing(cumulative_results)
            cumulative_results = copy(res_mat)
        else
            cumulative_results .+= res_mat
        end
        push!(bond_dims, bond)
        
        if n % 10 == 0 || n == NUM_TRAJECTORIES
            avg_res = cumulative_results ./ n
            T_steps = size(avg_res, 2)
            current_stag = [staggered_magnetization(avg_res[:, t], NUM_QUBITS) for t in 1:T_steps]
            
            mse_str = ""
            if !isnothing(exact_stag)
                mse = compute_mse(current_stag, exact_stag)
                mse_str = @sprintf("MSE=%.2e", mse)
                final_mse = mse
            end
            
            @printf "  Traj %d/%d: %s\n" n NUM_TRAJECTORIES mse_str
        end
    end
    
    avg_res = cumulative_results ./ NUM_TRAJECTORIES
    T_steps = size(avg_res, 2)
    final_stag = [staggered_magnetization(avg_res[:, t], NUM_QUBITS) for t in 1:T_steps]
    return NUM_TRAJECTORIES, final_mse, final_stag
end

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

println("Configuration: Periodic=$PERIODIC, N=$NUM_QUBITS, Layers=$NUM_LAYERS, Noise=$NOISE_STRENGTH")

# 1. Prepare Circuits
println("Building Circuits...")
trotter_step = mqt_circuit_lib.create_heisenberg_circuit(NUM_QUBITS, 1.0, 1.0, 1.0, 0.0, TAU, 1, periodic=PERIODIC)

init_circuit = qiskit.QuantumCircuit(NUM_QUBITS)
for i in 0:(NUM_QUBITS-1)
    if i % 4 == 3
        init_circuit.x(i)
    end
end

# Julia Circuit Construction
circ_jl = DigitalCircuit(NUM_QUBITS)
add_gate!(circ_jl, GateLibrary.Barrier("SAMPLE_OBSERVABLES"), Int[])

py_instructions = trotter_step.data
jl_gates_step = []
for instr in py_instructions
    g = CircuitIngestion.convert_instruction_to_gate(instr, trotter_step)
    if !isnothing(g)
        # Skip Rz if returned (for consistency with original benchmark logic)
        if g.op isa GateLibrary.RzGate
             continue
        end
        push!(jl_gates_step, g)
    end
end

for _ in 1:NUM_LAYERS
    for g in jl_gates_step
        add_gate!(circ_jl, g.op, g.sites; generator=g.generator)
    end
    add_gate!(circ_jl, GateLibrary.Barrier("SAMPLE_OBSERVABLES"), Int[])
end

# 2. Noise Setup
processes_jl_dicts = Vector{Dict{String, Any}}()

# 1. Single site Y error on ALL qubits
for i in 1:NUM_QUBITS
    d = Dict{String, Any}()
    d["name"] = "pauli_y"
    d["sites"] = [i]
    d["strength"] = NOISE_STRENGTH
    push!(processes_jl_dicts, d)
end

# 2. Two-site Crosstalk YY on ALL pairs (Neighboring + Long Range)
# We iterate all unique pairs (i, j) with i < j
for i in 1:NUM_QUBITS
    for j in (i+1):NUM_QUBITS
        d = Dict{String, Any}()
        d["name"] = "crosstalk_yy"
        d["sites"] = [i, j]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
    end
end

noise_model_jl = NoiseModel(processes_jl_dicts, NUM_QUBITS)

# Qiskit Noise
# Note: This Qiskit model applies error_2q to 'rzz', 'rxx', 'ryy' gates.
# In the Heisenberg circuit, these gates exist for neighbors and the periodic bond (1-N).
# It does NOT apply noise to non-connected long-range pairs unless gates exist there.
# If Julia has all-to-all noise, it will be noisier than Qiskit.
# To make a fair comparison for "all-to-all" noise, we would need all-to-all gates or ID gates.
# However, to avoid altering the circuit structure and follow "qiskit noise as it was before",
# we keep this. If MSE is high, it's due to the extra noise in Julia.
#
# BUT: To respect the user's implicit desire for matching results ("verify... results match"),
# we might need to restrict Julia to only gate-connected pairs?
# The prompt says: "julia noise should again be... yy on all long range qubit pairs".
# This sounds explicit. I will stick to the explicit instruction.

qiskit_noise_model = aer_noise.NoiseModel()
# Probabilities: 3*p for I, p for IY, p for YI, p for YY?
# Wait, previous was: ("II", 1-3p), ("IY", p), ("YI", p), ("YY", p).
# This is a specific channel.
ops_2q = [("II", 1.0 - 3*NOISE_STRENGTH), ("IY", NOISE_STRENGTH), ("YI", NOISE_STRENGTH), ("YY", NOISE_STRENGTH)]
error_2q = aer_noise.errors.pauli_error(pybuiltins.list(ops_2q))
inst_2q = pybuiltins.list(["rzz", "rxx", "ryy"])
qiskit_noise_model.add_all_qubit_quantum_error(error_2q, inst_2q)

ops_1q = [("Y", NOISE_STRENGTH), ("I", 1.0 - NOISE_STRENGTH)]
error_1q = aer_noise.errors.pauli_error(pybuiltins.list(ops_1q))
inst_1q = pybuiltins.list(["x", "y", "z", "rx", "ry", "rz"])
qiskit_noise_model.add_all_qubit_quantum_error(error_1q, inst_1q)


# 3. Exact Reference
exact_stag_ref = nothing
println("Computing Exact Reference (Density Matrix)...")
ref_job = mqt_simulators.run_qiskit_exact(
    NUM_QUBITS, NUM_LAYERS, init_circuit, trotter_step, qiskit_noise_model,
    method="density_matrix", observable_basis=OBSERVABLE_BASIS
)
ref_expvals = pyconvert(Matrix{Float64}, ref_job)
T_steps_ref = size(ref_expvals, 2)
exact_stag_ref = [staggered_magnetization(ref_expvals[:, t], NUM_QUBITS) for t in 1:T_steps_ref]
println("Reference computed. Length: $T_steps_ref")


# 4. Runners
function runner_julia_v2()
    psi = MPS(NUM_QUBITS; state="zeros")
    for i in 1:NUM_QUBITS
        if (i-1) % 4 == 3
            DigitalTJMV2.apply_single_qubit_gate!(psi, DigitalGate(XGate(), [i], nothing))
        end
    end
    
    obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]
    sim_params = TimeEvolutionConfig(obs, Float64(NUM_LAYERS); dt=1.0, num_traj=1, max_bond_dim=64, truncation_threshold=1e-6)
    
    Simulator.run(psi, circ_jl, sim_params, noise_model_jl; parallel=false)
    
    results = zeros(ComplexF64, length(obs), length(sim_params.times))
    for (i, o) in enumerate(obs)
        results[i, :] = o.trajectories[1, :]
    end
    
    bond_dim = MPSModule.write_max_bond_dim(psi)
    return real.(results), bond_dim
end

function runner_qiskit_mps()
    res_tuple = mqt_simulators.run_qiskit_mps(
        NUM_QUBITS, NUM_LAYERS, init_circuit, trotter_step, qiskit_noise_model,
        num_traj=1, observable_basis=OBSERVABLE_BASIS
    )
    expvals = pyconvert(Matrix{Float64}, res_tuple[0])
    return expvals, 0
end

# 5. Execution
results_data = Dict()

if RUN_QISKIT_MPS
    try
        n, mse, stag = run_trajectories(runner_qiskit_mps, exact_stag_ref, "Qiskit MPS")
        results_data["Qiskit MPS"] = (n, mse, stag)
    catch e
        println("Qiskit MPS Failed: $e")
    end
end

if RUN_JULIA_V2
    try
        n, mse, stag = run_trajectories(runner_julia_v2, exact_stag_ref, "Julia V2")
        results_data["Julia V2"] = (n, mse, stag)
    catch e
        println("Julia V2 Failed: $e")
        rethrow(e)
    end
end

# 6. Plotting
println("\nGenerating Plot...")
plt = pyimport("matplotlib.pyplot")
fig, ax = plt.subplots(figsize=(10, 6))
x_axis = 0:NUM_LAYERS

if !isnothing(exact_stag_ref)
    len = length(exact_stag_ref)
    ax.plot(x_axis[1:len], exact_stag_ref, "k-", linewidth=2, label="Exact (DM)")
end

colors = Dict("Qiskit MPS"=>"b", "Julia V2"=>"r")

for (name, (n, mse, stag)) in results_data
    c = get(colors, name, "k")
    if length(stag) > length(x_axis)
        stag = stag[1:length(x_axis)]
    end
    ax.plot(x_axis[1:length(stag)], stag, "--o", color=c, label="$name (MSE=$(@sprintf("%.1e", mse)))")
end

ax.set_xlabel("Layer")
ax.set_ylabel("Staggered Magnetization")
ax.set_title("Periodic Benchmark (N=$NUM_QUBITS, Noise=$NOISE_STRENGTH)")
ax.legend()
ax.grid(true)

fname = "benchmark_periodic_jl.png"
plt.savefig(fname)
println("Saved plot to $fname")

