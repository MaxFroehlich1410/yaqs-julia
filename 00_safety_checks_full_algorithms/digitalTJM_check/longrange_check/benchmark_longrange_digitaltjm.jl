using LinearAlgebra
using PythonCall
using Random
using Printf
using Statistics
using Dates
using Pickle

# Include Yaqs source
include("../../../src/Yaqs.jl")
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.GateLibrary
using .Yaqs.NoiseModule
using .Yaqs.SimulationConfigs
using .Yaqs.DigitalTJMV2
using .Yaqs.DigitalTJM: DigitalCircuit, add_gate!, DigitalGate
using .Yaqs.Simulator
using .Yaqs.CircuitIngestion
using .Yaqs.CircuitLibrary

# ==============================================================================
# CONFIGURATION
# ==============================================================================

NUM_QUBITS = 12
NUM_LAYERS = 30
TAU = 0.1
NOISE_STRENGTH = 1e-05
NUM_TRAJECTORIES = 500

# Circuit selection: "XY_longrange" or "longrange_test"
CIRCUIT_TYPE = "XY_longrange" # Use "longrange_test" to test long-range noise isolation

pauli_y_error = false
pauli_x_error = true
pauli_z_error = false

# Observables
OBSERVABLE_BASIS = "Z"

# ==============================================================================
# PYTHON SETUP (Only for Circuit Generation)
# ==============================================================================

sys = pyimport("sys")
# Ensure local modules can be imported
sys.path.append(abspath(@__DIR__))
# Add path to src so Qiskit_simulator can be imported as a package
sys.path.append(joinpath(abspath(@__DIR__), "../../../src"))

qiskit = pyimport("qiskit")
aer_noise = pyimport("qiskit_aer.noise")

# Import local circuit library
circuit_lib = pyimport("Qiskit_simulator.circuit_library")

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

function _format_float_short(value::Float64)
    return replace(@sprintf("%.4g", value), "." => "p")
end

function _build_experiment_name(num_qubits, num_layers, tau, noise_strength, run_density_matrix, threshold_mse, fixed_trajectories, basis_label, observable_basis="Z", error_suffix=nothing)
    tokens = [
        "unraveling_eff",
        "N$(num_qubits)",
        "L$(num_layers)",
        "tau$(_format_float_short(tau))",
        "noise$(_format_float_short(noise_strength))",
        "basis$(basis_label)",
        "obs$(observable_basis)",
        run_density_matrix ? "modeDM" : "modeLarge"
    ]
    if run_density_matrix
        push!(tokens, "mse$(_format_float_short(threshold_mse))")
    else
        push!(tokens, "traj$(fixed_trajectories)")
    end
    if !isnothing(error_suffix)
        push!(tokens, "err$(error_suffix)")
    end
    return "JULIA_" * join(tokens, "_")
end

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

println("Configuration: N=$NUM_QUBITS, Layers=$NUM_LAYERS, Noise=$NOISE_STRENGTH")

# 1. Prepare Circuits (Using Python Qiskit for definition)
println("Building Circuits...")
println("Circuit type: $CIRCUIT_TYPE")

if CIRCUIT_TYPE == "longrange_test"
    # Use the test circuit that isolates long-range noise effects
    basis_label = "longrange_test"
    # Use π/4 as a standard test angle
    test_theta = π/4
    trotter_step = circuit_lib.longrange_test_circuit(NUM_QUBITS, test_theta)
    
    # For test circuit, start from |0...0⟩ (all zeros)
    init_circuit = qiskit.QuantumCircuit(NUM_QUBITS)
    
    # Julia Circuit Construction
    circ_jl = DigitalCircuit(NUM_QUBITS)
    # Add initial sample barrier for t=0
    add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    
    # Convert Python circuit to Julia gates
    py_instructions = trotter_step.data
    jl_gates_step = []
    for instr in py_instructions
        g = CircuitIngestion.convert_instruction_to_gate(instr, trotter_step)
        if !isnothing(g)
            push!(jl_gates_step, g)
        end
    end
    
    # For test circuit, we repeat the same layer multiple times
    for _ in 1:NUM_LAYERS
        for g in jl_gates_step
            add_gate!(circ_jl, g.op, g.sites; generator=g.generator)
        end
        # Add sample barrier after each step
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end
else
    # Original XY_longrange circuit
    basis_label = "XY_longrange"
    trotter_step = circuit_lib.xy_trotter_layer_longrange(NUM_QUBITS, TAU, order="YX")
    
    init_circuit = qiskit.QuantumCircuit(NUM_QUBITS)
    for i in 0:(NUM_QUBITS-1)
        if i % 4 == 3
            init_circuit.x(i)
        end
    end
    
    # Julia Circuit Construction
    circ_jl = DigitalCircuit(NUM_QUBITS)
    # Add initial sample barrier for t=0
    add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    
    py_instructions = trotter_step.data
    jl_gates_step = []
    for instr in py_instructions
        g = CircuitIngestion.convert_instruction_to_gate(instr, trotter_step)
        if !isnothing(g)
            # Skip Rz if returned (for consistency with original benchmark logic if needed)
            # The Python script uses xy_trotter_layer_longrange which uses RXX, RYY.
            push!(jl_gates_step, g)
        end
    end
    
    for _ in 1:NUM_LAYERS
        for g in jl_gates_step
            add_gate!(circ_jl, g.op, g.sites; generator=g.generator)
        end
        # Add sample barrier after each step
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end
end

# 2. Noise Setup
processes_jl_dicts = Vector{Dict{String, Any}}()


if pauli_x_error
    # 1. Single site X error on ALL qubits
    for i in 1:NUM_QUBITS
        d = Dict{String, Any}()
        d["name"] = "pauli_x"
        d["sites"] = [i]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
    end

     # 2. Two-site Crosstalk XX on Neighboring Sites + Periodic Boundary
    # Nearest neighbors [i, i+1]
    for i in 1:(NUM_QUBITS-1)
        d = Dict{String, Any}()
        d["name"] = "crosstalk_xx"
        d["sites"] = [i, i+1]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
    end

    # Periodic Term [1, N]
    d_periodic = Dict{String, Any}()
    d_periodic["name"] = "crosstalk_xx"
    d_periodic["sites"] = [1, NUM_QUBITS]
    d_periodic["strength"] = NOISE_STRENGTH
    push!(processes_jl_dicts, d_periodic)
end



if pauli_y_error
    # 1. Single site Y error on ALL qubits
    for i in 1:NUM_QUBITS
        d = Dict{String, Any}()
        d["name"] = "pauli_y"
        d["sites"] = [i]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
    end

    # 2. Two-site Crosstalk YY on Neighboring Sites + Periodic Boundary
    # Nearest neighbors [i, i+1]
    for i in 1:(NUM_QUBITS-1)
        d = Dict{String, Any}()
        d["name"] = "crosstalk_yy"
        d["sites"] = [i, i+1]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
    end

    # Periodic Term [1, N]
    d_periodic = Dict{String, Any}()
    d_periodic["name"] = "crosstalk_yy"
    d_periodic["sites"] = [1, NUM_QUBITS]
    d_periodic["strength"] = NOISE_STRENGTH
    push!(processes_jl_dicts, d_periodic)

end



if pauli_z_error
    # 1. Single site Y error on ALL qubits
    for i in 1:NUM_QUBITS
        d = Dict{String, Any}()
        d["name"] = "pauli_z"
        d["sites"] = [i]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
    end

    # 2. Two-site Crosstalk YY on Neighboring Sites + Periodic Boundary
    # Nearest neighbors [i, i+1]
    for i in 1:(NUM_QUBITS-1)
        d = Dict{String, Any}()
        d["name"] = "crosstalk_zz"
        d["sites"] = [i, i+1]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
    end

    # Periodic Term [1, N]
    d_periodic = Dict{String, Any}()
    d_periodic["name"] = "crosstalk_zz"
    d_periodic["sites"] = [1, NUM_QUBITS]
    d_periodic["strength"] = NOISE_STRENGTH
    push!(processes_jl_dicts, d_periodic)

end


noise_model_jl = NoiseModel(processes_jl_dicts, NUM_QUBITS)


# 3. Runner
function run_simulation_julia()
    # Initial State
    psi = MPS(NUM_QUBITS; state="zeros")

    # Apply initial state preparation based on circuit type
    if CIRCUIT_TYPE == "longrange_test"
        # Test circuit starts from |0...0⟩, no X gates needed
        println("Initial state: |0...0⟩ (all zeros)")
    else
        # XY_longrange: Apply initial X gates to match Qiskit setup (indices 3, 7, ... in 0-based => 4, 8, ... in 1-based)
        # Qiskit: if i % 4 == 3 -> X(i)
        println("Applying initial state preparation (X gates)...")
        for i in 1:NUM_QUBITS
            if (i - 1) % 4 == 3
                println("  Applying X to site $i")
                T = psi.tensors[i]
                # Apply X gate: Swap physical indices 1 and 2
                # T is (Left, Phys, Right)
                T_new = zeros(ComplexF64, size(T))
                T_new[:, 1, :] = T[:, 2, :]
                T_new[:, 2, :] = T[:, 1, :]
                psi.tensors[i] = T_new
            end
        end
    end

    # Observables
    # Python saves "Z" expectation on all qubits.
    obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]

    # We want results at t=0, 1, ..., NUM_LAYERS.
    # TimeEvolutionConfig computes points.
    sim_params = TimeEvolutionConfig(obs, Float64(NUM_LAYERS); dt=1.0, num_traj=NUM_TRAJECTORIES, max_bond_dim=64, truncation_threshold=1e-6)

    println("Running Julia DigitalTJM ($NUM_TRAJECTORIES trajectories)...")
    # Parallel execution
    Simulator.run(psi, circ_jl, sim_params, noise_model_jl; parallel=true)


    # Shape: (num_qubits, num_layers + 1)
    expvals_mean = zeros(Float64, NUM_QUBITS, length(sim_params.times))
    expvals_var = zeros(Float64, NUM_QUBITS, length(sim_params.times))

    for (i, o) in enumerate(obs)
        # o.trajectories is Matrix [num_traj, num_times]
        # Real part of Z
        data = real.(o.trajectories)
        expvals_mean[i, :] = vec(mean(data, dims=1))
        expvals_var[i, :] = vec(var(data, dims=1))
    end
    
    # We will return dummy bonds for now as tracking full bond history requires modification.
    bonds = Dict(
        "per_shot_per_layer_max_bond_dim" => zeros(Int, NUM_TRAJECTORIES, length(sim_params.times)-1), # Mock
        "per_layer_mean_across_shots" => zeros(Float64, length(sim_params.times)-1)
    )

    return expvals_mean, expvals_var, bonds
end


# 4. Execution & Saving

expvals, vars, bonds = run_simulation_julia()

# Calculate Staggered Magnetization series
num_steps = size(expvals, 2)
stag_series = [staggered_magnetization(expvals[:, t], NUM_QUBITS) for t in 1:num_steps]

# Local Expvals: list of arrays
local_expvals = [expvals[:, t] for t in 1:num_steps]

# Variance: Dict
# "staggered" -> sum(var) / N^2
var_staggered = vec(sum(vars, dims=1)) ./ (NUM_QUBITS^2)
middle_qubit = NUM_QUBITS ÷ 2 + 1 # 1-based
var_middle = vars[middle_qubit, :]

variance_dict = Dict(
    "staggered" => var_staggered,
    "local_middle" => var_middle
)

# Build Python-compatible results dictionary to avoid 'juliacall' dependency in pickle
    np = pyimport("numpy")
    py_dict = pybuiltins.dict
    
    # Convert data to Python types
    py_stag_series = np.array(stag_series)
    
    # local_expvals is Vector{Vector{Float64}} -> convert to List of Arrays or 2D Array
    # Python script expects: list of (num_qubits,) arrays -> which becomes (Time, N) when stacked
    # We can just send a list of numpy arrays
    py_local_expvals = PyList([np.array(v) for v in local_expvals])
    
    py_bonds = py_dict()
    py_bonds["per_shot_per_layer_max_bond_dim"] = np.array(bonds["per_shot_per_layer_max_bond_dim"])
    py_bonds["per_layer_mean_across_shots"] = np.array(bonds["per_layer_mean_across_shots"])

    py_variance = py_dict()
    py_variance["staggered"] = np.array(variance_dict["staggered"])
    py_variance["local_middle"] = np.array(variance_dict["local_middle"])

    py_results_inner = py_dict()
    py_results_inner["trajectories"] = NUM_TRAJECTORIES
    py_results_inner["mse"] = nothing
    py_results_inner["staggered_magnetization"] = py_stag_series
    py_results_inner["local_expvals"] = py_local_expvals
    py_results_inner["bonds"] = py_bonds
    py_results_inner["variance"] = py_variance

    results = py_dict()
    results["Julia V2"] = py_results_inner

    # Build Filename
    # Construct error suffix based on active flags
    suffixes = String[]
    if pauli_x_error push!(suffixes, "X") end
    if pauli_y_error push!(suffixes, "Y") end
    if pauli_z_error push!(suffixes, "Z") end
    # Sort to match Python's alphabetical sorting if multiple
    sort!(suffixes)
    error_suffix = isempty(suffixes) ? "None" : join(suffixes)

    experiment_name = _build_experiment_name(
        NUM_QUBITS, NUM_LAYERS, TAU, NOISE_STRENGTH,
        false, nothing, NUM_TRAJECTORIES, # run_density_matrix=false, threshold=nothing
        basis_label, OBSERVABLE_BASIS, error_suffix
    )

    filename = "$(experiment_name)_results.pkl"
    parent_dir = joinpath(dirname(@__FILE__), "results")
    if !isdir(parent_dir)
        mkdir(parent_dir)
    end
    filepath = joinpath(parent_dir, filename)

    println("Saving results to $filepath")
    
    # Save using Python Pickle
    # Use Python's open for binary write compatibility with pickle
    builtins = pyimport("builtins")
    f = builtins.open(filepath, "wb")
    pickle = pyimport("pickle")
    pickle.dump(results, f)
    f.close()
    println("Done.")

