using LinearAlgebra
using PythonCall
using Random
using Printf
using Statistics
using Dates

# Include Yaqs source
include("../../../src/Yaqs.jl")
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.GateLibrary
using .Yaqs.NoiseModule
using .Yaqs.SimulationConfigs
using .Yaqs.DigitalTJM: DigitalCircuit, add_gate!, DigitalGate

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Simulation Size
NUM_QUBITS = 6
NUM_LAYERS = 20
TAU = 0.1

# Noise
NOISE_STRENGTH = 0.01
ENABLE_X_ERROR = true
ENABLE_Y_ERROR = true
ENABLE_Z_ERROR = true

# Unraveling
NUM_TRAJECTORIES = 100
MODE = "DM" # "DM" to verify against Density Matrix, "Large" for just performance

# Observables
OBSERVABLE_BASIS = "Z"
THRESHOLD_MSE = 1e-3
SITES_TO_PLOT = [1,3,6] # 1-based index, will be adjusted for Python/Plots

# Flags
RUN_QISKIT_MPS = true
RUN_PYTHON_YAQS = true
RUN_JULIA_V2 = true

# Circuit Selection
CIRCUIT = "longrange_test" # Options: "Heisenberg", "Ising"
TAU = 0.1

# ==============================================================================
# PYTHON SETUP
# ==============================================================================

sys = pyimport("sys")
# Add mqt-yaqs paths
mqt_yaqs_src = abspath(joinpath(@__DIR__, "../../../../mqt-yaqs/src"))
mqt_yaqs_inner = abspath(joinpath(@__DIR__, "../../../../mqt-yaqs/src/mqt/yaqs"))

if !(mqt_yaqs_src in sys.path)
    sys.path.insert(0, mqt_yaqs_src)
end
if !(mqt_yaqs_inner in sys.path)
    sys.path.insert(0, mqt_yaqs_inner)
end

qiskit = pyimport("qiskit")
aer_noise = pyimport("qiskit_aer.noise")
quantum_info = pyimport("qiskit.quantum_info")
oe = pyimport("opt_einsum")

# Import MQT YAQS modules
# Note: These paths depend on mqt-yaqs package structure
try
    global mqt_circuit_lib = pyimport("mqt.yaqs.core.libraries.circuit_library")
    global mqt_simulators = pyimport("mqt.yaqs.codex_experiments.worker_functions.qiskit_simulators")
    global mqt_sim = pyimport("mqt.yaqs.simulator")
    global mqt_networks = pyimport("mqt.yaqs.core.data_structures.networks")
    global mqt_params = pyimport("mqt.yaqs.core.data_structures.simulation_parameters")
    global mqt_gates = pyimport("mqt.yaqs.core.libraries.gate_library")
    global mqt_noise_utils = pyimport("mqt.yaqs.codex_experiments.worker_functions.yaqs_simulator")
    global mqt_noise_model = pyimport("mqt.yaqs.core.data_structures.noise_model")
catch e
    println("Error importing Python modules: ", e)
    println("Sys path: ", sys.path)
    rethrow(e)
end

# Conditionally import local circuit library if needed
# This is only needed for circuits defined in src/Qiskit_simulator/circuit_library.py
local_circuit_lib = nothing
if CIRCUIT == "longrange_test"
    # Add local src path for Qiskit_simulator
    # @__DIR__ is: 00_safety_checks_full_algorithms/digitalTJM_check/longrange_check/
    # Go up 3 levels to repo root, then into src
    # Use normpath to ensure proper path resolution
    local_src_path = normpath(joinpath(@__DIR__, "../../../src"))
    # Verify the path exists and contains Qiskit_simulator
    if !isdir(local_src_path)
        # Fallback: try to find repo root by looking for Project.toml
        current = @__DIR__
        for _ in 1:5  # Try up to 5 levels up
            current = dirname(current)
            candidate = joinpath(current, "src")
            if isdir(candidate) && isfile(joinpath(current, "Project.toml"))
                local_src_path = candidate
                println("Found repo root by searching for Project.toml: ", current)
                break
            end
        end
        if !isdir(local_src_path)
            error("Local src path does not exist: $local_src_path (tried: $(normpath(joinpath(@__DIR__, "../../../src")))")
        end
    end
    qiskit_sim_path = joinpath(local_src_path, "Qiskit_simulator")
    if !isdir(qiskit_sim_path)
        error("Qiskit_simulator directory not found at: $qiskit_sim_path")
    end
    if !(local_src_path in sys.path)
        sys.path.insert(0, local_src_path)
    end
    try
        global local_circuit_lib = pyimport("Qiskit_simulator.circuit_library")
        println("Successfully imported local circuit library from: ", local_src_path)
    catch e
        println("Error: Could not import local circuit library: ", e)
        println("Attempted path: ", local_src_path)
        println("Qiskit_simulator path: ", qiskit_sim_path)
        println("Directory exists: ", isdir(local_src_path))
        println("Qiskit_simulator exists: ", isdir(qiskit_sim_path))
        println("Sys path: ", sys.path)
        rethrow(e)
    end
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
        println("Warning: Series length mismatch $(length(pred_series)) vs $(length(exact_series))")
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
    
    # Warmup run to trigger JIT compilation (excluded from timing)
    println("  Warming up (JIT compilation)...")
    runner_single_shot()  # Discard result
    
    cumulative_results = nothing
    bond_dims = Int[]
    
    final_stag = Float64[]
    final_mse = 0.0
    
    # Measure pure simulation loop time (after warmup)
    t_start = time()
    
    for n in 1:NUM_TRAJECTORIES
        res_mat, bond = runner_single_shot()
        
        if isnothing(cumulative_results)
            cumulative_results = copy(res_mat)
        else
            cumulative_results .+= res_mat
        end
        push!(bond_dims, bond)
        
        # Periodic check
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
    
    t_elapsed = time() - t_start
    @printf "  Time: %.4f s (%.4f s/traj)\n" t_elapsed (t_elapsed / NUM_TRAJECTORIES)
    
    avg_res = cumulative_results ./ NUM_TRAJECTORIES
    T_steps = size(avg_res, 2)
    final_stag = [staggered_magnetization(avg_res[:, t], NUM_QUBITS) for t in 1:T_steps]
    
    return NUM_TRAJECTORIES, final_mse, final_stag, avg_res, t_elapsed
end

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

println("Configuration: N=$NUM_QUBITS, Layers=$NUM_LAYERS, Noise=$NOISE_STRENGTH")

# 1. Prepare Circuits
# -------------------
println("Building Circuits ($CIRCUIT)...")

if CIRCUIT == "Heisenberg"
    trotter_step = mqt_circuit_lib.create_heisenberg_circuit(NUM_QUBITS, 1.0, 1.0, 1.0, 0.0, TAU, 1, periodic=true)
elseif CIRCUIT == "Ising"
    # J=1.0 (ZZ), g=1.0 (X field)
    trotter_step = mqt_circuit_lib.create_ising_circuit(NUM_QUBITS, 1.0, 1.0, TAU, 1, periodic=true)
elseif CIRCUIT == "XY"
    # Heisenberg with Jz=0: XX + YY
    trotter_step = mqt_circuit_lib.xy_trotter_layer_longrange(NUM_QUBITS, TAU, order="YX")
elseif CIRCUIT == "longrange_test"
    # Use local circuit library for this specific test circuit
    if isnothing(local_circuit_lib)
        error("local_circuit_lib is not available. Make sure CIRCUIT='longrange_test' is set before Python imports.")
    end
    trotter_step = local_circuit_lib.longrange_test_circuit(NUM_QUBITS, 1.0)
else
    error("Unknown CIRCUIT: $CIRCUIT. Supported: Heisenberg, Ising, XY, longrange_test")
end

init_circuit = qiskit.QuantumCircuit(NUM_QUBITS)
for i in 0:(NUM_QUBITS-1)
    if i % 4 == 3
        init_circuit.x(i)
    end
end

# Julia Circuit Construction
circ_jl = DigitalCircuit(NUM_QUBITS)
# Initial State X gates are handled in state preparation, not circuit? 
# Actually runner_julia_v2 applies them.
# So we just need the evolution layers.

# Add initial barrier to capture t=0
add_gate!(circ_jl, GateLibrary.Barrier("SAMPLE_OBSERVABLES"), Int[])

# Extract gates from Trotter Step once
py_instructions = trotter_step.data
jl_gates_step = []
for instr in py_instructions
    g = CircuitIngestion.convert_instruction_to_gate(instr, trotter_step)
    if !isnothing(g)
        # Skip Rz for completeness/consistency with previous logic
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

# Pre-construct Python YAQS Circuit to match Julia's one-time construction
full_yaqs_circ = qiskit.QuantumCircuit(NUM_QUBITS)
full_yaqs_circ.barrier(label="SAMPLE_OBSERVABLES")
for _ in 1:NUM_LAYERS
    full_yaqs_circ.compose(trotter_step, inplace=true)
    full_yaqs_circ.barrier(label="SAMPLE_OBSERVABLES")
end


# 2. Noise Setup
# --------------
processes_jl_dicts = Vector{Dict{String, Any}}()
processes_py = [] # For Python YAQS
if ENABLE_X_ERROR
    for i in 1:NUM_QUBITS
        # Julia Dict
        d = Dict{String, Any}()
        d["name"] = "pauli_x"
        d["sites"] = [i]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        
        # Python Dict (0-based)
        push!(processes_py, Dict("name"=>"pauli_x", "sites"=>[i-1], "strength"=>NOISE_STRENGTH))
    end
    
    # Add Crosstalk XX on pairs
    for i in 1:(NUM_QUBITS-1)
        d = Dict{String, Any}()
        d["name"] = "crosstalk_xx"
        d["sites"] = [i, i+1]
        d["strength"] = NOISE_STRENGTH



    
        push!(processes_jl_dicts, d)
        
        push!(processes_py, Dict("name"=>"crosstalk_xx", "sites"=>[i-1, i], "strength"=>NOISE_STRENGTH))
    end

    # Periodic Term [1, N] (Julia uses 1-based, Python uses 0-based)
    d_periodic = Dict{String, Any}()
    d_periodic["name"] = "crosstalk_xx"
    d_periodic["sites"] = [1, NUM_QUBITS]  # Julia: 1-based
    d_periodic["strength"] = NOISE_STRENGTH
    push!(processes_py, Dict("name"=>"crosstalk_xx", "sites"=>[0, NUM_QUBITS-1], "strength"=>NOISE_STRENGTH))  # Python: 0-based
    push!(processes_jl_dicts, d_periodic)
end


if ENABLE_Y_ERROR
    for i in 1:NUM_QUBITS
        # Julia Dict
        d = Dict{String, Any}()
        d["name"] = "pauli_y"
        d["sites"] = [i]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        
        # Python Dict (0-based)
        push!(processes_py, Dict("name"=>"pauli_y", "sites"=>[i-1], "strength"=>NOISE_STRENGTH))
    end
    
    # Add Crosstalk YY on pairs
    for i in 1:(NUM_QUBITS-1)
        d = Dict{String, Any}()
        d["name"] = "crosstalk_yy"
        d["sites"] = [i, i+1]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        push!(processes_py, Dict("name"=>"crosstalk_yy", "sites"=>[i-1, i], "strength"=>NOISE_STRENGTH))
    end
    # Periodic Term [1, N] (Julia uses 1-based, Python uses 0-based)
    d_periodic = Dict{String, Any}()
    d_periodic["name"] = "crosstalk_yy"
    d_periodic["sites"] = [1, NUM_QUBITS]  # Julia: 1-based
    d_periodic["strength"] = NOISE_STRENGTH
    push!(processes_py, Dict("name"=>"crosstalk_yy", "sites"=>[0, NUM_QUBITS-1], "strength"=>NOISE_STRENGTH))  # Python: 0-based
    push!(processes_jl_dicts, d_periodic)
end


if ENABLE_Z_ERROR
    for i in 1:NUM_QUBITS
        # Julia Dict
        d = Dict{String, Any}()
        d["name"] = "pauli_z"
        d["sites"] = [i]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        push!(processes_py, Dict("name"=>"pauli_z", "sites"=>[i-1], "strength"=>NOISE_STRENGTH))
    end
    
    # Add Crosstalk ZZ on pairs
    for i in 1:(NUM_QUBITS-1)
        d = Dict{String, Any}()
        d["name"] = "crosstalk_zz"
        d["sites"] = [i, i+1]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        push!(processes_py, Dict("name"=>"crosstalk_zz", "sites"=>[i-1, i], "strength"=>NOISE_STRENGTH))
    end
    # Periodic Term [1, N] (Julia uses 1-based, Python uses 0-based)
    d_periodic = Dict{String, Any}()
    d_periodic["name"] = "crosstalk_zz"
    d_periodic["sites"] = [1, NUM_QUBITS]  # Julia: 1-based
    d_periodic["strength"] = NOISE_STRENGTH
    push!(processes_py, Dict("name"=>"crosstalk_zz", "sites"=>[0, NUM_QUBITS-1], "strength"=>NOISE_STRENGTH))  # Python: 0-based
    push!(processes_jl_dicts, d_periodic)
end

noise_model_jl = NoiseModel(processes_jl_dicts, NUM_QUBITS)

# # Qiskit Noise
# qiskit_noise_model = aer_noise.NoiseModel()
# inst_2q = pybuiltins.list(["cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz", "cu", "csx", "cs", "csdg", "swap", "iswap", "rxx", "ryy", "rzz", "rzx", "ecr", "dcx", "xx_minus_yy", "xx_plus_yy"])

# if ENABLE_X_ERROR
#     # ops_2q = [("IX", NOISE_STRENGTH), ("XI", NOISE_STRENGTH), ("XX", NOISE_STRENGTH)]
#     # error_2q = aer_noise.errors.pauli_error(pybuiltins.list(ops_2q))
#     generators_two_qubit = [quantum_info.Pauli("IX"), quantum_info.Pauli("XI"), quantum_info.Pauli("XX")]
#     error_2q = aer_noise.errors.PauliLindbladError(generators_two_qubit, [NOISE_STRENGTH, NOISE_STRENGTH, NOISE_STRENGTH])
#     for i in 1:(NUM_QUBITS-1)
#         qiskit_noise_model.add_quantum_error(error_2q, inst_2q, [i, i+1])  # Qiskit uses 0-based indexing
#     end
# end




# if ENABLE_Y_ERROR
#     # ops_2q = [("IY", NOISE_STRENGTH), ("YI", NOISE_STRENGTH), ("YY", NOISE_STRENGTH)]
#     # error_2q = aer_noise.errors.pauli_error(pybuiltins.list(ops_2q))
#     generators_two_qubit = [quantum_info.Pauli("IY"), quantum_info.Pauli("YI"), quantum_info.Pauli("YY")]
#     error_2q = aer_noise.errors.PauliLindbladError(generators_two_qubit, [NOISE_STRENGTH, NOISE_STRENGTH, NOISE_STRENGTH])
#     for i in 1:(NUM_QUBITS-1)
#         qiskit_noise_model.add_quantum_error(error_2q, inst_2q, [i, i+1])  # Qiskit uses 0-based indexing
#     end
# end


# if ENABLE_Z_ERROR
#     # ops_2q = [("IZ", NOISE_STRENGTH), ("ZI", NOISE_STRENGTH), ("ZZ", NOISE_STRENGTH)]
#     # error_2q = aer_noise.errors.pauli_error(pybuiltins.list(ops_2q))
#     generators_two_qubit = [quantum_info.Pauli("IZ"), quantum_info.Pauli("ZI"), quantum_info.Pauli("ZZ")]
#     error_2q = aer_noise.errors.PauliLindbladError(generators_two_qubit, [NOISE_STRENGTH, NOISE_STRENGTH, NOISE_STRENGTH])
#     for i in 1:(NUM_QUBITS-1)
#         qiskit_noise_model.add_quantum_error(error_2q, inst_2q, [i, i+1])  # Qiskit uses 0-based indexing
#     end
# end


# Qiskit Noise
qiskit_noise_model = aer_noise.NoiseModel()
inst_2q = pybuiltins.list([
    "cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz", "cu",
    "csx", "cs", "csdg", "swap", "iswap",
    "rxx", "ryy", "rzz", "rzx", "ecr", "dcx",
    "xx_minus_yy", "xx_plus_yy"
])

function pauli_lindblad_error_from_labels(labels::Vector{String}, rate::Float64)
    # Build a Python PauliList from label strings
    py_labels = pybuiltins.list(labels)
    pl = quantum_info.PauliList(py_labels)
    # Rates as Python list
    py_rates = pybuiltins.list(fill(rate, length(labels)))
    return aer_noise.errors.PauliLindbladError(pl, py_rates)
end

if ENABLE_X_ERROR
    # generators: IX, XI, XX
    error_2q_X = pauli_lindblad_error_from_labels(["IX", "XI", "XX"], NOISE_STRENGTH)
    for i in 1:(NUM_QUBITS-1)
        qiskit_noise_model.add_quantum_error(error_2q_X, inst_2q, [i-1, i])
    end
end

if ENABLE_Y_ERROR
    # generators: IY, YI, YY
    error_2q_Y = pauli_lindblad_error_from_labels(["IY", "YI", "YY"], NOISE_STRENGTH)
    for i in 1:(NUM_QUBITS-1)
        qiskit_noise_model.add_quantum_error(error_2q_Y, inst_2q, [i-1, i])
    end
end

if ENABLE_Z_ERROR
    # generators: IZ, ZI, ZZ
    error_2q_Z = pauli_lindblad_error_from_labels(["IZ", "ZI", "ZZ"], NOISE_STRENGTH)
    for i in 1:(NUM_QUBITS-1)
        qiskit_noise_model.add_quantum_error(error_2q_Z, inst_2q, [i-1, i])
    end
end


# Pre-build Python Noise Model
# IMPORTANT: Must pass num_qubits explicitly for long-range noise to work correctly
# The build_noise_models function doesn't take num_qubits, so we create NoiseModel directly
# Following the pattern from longrange_test.py
nm_py = mqt_noise_model.NoiseModel(processes_py, num_qubits=NUM_QUBITS)


# 3. Exact Reference
# ------------------
exact_stag_ref = nothing
ref_expvals = nothing

if MODE == "DM"
    println("Computing Exact Reference (Density Matrix)...")
    # We pass the full circuit construction to Qiskit simulator
    ref_job = mqt_simulators.run_qiskit_exact(
        NUM_QUBITS, NUM_LAYERS, init_circuit, trotter_step, qiskit_noise_model,
        method="density_matrix", observable_basis=OBSERVABLE_BASIS
    )
    # ref_job is expvals matrix (N x T) where rows are qubits, columns are time
    raw_ref_expvals = pyconvert(Matrix{Float64}, ref_job)
    
    # CRITICAL FIX: qiskit_noisy_simulator returns qubits in reverse order (evs[::-1])
    # In run_qiskit_exact: baseline[q] stores vals[q], but vals is reversed
    # So baseline[0] = qubit N-1, baseline[N-1] = qubit 0
    # After conversion to Julia (1-based): row 1 = qubit N, row N = qubit 1
    # We need to reverse the rows to get: row 1 = qubit 1, row N = qubit N
    N_rows = size(raw_ref_expvals, 1)
    reversed_indices = [N_rows - i + 1 for i in 1:N_rows]  # [N, N-1, ..., 2, 1]
    raw_ref_expvals = raw_ref_expvals[reversed_indices, :]
    
    # Prepend t=0 state
    init_vals = ones(Float64, NUM_QUBITS)
    for i in 1:NUM_QUBITS
        if (i-1) % 4 == 3
            init_vals[i] = -1.0 # X gate applied -> |1> -> -1
        end
    end
    ref_expvals = hcat(init_vals, raw_ref_expvals)

    T_steps_ref = size(ref_expvals, 2)
    exact_stag_ref = [staggered_magnetization(ref_expvals[:, t], NUM_QUBITS) for t in 1:T_steps_ref]
    println("Reference computed. Length: $T_steps_ref")
end


# 4. Runners
# ----------

function runner_julia_v2()
    # Init State
    psi = MPS(NUM_QUBITS; state="zeros")
    for i in 1:NUM_QUBITS
        if (i-1) % 4 == 3
            Yaqs.DigitalTJM.apply_single_qubit_gate!(psi, DigitalGate(XGate(), [i], nothing))
        end
    end
    
    # Observables
    obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]
    
    # Sim Config
    # Match time steps to layers for Simulator.jl pre-allocation
    sim_params = TimeEvolutionConfig(obs, Float64(NUM_LAYERS); dt=1.0, num_traj=1, max_bond_dim=64, truncation_threshold=1e-6)
    
    # Run using Simulator interface
    # This will populate obs[i].trajectories
    Simulator.run(psi, circ_jl, sim_params, noise_model_jl; parallel=false)
    
    # Extract results from trajectories
    # We ran num_traj=1, so we take the first row of trajectories
    results = zeros(ComplexF64, length(obs), length(sim_params.times))
    # Note: sim_params.times length might mismatch actual digital steps if not aligned
    # But Simulator copies min length.
    
    # We can infer actual result length from the first observable
    actual_len = 0
    # Find max non-zero index? No, DigitalTJM fills it.
    # Actually TimeEvolutionConfig creates `times` array.
    # We should trust what's in trajectories.
    
    for (i, o) in enumerate(obs)
        results[i, :] = o.trajectories[1, :]
    end
    
    # results is ComplexF64 (N_obs x N_steps)
    bond_dim = MPSModule.write_max_bond_dim(psi)
    return real.(results), bond_dim
end

function runner_py_yaqs()
    obs_yaqs = [mqt_params.Observable(mqt_gates.Z(), i) for i in 0:(NUM_QUBITS-1)]
    # StrongSimParams
    sp = mqt_params.StrongSimParams(
        observables=obs_yaqs, num_traj=1, max_bond_dim=64,
        sample_layers=true, num_mid_measurements=NUM_LAYERS
    )
    sp.dt = 1.0 # Force dt
    
    # Build circuit with init_circuit included (like run_yaqs does)
    # This ensures initial state preparation is part of the circuit
    circ_py = init_circuit.copy()
    # Use Python's range: range(num_qubits) creates range(0, num_qubits)
    # Convert Julia range to Python list for Qiskit compatibility
    qubit_list = pybuiltins.list(collect(0:(NUM_QUBITS-1)))
    circ_py.compose(trotter_step, qubits=qubit_list, inplace=true)
    circ_py.barrier(label="SAMPLE_OBSERVABLES")
    for _ in 1:(NUM_LAYERS-1)
        circ_py.compose(trotter_step, qubits=qubit_list, inplace=true)
        circ_py.barrier(label="SAMPLE_OBSERVABLES")
    end
    
    # Start from |0...0⟩ - init_circuit will prepare the state
    psi_py = mqt_networks.MPS(NUM_QUBITS, state="zeros", pad=2)
    
    mqt_sim.run(psi_py, circ_py, sp, nm_py, parallel=false)
    
    # Extract results from observables (updated in place)
    # obs_yaqs is a Julia Vector of Py objects
    first_res_py = obs_yaqs[1].results
    first_res = pyconvert(Vector{Float64}, first_res_py)
    T_steps = length(first_res)
    
    res_mat = zeros(Float64, length(obs_yaqs), T_steps)
    
    for i in 1:length(obs_yaqs)
        vals_py = obs_yaqs[i].results
        res_mat[i, :] = pyconvert(Vector{Float64}, vals_py)
    end
    
    return res_mat, 0
end

function runner_qiskit_mps()
    # Note: run_qiskit_mps builds the full circuit including init_circuit
    res_tuple = mqt_simulators.run_qiskit_mps(
        NUM_QUBITS, NUM_LAYERS, init_circuit, trotter_step, qiskit_noise_model,
        num_traj=1, observable_basis=OBSERVABLE_BASIS
    )
    expvals = pyconvert(Matrix{Float64}, res_tuple[0])
    
    # Prepend t=0 state
    # Init state: |00...0> with X on i where i%4==3
    # Z-basis expectation: |0> -> +1, |1> -> -1
    init_vals = ones(Float64, NUM_QUBITS)
    for i in 1:NUM_QUBITS
        # Python index i-1
        if (i-1) % 4 == 3
            init_vals[i] = -1.0 # X gate applied -> |1> -> -1
        end
    end
    
    # Concatenate init_vals column to expvals
    full_expvals = hcat(init_vals, expvals)
    
    return full_expvals, 0
end


# 5. Execution
# ------------
results_data = Dict()

if RUN_QISKIT_MPS
    try
        n, mse, stag, res_mat, t_total = run_trajectories(runner_qiskit_mps, exact_stag_ref, "Qiskit MPS")
        results_data["Qiskit MPS"] = (n, mse, stag, res_mat, t_total)
    catch e
        println("Qiskit MPS Failed: $e")
    end
end

if RUN_PYTHON_YAQS
    try
        n, mse, stag, res_mat, t_total = run_trajectories(runner_py_yaqs, exact_stag_ref, "Python YAQS")
        results_data["Python YAQS"] = (n, mse, stag, res_mat, t_total)
    catch e
        println("Python YAQS Failed: $e")
    end
end

if RUN_JULIA_V2
    try
        n, mse, stag, res_mat, t_total = run_trajectories(runner_julia_v2, exact_stag_ref, "Julia V2")
        results_data["Julia V2"] = (n, mse, stag, res_mat, t_total)
    catch e
        println("Julia V2 Failed: $e")
        rethrow(e)
    end
end


# 6. Plotting
# -----------
println("\nGenerating Plot...")
plt = pyimport("matplotlib.pyplot")
# Create subplots: 1 for staggered magnetization + 1 for each site in SITES_TO_PLOT
num_site_plots = length(SITES_TO_PLOT)
num_total_plots = 1 + num_site_plots
fig, axes = plt.subplots(num_total_plots, 1, figsize=(10, 4 + 3*num_site_plots), sharex=false)

# Convert axes to Julia array for easier indexing
# matplotlib returns a single Axes object if nrows=1, or an array if nrows>1
axes_array = Vector{Any}(undef, num_total_plots)
if num_total_plots == 1
    axes_array[1] = axes
else
    # axes is a Python array/list, convert to Julia array using PythonCall indexing
    for i in 1:num_total_plots
        axes_array[i] = axes[i-1]  # Python 0-based to Julia 1-based
    end
end

ax1 = axes_array[1]  # First subplot for staggered magnetization
site_axes = axes_array[2:num_total_plots]  # Remaining subplots for individual sites

x_axis = 0:NUM_LAYERS

# Subplot 1: Staggered Magnetization
if !isnothing(exact_stag_ref)
    len = length(exact_stag_ref)
    # exact_stag_ref now includes t=0, so index 0..len-1
    ax1.plot(0:(len-1), exact_stag_ref, "k-", linewidth=2, label="Exact (DM)")
    
    # Plot exact site expectation values in their own subplots
    for (idx, site) in enumerate(SITES_TO_PLOT)
        if !isnothing(ref_expvals) && site <= size(ref_expvals, 1)
            site_data = ref_expvals[site, :]
            t_indices_site = 0:(length(site_data)-1)
            ax_site = site_axes[idx]
            ax_site.plot(t_indices_site, site_data, "k-", linewidth=2, label="Exact Site $site")
        end
    end
end

colors = Dict("Qiskit MPS"=>"b", "Python YAQS"=>"g", "Julia V2"=>"r")

for (name, (n, mse, stag, res_mat, t_total)) in results_data
    c = get(colors, name, "k")
    # Check length matches
    if length(stag) != length(x_axis)
        println("Warning: $name length $(length(stag)) mismatch expected $(length(x_axis))")
        # truncate or pad?
    end
    
    # Plot Staggered Magnetization
    # Use 1:length(stag) to handle mismatches gracefully
    # Assuming stag starts at t=0 if prepended correctly
    
    # Qiskit now has t=0. Julia V2 has t=0. 
    # Plot against 0:len-1
    t_len = length(stag)
    t_indices = 0:(t_len-1)
    
    label_str = "$name (MSE=$(@sprintf("%.1e", mse)), Time=$(@sprintf("%.2f", t_total))s)"
    ax1.plot(t_indices, stag, "--o", color=c, label=label_str)
    
    # Plot each site in its own subplot
    for (idx, site) in enumerate(SITES_TO_PLOT)
        # res_mat is (N_qubits, T_steps)
        if site <= size(res_mat, 1)
            site_data = res_mat[site, :]
            t_indices_site = 0:(length(site_data)-1)
            ax_site = site_axes[idx]
            ax_site.plot(t_indices_site, site_data, "--o", color=c, markersize=4, label="$name Site $site")
        end
    end
end

ax1.set_ylabel("Staggered Magnetization")
ax1.set_title("Unraveling Benchmark (N=$NUM_QUBITS, Noise=$NOISE_STRENGTH)")
ax1.legend()
ax1.grid(true)

# Set labels and titles for each site subplot
for (idx, site) in enumerate(SITES_TO_PLOT)
    ax_site = site_axes[idx]
    ax_site.set_ylabel("Expectation Value <Z>")
    ax_site.set_title("Site $site Evolution")
    ax_site.legend()
    ax_site.grid(true)
    # Only set xlabel on the last subplot to avoid clutter
    if idx == num_site_plots
        ax_site.set_xlabel("Layer")
    end
end

results_dir = joinpath(@__DIR__, "results")
if !isdir(results_dir)
    mkpath(results_dir)
end
fname = joinpath(results_dir, "benchmark_unraveling_jl.png")
plt.savefig(fname)
println("Saved plot to $fname")

