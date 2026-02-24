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
using .Yaqs.CircuitLibrary
using .Yaqs.CircuitTJM: DigitalCircuit, add_gate!, DigitalGate
using .Yaqs.CircuitIngestion

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Select Circuit: "Ising", "Heisenberg", "XY", "QAOA", "HEA", "longrange_test"
CIRCUIT_NAME = "Ising"
periodic = false

# Simulation Size
NUM_QUBITS = 12
NUM_LAYERS = 20
TAU = 0.1
dt = TAU  # Alias for consistency with circuit construction

# Noise
NOISE_STRENGTH = 0.001
ENABLE_X_ERROR = true
ENABLE_Y_ERROR = false
ENABLE_Z_ERROR = false

# Unraveling
NUM_TRAJECTORIES = 200
MODE = "Large" # "DM" to verify against Density Matrix, "Large" for just performance

longrange_mode = "TDVP" # "TEBD" or "TDVP"
local_mode = "TDVP" # "TEBD" or "TDVP"

# Model Specific Params
# Ising
J = 1.0
g = 1.0

# Heisenberg
Jx, Jy, Jz = 1.0, 1.0, 1.0
h_field = 0.0

# XY
tau = TAU  # "dt" for XY often called tau

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

# Observables
OBSERVABLE_BASIS = "Z"
THRESHOLD_MSE = 1e-3
SITES_TO_PLOT = [1,2,3,4,5,6] # 1-based index, will be adjusted for Python/Plots

# Flags
RUN_QISKIT_MPS = true
RUN_PYTHON_YAQS = false
RUN_JULIA = true
RUN_JULIA_ANALOG_2PT = true
RUN_JULIA_ANALOG_GAUSS = true
RUN_JULIA_PROJECTOR = true

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
    
    # Warmup run for Julia to trigger JIT compilation (excluded from timing)
    if label == "Julia"
        println("  Warming up (JIT compilation)...")
        runner_single_shot()  # Discard result
    end
    
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

println("Configuration: Circuit=$CIRCUIT_NAME, N=$NUM_QUBITS, Layers=$NUM_LAYERS, Noise=$NOISE_STRENGTH")

# 1. Prepare Circuits
# -------------------
println("Building Circuits ($CIRCUIT_NAME)...")

# Julia Circuit Construction
circ_jl = DigitalCircuit(NUM_QUBITS)
# Initial Barrier
add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])

# Circuit Construction
if CIRCUIT_NAME == "Ising"
    circ_jl = create_ising_circuit(NUM_QUBITS, J, g, dt, NUM_LAYERS, periodic=periodic)
    
elseif CIRCUIT_NAME == "XY"
    for _ in 1:NUM_LAYERS
        layer = xy_trotter_layer(NUM_QUBITS, tau) 
        for g in layer.gates; add_gate!(circ_jl, g.op, g.sites); end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

elseif CIRCUIT_NAME == "Heisenberg"
    circ_jl = create_heisenberg_circuit(NUM_QUBITS, Jx, Jy, Jz, h_field, dt, NUM_LAYERS, periodic=periodic)

elseif CIRCUIT_NAME == "QAOA"
    for _ in 1:NUM_LAYERS
        layer = qaoa_ising_layer(NUM_QUBITS; beta=beta_qaoa, gamma=gamma_qaoa)
        for g in layer.gates; add_gate!(circ_jl, g.op, g.sites); end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

elseif CIRCUIT_NAME == "HEA"
    phis = fill(phi_hea, NUM_QUBITS)
    thetas = fill(theta_hea, NUM_QUBITS)
    lams = fill(lam_hea, NUM_QUBITS)
    for _ in 1:NUM_LAYERS
        layer = hea_layer(NUM_QUBITS; phis=phis, thetas=thetas, lams=lams, start_parity=start_parity_hea)
        for g in layer.gates; add_gate!(circ_jl, g.op, g.sites); end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

elseif CIRCUIT_NAME == "longrange_test"
    # Longrange test circuit: H gates on all qubits, then one RXX gate between qubits L and 1
    for _ in 1:NUM_LAYERS
        # Apply H gates to all qubits
        for q in 1:NUM_QUBITS
            add_gate!(circ_jl, HGate(), [q])
        end
        # Apply exactly ONE long-range two-qubit gate: RXX between qubits L and 1
        add_gate!(circ_jl, RzzGate(longrange_theta), [NUM_QUBITS, 1])
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end
else
    error("Unknown CIRCUIT_NAME: $CIRCUIT_NAME. Supported: Ising, XY, Heisenberg, QAOA, HEA, longrange_test")
end

# Python Circuit Construction
init_circuit = qiskit.QuantumCircuit(NUM_QUBITS)
for i in 0:(NUM_QUBITS-1)
    if i % 4 == 3
        init_circuit.x(i)
    end
end

# Construct Trotter Step for Python
trotter_step = nothing
if CIRCUIT_NAME == "Ising"
    trotter_step = mqt_circuit_lib.create_ising_circuit(NUM_QUBITS, J, g, dt, 1, periodic=periodic)
elseif CIRCUIT_NAME == "XY"
    trotter_step = mqt_circuit_lib.xy_trotter_layer(NUM_QUBITS, tau)
elseif CIRCUIT_NAME == "Heisenberg"
    trotter_step = mqt_circuit_lib.create_heisenberg_circuit(NUM_QUBITS, Jx, Jy, Jz, h_field, dt, 1, periodic=periodic)
elseif CIRCUIT_NAME == "QAOA"
    trotter_step = mqt_circuit_lib.qaoa_ising_layer(NUM_QUBITS, beta=beta_qaoa, gamma=gamma_qaoa)
elseif CIRCUIT_NAME == "HEA"
    phis_list = [phi_hea for _ in 1:NUM_QUBITS]
    thetas_list = [theta_hea for _ in 1:NUM_QUBITS]
    lams_list = [lam_hea for _ in 1:NUM_QUBITS]
    trotter_step = mqt_circuit_lib.hea_layer(NUM_QUBITS, phis=phis_list, thetas=thetas_list, lams=lams_list, start_parity=start_parity_hea)
elseif CIRCUIT_NAME == "longrange_test"
    # Import local circuit library for longrange_test
    local_src_path = normpath(joinpath(@__DIR__, "../../../src"))
    if !(local_src_path in sys.path)
        sys.path.insert(0, local_src_path)
    end
    local_circuit_lib = pyimport("Qiskit_simulator.circuit_library")
    trotter_step = local_circuit_lib.longrange_test_circuit(NUM_QUBITS, longrange_theta)
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
processes_py = pybuiltins.list() # For Python YAQS - use Python list


if ENABLE_X_ERROR
    for i in 1:NUM_QUBITS
        # Julia Dict
        d = Dict{String, Any}()
        d["name"] = "pauli_x"
        d["sites"] = [i]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        
        # Python Dict (0-based) - use pure Python types
        d_py = pybuiltins.dict()
        d_py["name"] = "pauli_x"
        d_py["sites"] = pybuiltins.list([i-1])  # Convert Julia Vector to Python List
        d_py["strength"] = NOISE_STRENGTH
        processes_py.append(d_py)
    end
    
    # Add Crosstalk XX on pairs
    for i in 1:(NUM_QUBITS-1)
        d = Dict{String, Any}()
        d["name"] = "crosstalk_xx"
        d["sites"] = [i, i+1]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        
        # Python Dict (0-based) - use pure Python types
        d_py = pybuiltins.dict()
        d_py["name"] = "crosstalk_xx"
        d_py["sites"] = pybuiltins.list([i-1, i])  # Convert Julia Vector to Python List
        d_py["strength"] = NOISE_STRENGTH
        processes_py.append(d_py)
    end
end


if ENABLE_Y_ERROR
    for i in 1:NUM_QUBITS
        # Julia Dict
        d = Dict{String, Any}()
        d["name"] = "pauli_y"
        d["sites"] = [i]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        
        # Python Dict (0-based) - use pure Python types
        d_py = pybuiltins.dict()
        d_py["name"] = "pauli_y"
        d_py["sites"] = pybuiltins.list([i-1])  # Convert Julia Vector to Python List
        d_py["strength"] = NOISE_STRENGTH
        processes_py.append(d_py)
    end
    
    # Add Crosstalk YY on pairs
    for i in 1:(NUM_QUBITS-1)
        d = Dict{String, Any}()
        d["name"] = "crosstalk_yy"
        d["sites"] = [i, i+1]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        
        # Python Dict (0-based) - use pure Python types
        d_py = pybuiltins.dict()
        d_py["name"] = "crosstalk_yy"
        d_py["sites"] = pybuiltins.list([i-1, i])  # Convert Julia Vector to Python List
        d_py["strength"] = NOISE_STRENGTH
        processes_py.append(d_py)
    end
end


if ENABLE_Z_ERROR
    for i in 1:NUM_QUBITS
        # Julia Dict
        d = Dict{String, Any}()
        d["name"] = "pauli_z"
        d["sites"] = [i]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        
        # Python Dict (0-based) - use pure Python types
        d_py = pybuiltins.dict()
        d_py["name"] = "pauli_z"
        d_py["sites"] = pybuiltins.list([i-1])  # Convert Julia Vector to Python List
        d_py["strength"] = NOISE_STRENGTH
        processes_py.append(d_py)
    end
    
    # Add Crosstalk ZZ on pairs
    for i in 1:(NUM_QUBITS-1)
        d = Dict{String, Any}()
        d["name"] = "crosstalk_zz"
        d["sites"] = [i, i+1]
        d["strength"] = NOISE_STRENGTH
        push!(processes_jl_dicts, d)
        
        # Python Dict (0-based) - use pure Python types
        d_py = pybuiltins.dict()
        d_py["name"] = "crosstalk_zz"
        d_py["sites"] = pybuiltins.list([i-1, i])  # Convert Julia Vector to Python List
        d_py["strength"] = NOISE_STRENGTH
        processes_py.append(d_py)
    end
end

noise_model_jl = NoiseModel(processes_jl_dicts, NUM_QUBITS)
nm_py = mqt_noise_model.NoiseModel(processes_py, num_qubits=NUM_QUBITS)

# Analog 2pt Noise Model
processes_analog_dicts = [copy(d) for d in processes_jl_dicts]
for d in processes_analog_dicts
    d["unraveling"] = "unitary_2pt"
    d["theta0"] = π/3 # Larger rotation to avoid saturation at dt=1.0
end
noise_model_analog = NoiseModel(processes_analog_dicts, NUM_QUBITS)

# Analog Gauss Noise Model
processes_gauss_dicts = [copy(d) for d in processes_jl_dicts]
for d in processes_gauss_dicts
    d["unraveling"] = "unitary_gauss"
    # defaults: sigma=1.0, M=11, k=4
end
noise_model_gauss = NoiseModel(processes_gauss_dicts, NUM_QUBITS; sigma=1.0)

# Projector Noise Model
processes_proj_dicts = [copy(d) for d in processes_jl_dicts]
for d in processes_proj_dicts
    d["unraveling"] = "projector"
end
noise_model_proj = NoiseModel(processes_proj_dicts, NUM_QUBITS)


# Qiskit Noise
qiskit_noise_model = aer_noise.NoiseModel()
inst_2q = pybuiltins.list([
    "cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz", "cu",
    "csx", "cs", "csdg", "swap", "iswap",
    "rxx", "ryy", "rzz", "rzx", "ecr", "dcx",
    "xx_minus_yy", "xx_plus_yy"
])

function pauli_lindblad_error_from_labels(labels::Vector{String}, rates::Vector{Float64})
    # Build a Python PauliList from label strings
    py_labels = pybuiltins.list(labels)
    pl = quantum_info.PauliList(py_labels)
    # Rates as Python list
    py_rates = pybuiltins.list(rates)
    return aer_noise.errors.PauliLindbladError(pl, py_rates)
end

if ENABLE_X_ERROR
    # generators: IX, XI, XX
    error_2q_X = pauli_lindblad_error_from_labels(["IX", "XI", "XX"], [NOISE_STRENGTH, NOISE_STRENGTH, NOISE_STRENGTH])
    for i in 1:(NUM_QUBITS-1)
        qiskit_noise_model.add_quantum_error(error_2q_X, inst_2q, [i-1, i])
    end
end

if ENABLE_Y_ERROR
    # generators: IY, YI, YY
    error_2q_Y = pauli_lindblad_error_from_labels(["IY", "YI", "YY"], [NOISE_STRENGTH, NOISE_STRENGTH, NOISE_STRENGTH])
    for i in 1:(NUM_QUBITS-1)
        qiskit_noise_model.add_quantum_error(error_2q_Y, inst_2q, [i-1, i])
    end
end

if ENABLE_Z_ERROR
    # generators: IZ, ZI, ZZ
    error_2q_Z = pauli_lindblad_error_from_labels(["IZ", "ZI", "ZZ"], [NOISE_STRENGTH, NOISE_STRENGTH, NOISE_STRENGTH])
    for i in 1:(NUM_QUBITS-1)
        qiskit_noise_model.add_quantum_error(error_2q_Z, inst_2q, [i-1, i])
    end
end





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
    # ref_job is expvals matrix (N x T)
    raw_ref_expvals = pyconvert(Matrix{Float64}, ref_job)
    
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

function runner_julia()
    # Init State
    psi = MPS(NUM_QUBITS; state="zeros")
    for i in 1:NUM_QUBITS
        if (i-1) % 4 == 3
            Yaqs.CircuitTJM.apply_single_qubit_gate!(psi, DigitalGate(XGate(), [i], nothing))
        end
    end
    
    # Observables
    obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]
    
    # Sim Config
    evolution_options = Yaqs.CircuitTJM.TJMOptions(local_method=Symbol(local_mode), long_range_method=Symbol(longrange_mode))
    
    sim_params = TimeEvolutionConfig(obs, Float64(NUM_LAYERS); dt=1.0, num_traj=1, max_bond_dim=64, truncation_threshold=1e-6)
    
    # Run using Simulator interface
    Simulator.run(psi, circ_jl, sim_params, noise_model_jl; parallel=false, alg_options=evolution_options)
    
    # Extract results
    results = zeros(ComplexF64, length(obs), length(sim_params.times))
    for (i, o) in enumerate(obs)
        results[i, :] = o.trajectories[1, :]
    end
    
    bond_dim = MPSModule.write_max_bond_dim(psi)
    return real.(results), bond_dim
end

function runner_julia_analog_2pt()
    # Init State
    psi = MPS(NUM_QUBITS; state="zeros")
    for i in 1:NUM_QUBITS
        if (i-1) % 4 == 3
            Yaqs.CircuitTJM.apply_single_qubit_gate!(psi, DigitalGate(XGate(), [i], nothing))
        end
    end
    
    # Observables
    obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]
    
    # Sim Config
    evolution_options = Yaqs.CircuitTJM.TJMOptions(local_method=Symbol(local_mode), long_range_method=Symbol(longrange_mode))
    
    sim_params = TimeEvolutionConfig(obs, Float64(NUM_LAYERS); dt=1.0, num_traj=1, max_bond_dim=64, truncation_threshold=1e-6)
    
    # Run using Simulator interface with Analog Noise Model
    Simulator.run(psi, circ_jl, sim_params, noise_model_analog; parallel=false, alg_options=evolution_options)
    
    # Extract results
    results = zeros(ComplexF64, length(obs), length(sim_params.times))
    for (i, o) in enumerate(obs)
        results[i, :] = o.trajectories[1, :]
    end
    
    bond_dim = MPSModule.write_max_bond_dim(psi)
    return real.(results), bond_dim
end

function runner_julia_analog_gauss()
    # Init State
    psi = MPS(NUM_QUBITS; state="zeros")
    for i in 1:NUM_QUBITS
        if (i-1) % 4 == 3
            Yaqs.CircuitTJM.apply_single_qubit_gate!(psi, DigitalGate(XGate(), [i], nothing))
        end
    end
    
    # Observables
    obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]
    
    # Sim Config
    evolution_options = Yaqs.CircuitTJM.TJMOptions(local_method=Symbol(local_mode), long_range_method=Symbol(longrange_mode))
    
    sim_params = TimeEvolutionConfig(obs, Float64(NUM_LAYERS); dt=1.0, num_traj=1, max_bond_dim=64, truncation_threshold=1e-6)
    
    # Run using Simulator interface with Gauss Noise Model
    Simulator.run(psi, circ_jl, sim_params, noise_model_gauss; parallel=false, alg_options=evolution_options)
    
    # Extract results
    results = zeros(ComplexF64, length(obs), length(sim_params.times))
    for (i, o) in enumerate(obs)
        results[i, :] = o.trajectories[1, :]
    end
    
    bond_dim = MPSModule.write_max_bond_dim(psi)
    return real.(results), bond_dim
end

function runner_julia_projector()
    # Init State
    psi = MPS(NUM_QUBITS; state="zeros")
    for i in 1:NUM_QUBITS
        if (i-1) % 4 == 3
            Yaqs.CircuitTJM.apply_single_qubit_gate!(psi, DigitalGate(XGate(), [i], nothing))
        end
    end
    
    # Observables
    obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]
    
    # Sim Config
    evolution_options = Yaqs.CircuitTJM.TJMOptions(local_method=Symbol(local_mode), long_range_method=Symbol(longrange_mode))
    
    sim_params = TimeEvolutionConfig(obs, Float64(NUM_LAYERS); dt=1.0, num_traj=1, max_bond_dim=64, truncation_threshold=1e-6)
    
    # Run using Simulator interface with Projector Noise Model
    Simulator.run(psi, circ_jl, sim_params, noise_model_proj; parallel=false, alg_options=evolution_options)
    
    # Extract results
    results = zeros(ComplexF64, length(obs), length(sim_params.times))
    for (i, o) in enumerate(obs)
        results[i, :] = o.trajectories[1, :]
    end
    
    bond_dim = MPSModule.write_max_bond_dim(psi)
    return real.(results), bond_dim
end

function runner_py_yaqs()
    obs_yaqs = [mqt_params.Observable(mqt_gates.Z(), i) for i in 0:(NUM_QUBITS-1)]
    # StrongSimParams
    # Set num_mid_measurements = NUM_LAYERS + 1 to include t=0 measurement
    sp = mqt_params.StrongSimParams(
        observables=obs_yaqs, num_traj=1, max_bond_dim=64,
        sample_layers=true, num_mid_measurements=NUM_LAYERS + 1
    )
    sp.dt = 1.0 # Force dt
    
    # Initialize MPS with basis string where qubits at positions i % 4 == 3 are in |1⟩ state
    # This matches Julia initialization where X gates are applied to those qubits
    basis_string = ""
    for i in 0:(NUM_QUBITS-1)
        if i % 4 == 3
            basis_string *= "1"
        else
            basis_string *= "0"
        end
    end
    # Create MPS initialized from basis string with padding
    psi_py = mqt_networks.MPS(NUM_QUBITS, state="basis", basis_string=basis_string, pad=2)
    
    # Build circuit starting from empty (no init_circuit)
    # The sample_layers=true parameter will automatically measure the correctly-initialized psi_py at t=0
    circ_py = qiskit.QuantumCircuit(NUM_QUBITS)
    # Use Python's range: range(num_qubits) creates range(0, num_qubits)
    # Convert Julia range to Python list for Qiskit compatibility
    qubit_list = pybuiltins.list(collect(0:(NUM_QUBITS-1)))
    for _ in 1:NUM_LAYERS
        circ_py.compose(trotter_step, qubits=qubit_list, inplace=true)
        circ_py.barrier(label="SAMPLE_OBSERVABLES")
    end

    
    mqt_sim.run(psi_py, circ_py, sp, nm_py, parallel=false)
    
    # Extract results from observables (updated in place)
    # The simulator automatically measures the initial state, so we expect NUM_LAYERS+1 points
    expected_steps = NUM_LAYERS + 1
    res_mat = zeros(Float64, length(obs_yaqs), expected_steps)
    
    for i in 1:length(obs_yaqs)
        vals_py = obs_yaqs[i].results
        vals = pyconvert(Vector{Float64}, vals_py)
        # Truncate to exactly NUM_LAYERS+1 points to match expected length
        res_mat[i, :] = vals[1:expected_steps]
    end
    
    # Python observables are created in index order 0..N-1, so res_mat[i, :] corresponds to qubit i-1 (0-based)
    # In Julia 1-based indexing, res_mat[1, :] = qubit 0 (first qubit), res_mat[2, :] = qubit 1 (second qubit), etc.
    # This matches Julia's indexing where obs[1] = site 1 (first qubit), so no reversal needed
    
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

if RUN_JULIA
    try
        n, mse, stag, res_mat, t_total = run_trajectories(runner_julia, exact_stag_ref, "Julia")
        results_data["Julia"] = (n, mse, stag, res_mat, t_total)
    catch e
        println("Julia Failed: $e")
        rethrow(e)
    end
end

if RUN_JULIA_ANALOG_2PT
    try
        n, mse, stag, res_mat, t_total = run_trajectories(runner_julia_analog_2pt, exact_stag_ref, "Julia Analog 2pt")
        results_data["Julia Analog 2pt"] = (n, mse, stag, res_mat, t_total)
    catch e
        println("Julia Analog 2pt Failed: $e")
        rethrow(e)
    end
end

if RUN_JULIA_ANALOG_GAUSS
    try
        n, mse, stag, res_mat, t_total = run_trajectories(runner_julia_analog_gauss, exact_stag_ref, "Julia Analog Gauss")
        results_data["Julia Analog Gauss"] = (n, mse, stag, res_mat, t_total)
    catch e
        println("Julia Analog Gauss Failed: $e")
        rethrow(e)
    end
end

if RUN_JULIA_PROJECTOR
    try
        n, mse, stag, res_mat, t_total = run_trajectories(runner_julia_projector, exact_stag_ref, "Julia Projector")
        results_data["Julia Projector"] = (n, mse, stag, res_mat, t_total)
    catch e
        println("Julia Projector Failed: $e")
        rethrow(e)
    end
end


# 6. Plotting
# -----------
println("\nGenerating Plot...")
plt = pyimport("matplotlib.pyplot")

num_plots = length(SITES_TO_PLOT)
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

# Determine minimum length across all results
all_lengths = Int[]
if !isempty(results_data)
    for (name, (n, mse, stag, res_mat, t_total)) in results_data
        if size(res_mat, 2) > 0
            push!(all_lengths, size(res_mat, 2))
        end
    end
end
if !isnothing(ref_expvals) && size(ref_expvals, 2) > 0
    push!(all_lengths, size(ref_expvals, 2))
end

min_len = isempty(all_lengths) ? 1 : minimum(all_lengths)
x = 0:(min_len-1)  # Start from 0 for layer indexing
colors = Dict("Qiskit MPS"=>"b", "Python YAQS"=>"g", "Julia"=>"r", "Julia Analog 2pt"=>"orange", "Julia Analog Gauss"=>"purple", "Julia Projector"=>"cyan")

for (i, site) in enumerate(SITES_TO_PLOT)
    ax = axes_array[i]
    
    # Plot exact reference if available
    if !isnothing(ref_expvals) && site <= size(ref_expvals, 1)
        exact = ref_expvals[site, 1:min_len]
        ax.plot(x, exact, label="Exact (DM)", color="k", linestyle="--", linewidth=1, alpha=0.6)
    end
    
    # Plot each method's results
    for (name, (n, mse, stag, res_mat, t_total)) in results_data
        c = get(colors, name, "k")
        if site <= size(res_mat, 1) && size(res_mat, 2) >= min_len
            site_data = res_mat[site, 1:min_len]
            ax.plot(x, site_data, label=name, color=c, linestyle="-", linewidth=2)
        end
    end
    
    ax.set_title("Site $site Evolution")
    ax.set_ylabel("<Z>")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(true)
    
    if i == num_plots
        ax.set_xlabel("Layer")
    end
end

fig.suptitle("$CIRCUIT_NAME (N=$NUM_QUBITS, Noise=$NOISE_STRENGTH, $NUM_LAYERS layers)")
plt.tight_layout()

results_dir = joinpath(@__DIR__, "results")
if !isdir(results_dir)
    mkpath(results_dir)
end
fname = joinpath(results_dir, "benchmark_unraveling_$(lowercase(CIRCUIT_NAME)).png")
plt.savefig(fname)
println("Plot saved to $fname")

