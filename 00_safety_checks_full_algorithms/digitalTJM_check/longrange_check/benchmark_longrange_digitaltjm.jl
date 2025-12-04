using LinearAlgebra
using PythonCall
using Printf
using Statistics
using Dates

# ==============================================================================
# 1. SETUP & INCLUDES
# ==============================================================================

# Adjust this to point to your Yaqs.jl src folder
include("../../../src/Yaqs.jl")
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.GateLibrary
using .Yaqs.SimulationConfigs
using .Yaqs.DigitalTJM
using .Yaqs.Simulator
using .Yaqs.CircuitLibrary # Now using the library you provided
using .Yaqs.NoiseModule

# ==============================================================================
# 2. PYTHON PATH CONFIGURATION
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

# Import MQT YAQS and the Python Circuit Library
try
    global mqt_circuit_lib = pyimport("mqt.yaqs.core.libraries.circuit_library")
    global mqt_simulators = pyimport("mqt.yaqs.codex_experiments.worker_functions.qiskit_simulators")
    global mqt_sim = pyimport("mqt.yaqs.simulator")
    global mqt_networks = pyimport("mqt.yaqs.core.data_structures.networks")
    global mqt_params = pyimport("mqt.yaqs.core.data_structures.simulation_parameters")
    global mqt_gates = pyimport("mqt.yaqs.core.libraries.gate_library")
    global mqt_noise_model = pyimport("mqt.yaqs.core.data_structures.noise_model")
    global qiskit = pyimport("qiskit")
    global aer_noise = pyimport("qiskit_aer.noise")
    global quantum_info = pyimport("qiskit.quantum_info")
catch e
    println("Error importing Python modules: ", e)
    println("Sys path: ", sys.path)
    rethrow(e)
end

# ==============================================================================
# 3. CONFIGURATION & CIRCUIT SELECTION
# ==============================================================================

# Choose Circuit: 
# "ising", "heisenberg", "ising_2d", "heisenberg_2d", 
# "fermi_hubbard_1d", "fermi_hubbard_2d", "longrange_test", "qaoa", "hea"
CIRCUIT_NAME = "ising"

# System Params
L_x = 3
L_y = 2
NUM_QUBITS = 5 # Total qubits
TIMESTEPS  = 10
DT         = 0.1
NUM_TRAJECTORIES = 500


MODE = "DM" # "DM" to verify against Density Matrix, "Large" for just performance
OBSERVABLE_BASIS = "Z"

SITES_TO_PLOT = [1,2,3,4,5] # 1-based index, will be adjusted for Python/Plots

RUN_QISKIT_MPS = true
RUN_PYTHON_YAQS = true
RUN_JULIA = true
RUN_QISKIT_EXACT = true

longrange_mode = "TDVP" # "TEBD" or "TDVP"
local_mode = "TDVP" # "TEBD" or "TDVP"


ENABLE_X_ERROR = true
ENABLE_Y_ERROR = false
ENABLE_Z_ERROR = false
NOISE_STRENGTH = 0.01


# ------------------------------------------------------------------------------
# DISPATCHER: Maps CIRCUIT_NAME to Julia and Python Function Calls
# ------------------------------------------------------------------------------

println(">>> Building Circuits for: $CIRCUIT_NAME")

circ_jl = nothing
trotter_step_py = nothing

if CIRCUIT_NAME == "ising"
    J, g = 1.0, 0.5
    circ_jl = create_ising_circuit(NUM_QUBITS, J, g, DT, TIMESTEPS, periodic=false)
    trotter_step_py = mqt_circuit_lib.create_ising_circuit(NUM_QUBITS, J, g, DT, 1, periodic=false)

elseif CIRCUIT_NAME == "heisenberg"
    Jx, Jy, Jz, h = 1.0, 1.0, 1.0, 0.5
    circ_jl = create_heisenberg_circuit(NUM_QUBITS, Jx, Jy, Jz, h, DT, TIMESTEPS, periodic=false)
    trotter_step_py = mqt_circuit_lib.create_heisenberg_circuit(NUM_QUBITS, Jx, Jy, Jz, h, DT, 1, periodic=false)

elseif CIRCUIT_NAME == "ising_2d"
    J, g = 1.0, 0.5
    circ_jl = create_2d_ising_circuit(L_y, L_x, J, g, DT, TIMESTEPS) # Note: Rows, Cols
    trotter_step_py = mqt_circuit_lib.create_2d_ising_circuit(L_y, L_x, J, g, DT, 1)

elseif CIRCUIT_NAME == "heisenberg_2d"
    Jx, Jy, Jz, h = 1.0, 1.0, 1.0, 0.5
    circ_jl = create_2d_heisenberg_circuit(L_y, L_x, Jx, Jy, Jz, h, DT, TIMESTEPS)
    trotter_step_py = mqt_circuit_lib.create_2d_heisenberg_circuit(L_y, L_x, Jx, Jy, Jz, h, DT, 1)

elseif CIRCUIT_NAME == "fermi_hubbard_1d"
    u, t, mu = 2.0, 1.0, 0.0
    # Note: L is number of spatial sites. Total qubits = 2*L
    # Julia function expects L (spatial)
    L_spatial = NUM_QUBITS ÷ 2 
    circ_jl = create_1d_fermi_hubbard_circuit(L_spatial, u, t, mu, 1, DT, TIMESTEPS)
    trotter_step_py = mqt_circuit_lib.create_1d_fermi_hubbard_circuit(L_spatial, u, t, mu, 1, DT, 1)

elseif CIRCUIT_NAME == "fermi_hubbard_2d"
    u, t, mu = 2.0, 1.0, 0.0
    # L_x * L_y must equal spatial sites
    spatial_sites = NUM_QUBITS ÷ 2
    # Assuming L_x, L_y set correctly above for the geometry
    circ_jl = create_2d_fermi_hubbard_circuit(L_x, L_y, u, t, mu, 1, DT, TIMESTEPS)
    trotter_step_py = mqt_circuit_lib.create_2d_fermi_hubbard_circuit(L_x, L_y, u, t, mu, 1, DT, 1)

elseif CIRCUIT_NAME == "longrange_test"
    theta = π/4
    circ_jl = longrange_test_circuit(NUM_QUBITS, theta)
    trotter_step_py = mqt_circuit_lib.longrange_test_circuit(NUM_QUBITS, theta)

elseif CIRCUIT_NAME == "qaoa"
    beta, gamma = 0.3, 0.5
circ_jl = DigitalCircuit(NUM_QUBITS)
    # Manual loop for QAOA as library provides 'layer'
    layer_jl = qaoa_ising_layer(NUM_QUBITS, beta=beta, gamma=gamma)
    trotter_step_py = mqt_circuit_lib.qaoa_ising_layer(NUM_QUBITS, beta=beta, gamma=gamma)
    
    # Compose for timesteps
    add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    for _ in 1:TIMESTEPS
        # Copy gates from layer to main
        for g in layer_jl.gates
            add_gate!(circ_jl, g.op, g.sites)
        end
        add_gate!(circ_jl, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end

else
    error("Unknown circuit: $CIRCUIT_NAME")
end

println("✓ Circuits Constructed")


# ==============================================================================
# 4. NOISE SETUP
# ==============================================================================

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


nm_py = mqt_noise_model.NoiseModel(processes_py, num_qubits=NUM_QUBITS)

# ==============================================================================
# 5. INITIALIZATION & EXACT REFERENCE
# ==============================================================================

# Create Qiskit Initial Circuit (All Zeros)
init_circuit_py = qiskit.QuantumCircuit(NUM_QUBITS)

# Exact Reference (Density Matrix)
exact_results = nothing
if RUN_QISKIT_EXACT
    println("\n>>> Running Qiskit Exact (Density Matrix)...")
    # This returns expvals matrix (N_qubits x N_steps)
    # Note: run_qiskit_exact typically returns just the evolution steps.
    # We assume it handles the initial state internally or we treat the output as t=1..T
    t_start = time()
    res_py = mqt_simulators.run_qiskit_exact(
        NUM_QUBITS, TIMESTEPS, init_circuit_py, trotter_step_py, qiskit_noise_model,
        method="density_matrix", observable_basis=OBSERVABLE_BASIS
    )
    println("Qiskit Exact Finished in $(round(time() - t_start, digits=3))s")
    
    # Convert to Julia Matrix
    # Shape: (N_qubits, TIMESTEPS)
    raw_vals = pyconvert(Matrix{Float64}, res_py)
    
    # Prepend t=0 (All zeros state -> Z=1 for all qubits)
    # If starting state is different, adjust 'init_vals'
    init_vals = ones(Float64, NUM_QUBITS)
    exact_results = hcat(init_vals, raw_vals)
end

# ==============================================================================
# 6. RUNNERS
# ==============================================================================

# --- 6a. Julia Runner ---
function run_julia_sim()
    println("\n>>> Running Julia (DigitalTJM)...")
    # Initial State: |0...0>
    psi = MPS(NUM_QUBITS; state="zeros")
    
    # Observables
    obs = [Observable("Z_$i", ZGate(), i) for i in 1:NUM_QUBITS]
    
    # Configuration
    # We use 'sample_timesteps=true' to get data at every step
    sim_params = TimeEvolutionConfig(
        obs, Float64(TIMESTEPS); 
        dt=1.0, 
        num_traj=NUM_TRAJECTORIES, 
        max_bond_dim=64,
        sample_timesteps=true
    )
    
    # Alg Options
    options = Yaqs.DigitalTJM.TJMOptions(
        local_method=Symbol(local_mode), 
        long_range_method=Symbol(longrange_mode)
    )

    t_start = time()
    # Run Simulation
    # Returns: (final_psi, results_matrix)
    # results_matrix dims: [num_obs, num_steps + 1] (includes t=0)
    _, res_matrix = run_digital_tjm(
        psi, circ_jl, noise_model_jl, sim_params; 
        alg_options=options
    )
    
    println("Julia Finished in $(round(time() - t_start, digits=3))s")
    return res_matrix
end

# --- 6b. Python YAQS Runner ---
function run_py_yaqs_sim()
    println("\n>>> Running Python YAQS...")
    
    # Construct Full Circuit (Init + Repeated Trotter Steps)
    full_circ = init_circuit_py.copy()
    full_circ.barrier(label="SAMPLE_OBSERVABLES") # t=0
    for _ in 1:TIMESTEPS
        full_circ.compose(trotter_step_py, inplace=true)
        full_circ.barrier(label="SAMPLE_OBSERVABLES")
    end
    
    # Observables
    obs_py = [mqt_params.Observable(mqt_gates.Z(), i) for i in 0:(NUM_QUBITS-1)]
    
    # Simulation Parameters
    sp = mqt_params.StrongSimParams(
        observables=obs_py, 
        num_traj=NUM_TRAJECTORIES, 
        max_bond_dim=64,
        sample_layers=true, 
        num_mid_measurements=TIMESTEPS
    )
    sp.dt = 1.0
    
    # Initial State
    psi_py = mqt_networks.MPS(NUM_QUBITS, state="zeros", pad=2)
    
    t_start = time()
    mqt_sim.run(psi_py, full_circ, sp, nm_py, parallel=false)
    println("PyYAQS Finished in $(round(time() - t_start, digits=3))s")
    
    # Extract Results
    # Python results structure: [metadata..., val_t0, val_t1...]
    # Typically metadata is length 2 or 3 depending on version. 
    # We assume standard YAQS output where data starts at index 3 (0-based 2)
    # Adjust slicing [3:end] based on your specific version if needed.
    
    # Initialize matrix: (N_qubits, TIMESTEPS + 1)
    # We need to determine the length of the time series from the first observable
    sample_res = pyconvert(Vector{Float64}, obs_py[1].results)
    # Heuristic check: if len > TIMESTEPS+5, it likely has metadata
    # We expect 1 (t=0) + TIMESTEPS data points.
    
    # Let's assume the data is at the end.
    data_len = TIMESTEPS + 1
    res_matrix = zeros(Float64, NUM_QUBITS, data_len)
    
    for i in 1:NUM_QUBITS
        raw = pyconvert(Vector{Float64}, obs_py[i].results)
        # Take the last 'data_len' elements
        if length(raw) >= data_len
            res_matrix[i, :] = raw[end-data_len+1:end]
        else
            println("Warning: YAQS result length mismatch for qubit $(i-1)")
        end
    end
    return res_matrix
end

# --- 6c. Qiskit MPS Runner ---
function run_qiskit_mps_sim()
    println("\n>>> Running Qiskit MPS...")
    
    t_start = time()
    # run_qiskit_mps returns a tuple, first element is mean expvals
    res_tuple = mqt_simulators.run_qiskit_mps(
        NUM_QUBITS, TIMESTEPS, init_circuit_py, trotter_step_py, qiskit_noise_model,
        num_traj=NUM_TRAJECTORIES, observable_basis=OBSERVABLE_BASIS
    )
    println("Qiskit MPS Finished in $(round(time() - t_start, digits=3))s")
    
    # res_tuple[0] is the mean expvals matrix (N_qubits x TIMESTEPS)
    raw_vals = pyconvert(Matrix{Float64}, res_tuple[0])
    
    # Prepend t=0 (All zeros state -> Z=1)
    init_vals = ones(Float64, NUM_QUBITS)
    full_vals = hcat(init_vals, raw_vals)
    return full_vals
end

# ==============================================================================
# 7. EXECUTION
# ==============================================================================

res_julia = RUN_JULIA ? run_julia_sim() : nothing
res_yaqs  = RUN_PYTHON_YAQS ? run_py_yaqs_sim() : nothing
res_mps   = RUN_QISKIT_MPS ? run_qiskit_mps_sim() : nothing

# ==============================================================================
# 8. PLOTTING & SAVING
# ==============================================================================

println("\nGenerating Plots...")
plt = pyimport("matplotlib.pyplot")

# Create results directory if it doesn't exist
results_dir = joinpath(@__DIR__, "results")
if !isdir(results_dir)
    mkpath(results_dir)
end

# Setup Subplots: One per site in SITES_TO_PLOT
num_plots = length(SITES_TO_PLOT)
fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3*num_plots), sharex=true)

# Handle single vs multiple axes
axes_list = num_plots == 1 ? [axes] : [axes[i-1] for i in 1:num_plots]

x_axis = 0:TIMESTEPS

    for (idx, site) in enumerate(SITES_TO_PLOT)
    ax = axes_list[idx]
    
    # 1. Exact Reference (Black Dashed)
    if !isnothing(exact_results)
        # Check bounds
        if site <= size(exact_results, 1)
            y = exact_results[site, :]
            len = min(length(x_axis), length(y))
            ax.plot(x_axis[1:len], y[1:len], label="Exact (DM)", color="black", linestyle="--", linewidth=1.5, zorder=10)
    end
end

    # 2. Julia (Red Solid)
    if !isnothing(res_julia)
        if site <= size(res_julia, 1)
            y = res_julia[site, :]
            len = min(length(x_axis), length(y))
            ax.plot(x_axis[1:len], y[1:len], label="Julia", color="red", linewidth=2.0, alpha=0.8)
        end
    end
    
    # 3. YAQS (Green Dotted)
    if !isnothing(res_yaqs)
        if site <= size(res_yaqs, 1)
            y = res_yaqs[site, :]
            len = min(length(x_axis), length(y))
            ax.plot(x_axis[1:len], y[1:len], label="PyYAQS", color="green", linestyle=":", linewidth=2.5)
        end
    end
    
    # 4. Qiskit MPS (Blue Dash-Dot)
    if !isnothing(res_mps)
        if site <= size(res_mps, 1)
            y = res_mps[site, :]
            len = min(length(x_axis), length(y))
            ax.plot(x_axis[1:len], y[1:len], label="Qiskit MPS", color="blue", linestyle="-.", linewidth=1.5)
    end
end

    ax.set_ylabel("<Z>")
    ax.set_title("Site $site Evolution")
    ax.grid(true, alpha=0.3)
    
    # Only show legend on the first plot to avoid clutter
    if idx == 1
        ax.legend(loc="upper right", fontsize="small", ncol=2)
    end
end

axes_list[end].set_xlabel("Layer (Step)")

# Add a super title with simulation details
fig.suptitle("Benchmark: $CIRCUIT_NAME (N=$NUM_QUBITS, Noise=$NOISE_STRENGTH, Traj=$NUM_TRAJECTORIES)")
plt.tight_layout()

# Construct filename
timestamp = Dates.format(now(), "MMdd_HHmm")
fname = joinpath(results_dir, "benchmark_noise_$(CIRCUIT_NAME)_$(timestamp).png")
plt.savefig(fname)
println("✓ Plot saved to: $fname")