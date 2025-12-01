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
ENABLE_Y_ERROR = true

# Unraveling
NUM_TRAJECTORIES = 100
MODE = "DM" # "DM" to verify against Density Matrix, "Large" for just performance

# Observables
OBSERVABLE_BASIS = "Z"
THRESHOLD_MSE = 1e-3
SITES_TO_PLOT = [1, 3] # 1-based index, will be adjusted for Python/Plots

# Flags
RUN_QISKIT_MPS = true
RUN_PYTHON_YAQS = true
RUN_JULIA_V2 = true

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
    
    cumulative_results = nothing
    bond_dims = Int[]
    
    final_stag = Float64[]
    final_mse = 0.0
    
    # Measure pure simulation loop time
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
println("Building Circuits...")
trotter_step = mqt_circuit_lib.create_heisenberg_circuit(NUM_QUBITS, 1.0, 1.0, 1.0, 0.0, TAU, 1, periodic=false)

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
end

noise_model_jl = NoiseModel(processes_jl_dicts, NUM_QUBITS)

# Qiskit Noise
qiskit_noise_model = aer_noise.NoiseModel()
if ENABLE_Y_ERROR
    # Create Python list for pauli_error
    ops = [("Y", NOISE_STRENGTH), ("I", 1.0 - NOISE_STRENGTH)]
    # PythonCall should convert Vector of Tuples to list of tuples, but explicit list() is safer if strict check
    py_ops = pybuiltins.list(ops)
    error_1q = aer_noise.errors.pauli_error(py_ops)
    ops_2q = [("II", 1.0 - 3*NOISE_STRENGTH), ("IY", NOISE_STRENGTH), ("YI", NOISE_STRENGTH), ("YY", NOISE_STRENGTH)]
    error_2q = aer_noise.errors.pauli_error(pybuiltins.list(ops_2q))
    
    # Explicit python list for instructions
    inst_2q = pybuiltins.list(["rzz", "rxx", "ryy"])
    qiskit_noise_model.add_all_qubit_quantum_error(error_2q, inst_2q)
    # Also 1q gates if any
    inst_1q = pybuiltins.list(["x", "y", "z", "rx", "ry", "rz"])
    qiskit_noise_model.add_all_qubit_quantum_error(error_1q, inst_1q)
end

# Pre-build Python Noise Model
nm_py_temp = mqt_noise_utils.build_noise_models(processes_py)
if pyisinstance(nm_py_temp, pybuiltins.tuple) || pyisinstance(nm_py_temp, pybuiltins.list)
    nm_py = nm_py_temp[0]
else
    nm_py = nm_py_temp
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
    
    # Init state
    psi_py = mqt_networks.MPS(NUM_QUBITS, state="zeros", pad=2)
    for i in 0:(NUM_QUBITS-1)
        if i % 4 == 3
            # Manual gate application since apply_one_site_gate is missing
            # Access python list with 1-based index
            tensor = psi_py.tensors[i+1]
            op = mqt_gates.X().matrix
            new_tensor = oe.contract("ab, bcd->acd", op, tensor)
            psi_py.tensors[i+1] = new_tensor
        end
    end
            
    # Use pre-constructed full_yaqs_circ and nm_py
    
    mqt_sim.run(psi_py, full_yaqs_circ, sp, nm_py, parallel=false)
    
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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=true)

x_axis = 0:NUM_LAYERS

# Subplot 1: Staggered Magnetization
if !isnothing(exact_stag_ref)
    len = length(exact_stag_ref)
    # exact_stag_ref now includes t=0, so index 0..len-1
    ax1.plot(0:(len-1), exact_stag_ref, "k-", linewidth=2, label="Exact (DM)")
    
    # Also plot exact site expectation values in Subplot 2
    styles = ["--", ":", "-."]
    for (idx, site) in enumerate(SITES_TO_PLOT)
        if !isnothing(ref_expvals) && site <= size(ref_expvals, 1)
            site_data = ref_expvals[site, :]
            t_indices_site = 0:(length(site_data)-1)
            style = get(styles, idx, "-")
            ax2.plot(t_indices_site, site_data, style, color="k", label="Exact Site $site")
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
    
    # Subplot 2: Chosen Sites
    # Plot each site with different marker/style but same color family?
    # Or separate legend?
    # Let's use dashed/dotted for different sites.
    # Re-define styles to silence warning if loop scope is an issue, or just reuse
    styles_site = ["--", ":", "-."]
    
    for (idx, site) in enumerate(SITES_TO_PLOT)
        # res_mat is (N_qubits, T_steps)
        if site <= size(res_mat, 1)
            site_data = res_mat[site, :]
            t_indices_site = 0:(length(site_data)-1)
            style = get(styles_site, idx, "-")
            ax2.plot(t_indices_site, site_data, style, color=c, marker="x", markersize=4, label="$name Site $site")
        end
    end
end

ax1.set_ylabel("Staggered Magnetization")
ax1.set_title("Unraveling Benchmark (N=$NUM_QUBITS, Noise=$NOISE_STRENGTH)")
ax1.legend()
ax1.grid(true)

ax2.set_xlabel("Layer")
ax2.set_ylabel("Expectation Value <Z>")
ax2.set_title("Single Site Evolution")
ax2.legend()
ax2.grid(true)

results_dir = joinpath(@__DIR__, "results")
if !isdir(results_dir)
    mkpath(results_dir)
end
fname = joinpath(results_dir, "benchmark_unraveling_jl.png")
plt.savefig(fname)
println("Saved plot to $fname")

