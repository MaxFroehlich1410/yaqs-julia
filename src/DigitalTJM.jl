module DigitalTJM

using LinearAlgebra
using Base.Threads
using TensorOperations
using ..MPSModule
using ..MPOModule
using ..GateLibrary
using ..NoiseModule
using ..DissipationModule
using ..SimulationConfigs
using ..Algorithms
using ..StochasticProcessModule
import ..Timing
using ..Timing: @t

export DigitalGate, DigitalCircuit, RepeatedDigitalCircuit, add_gate!, process_circuit, run_digital_tjm, TJMOptions,
       enable_timing!, set_timing_print_each_call!, reset_timing!, print_timing_summary!

"""
    enable_timing!(flag::Bool=true)

Enable timing collection for DigitalTJM execution (also enables deep timings in
TDVP / dissipation / stochastic submodules).
"""
enable_timing!(flag::Bool=true) = Timing.enable_timing!(flag)

set_timing_print_each_call!(flag::Bool=true) = Timing.set_timing_print_each_call!(flag)
reset_timing!() = Timing.reset_timing!()
print_timing_summary!(; header::AbstractString="DigitalTJM timing summary", top::Int=20) =
    Timing.print_timing_summary!(; header=header, top=top)

# --- Data Structures ---

struct DigitalGate
    op::AbstractOperator
    sites::Vector{Int}
    generator::Union{Vector{<:AbstractMatrix{ComplexF64}}, Nothing}
end

mutable struct DigitalCircuit
    num_qubits::Int
    gates::Vector{DigitalGate}
    layers::Vector{Vector{DigitalGate}} # Optional: processed layers
end

"""
    RepeatedDigitalCircuit(step::DigitalCircuit, repeats::Int)

Represents a circuit obtained by repeating the same `step` circuit `repeats` times.

This avoids materializing `repeats` copies of an identical gate list, which is
important for very large systems (e.g. IBM127-style circuits).
"""
struct RepeatedDigitalCircuit
    step::DigitalCircuit
    repeats::Int
end

function DigitalCircuit(n::Int)
    return DigitalCircuit(n, DigitalGate[], Vector{Vector{DigitalGate}}())
end

function add_gate!(circ::DigitalCircuit, op::AbstractOperator, sites::Vector{Int}; generator=nothing)
    push!(circ.gates, DigitalGate(op, sites, generator))
end

# --- Options ---

struct TJMOptions
    local_method::Symbol # :TEBD or :TDVP
    long_range_method::Symbol # :TEBD or :TDVP
end

# Default constructor
TJMOptions(;local_method=:TDVP, long_range_method=:TDVP) = TJMOptions(local_method, long_range_method)

# --- Circuit Processing ---

"""
    process_circuit(circuit::DigitalCircuit)

Process the circuit gates into layers of commuting operations.
Returns `(layers, barrier_map)`.
"""
function process_circuit(circuit::DigitalCircuit)
    # Simple greedy layering
    layers = Vector{Vector{DigitalGate}}()
    barrier_map = Dict{Int, Vector{String}}()
    
    current_layer = DigitalGate[]
    busy_qubits = Set{Int}()
    
    layer_idx = 1
    
    for gate in circuit.gates
        # Check for Barrier
        if gate.op isa GateLibrary.Barrier
            # If barrier, finish current layer
            if !isempty(current_layer)
                push!(layers, current_layer)
                current_layer = DigitalGate[]
                busy_qubits = Set{Int}()
                layer_idx += 1
            end
            
            # Record barrier
            idx = !isempty(layers) ? length(layers) : 0
            if !haskey(barrier_map, idx)
                barrier_map[idx] = String[]
            end
            push!(barrier_map[idx], gate.op.label)
            continue
        end
        
        # Check commutativity / qubit overlap
        overlap = false
        for q in gate.sites
            if q in busy_qubits
                overlap = true
                break
            end
        end
        
        if overlap
            # Push current layer
            push!(layers, current_layer)
            current_layer = DigitalGate[]
            busy_qubits = Set{Int}()
            layer_idx += 1
        end
        
        # Add to current
        push!(current_layer, gate)
        for q in gate.sites
            push!(busy_qubits, q)
        end
    end
    
    if !isempty(current_layer)
        push!(layers, current_layer)
    end
    
    return layers, barrier_map
end

@inline function _has_sample_barrier(barrier_map::Dict{Int, Vector{String}}, idx::Int)
    if !haskey(barrier_map, idx)
        return false
    end
    for label in barrier_map[idx]
        if uppercase(label) == "SAMPLE_OBSERVABLES"
            return true
        end
    end
    return false
end

@inline function _sample_plan(barrier_map::Dict{Int, Vector{String}}, num_layers::Int)
    sample_at_start = _has_sample_barrier(barrier_map, 0)
    sample_after = Int[]
    for l in 1:num_layers
        if _has_sample_barrier(barrier_map, l)
            push!(sample_after, l)
        end
    end
    return sample_at_start, sample_after
end

# --- Noise Helpers ---

function create_local_noise_model(noise_model::NoiseModel{T}, site1::Int, site2::Int) where T
    affected_sites = Set([site1, site2])
    local_procs = Vector{AbstractNoiseProcess{T}}()
    for proc in noise_model.processes
        p_sites = proc.sites
        if length(p_sites) == 1
            if p_sites[1] == site1 || p_sites[1] == site2
                push!(local_procs, proc)
            end
        elseif length(p_sites) == 2
            if (p_sites[1] == site1 && p_sites[2] == site2) || (p_sites[1] == site2 && p_sites[2] == site1)
                 push!(local_procs, proc)
            end
        end
    end
    return NoiseModel(local_procs)
end

# --- Helpers ---

function construct_window_mpo(gate::DigitalGate, window_start::Int, window_end::Int)
    L_window = window_end - window_start + 1
    tensors = Vector{Array{ComplexF64, 4}}(undef, L_window)
    phys_dims = fill(2, L_window)
    
    s1, s2 = sort(gate.sites)
    rel_s1 = s1 - window_start + 1
    rel_s2 = s2 - window_start + 1
    
    # Get generator from gate, or from operator if not set
    gen = gate.generator
    if isnothing(gen)
        gen = GateLibrary.generator(gate.op)
    end
    coeff = GateLibrary.hamiltonian_coeff(gate.op)
    
    @assert rel_s1 >= 1 && rel_s2 <= L_window "Gate sites outside window"

    # Determine which generator element goes to which site
    # gate.sites[1] corresponds to gen[1]
    # gate.sites[2] corresponds to gen[2]
    # s1 is min(sites), s2 is max(sites)
    
    g_s1 = gen[1]
    g_s2 = gen[2]
    
    if gate.sites[1] != s1
        # If sites were sorted/swapped (e.g. [2, 1] -> s1=1, s2=2), 
        # then s1 corresponds to sites[2] -> gen[2]
        g_s1 = gen[2]
        g_s2 = gen[1]
    end
    
    for i in 1:L_window
        T = zeros(ComplexF64, 1, 2, 2, 1)
        T[1, :, :, 1] = Matrix{ComplexF64}(I, 2, 2)
        if i == rel_s1
            T[1, :, :, 1] = coeff * g_s1
        elseif i == rel_s2
            T[1, :, :, 1] = g_s2
        end
        tensors[i] = T
    end
    return MPO(L_window, tensors, phys_dims, 0)
end

function apply_single_qubit_gate!(mps::MPS, gate::DigitalGate)
    site = gate.sites[1]
    @t :matrix_1q begin
        op_mat = matrix(gate.op)
        A = mps.tensors[site]
        L, d, R = size(A)
        @t :permutedims_1q A_perm = reshape(permutedims(A, (2, 1, 3)), d, L*R)
        @t :mul_1q New_A_mat = op_mat * A_perm
        @t :permutedims_1q New_A = permutedims(reshape(New_A_mat, d, L, R), (2, 1, 3))
        mps.tensors[site] = New_A
    end
end

function apply_local_gate_exact!(mps::MPS, op::AbstractOperator, s1::Int, s2::Int, config::AbstractSimConfig)
    # Standard TEBD update for nearest neighbor gate
    # Moves orthogonality center to s1 (assumes mixed/right canonical to right of s1)
    @t :shift_orth_center MPSModule.shift_orthogonality_center!(mps, s1)
    
    A1 = mps.tensors[s1]
    A2 = mps.tensors[s2]
    
    # Contract A1 * A2
    # A1: (l1, p1, b)
    # A2: (b, p2, r2)
    @t :contract_theta @tensor Theta[l1, p1, p2, r2] := A1[l1, p1, k] * A2[k, p2, r2]
    
    # Contract with Gate
    # op_mat: 4x4 (acting on p1, p2)
    # Reshape op_mat to (p1_out, p2_out, p1_in, p2_in)
    @t :matrix_2q op_mat = matrix(op)
    op_tensor = reshape(op_mat, 2, 2, 2, 2)
    
    @t :contract_gate @tensor Theta_prime[l1, p1_out, p2_out, r2] := op_tensor[p1_out, p2_out, p1_in, p2_in] * Theta[l1, p1_in, p2_in, r2]
    
    # SVD and Truncate
    l1_dim, p1_dim, p2_dim, r2_dim = size(Theta_prime)
    @t :reshape_theta Theta_matrix = reshape(Theta_prime, l1_dim * p1_dim, p2_dim * r2_dim)
    
    @t :svd F = svd(Theta_matrix)
    
    # Truncation Logic
    threshold = config.truncation_threshold
    max_bond = config.max_bond_dim
    
    # Truncate based on threshold
    current_sum = 0.0
    keep_count = length(F.S)
    @t :truncation_loop for k in length(F.S):-1:1
        if current_sum + F.S[k]^2 > threshold
            break
        end
        current_sum += F.S[k]^2
        keep_count -= 1
    end
    keep = clamp(keep_count, 1, max_bond)
    
    U = F.U[:, 1:keep]
    S = F.S[1:keep]
    Vt = F.Vt[1:keep, :]
    
    # Update MPS
    # A1 becomes U (Left Canonical)
    @t :update_mps mps.tensors[s1] = reshape(U, l1_dim, p1_dim, keep)
    
    # A2 becomes S * V (Right Canonical? No, Center)
    @t :update_mps mps.tensors[s2] = reshape(Diagonal(S) * Vt, keep, p2_dim, r2_dim)
    mps.orth_center = s2
end

function apply_window!(state::MPS, gate::DigitalGate, sim_params::AbstractSimConfig, alg_options::TJMOptions)
    s1, s2 = sort(gate.sites)
    is_long_range = (s2 > s1 + 1)
    
    method = is_long_range ? alg_options.long_range_method : alg_options.local_method
    
    if method == :TEBD
        if is_long_range
            # Swap Network + Local TEBD
            
            # Swap Path: (s2-1, s2), (s2-2, s2-1), ..., (s1+1, s1+2)
            # s2 moves LEFT to s1+1
            for k in (s2-1):-1:(s1+1)
                @t :apply_swap apply_local_gate_exact!(state, SWAPGate(), k, k+1, sim_params)
            end
            
            # Apply gate on (s1, s1+1)
            @t :apply_local_gate_exact apply_local_gate_exact!(state, gate.op, s1, s1+1, sim_params)
            
            # Unwind Swaps: (s1+1, s1+2), ..., (s2-1, s2)
            # s2 moves RIGHT back to s2
            for k in (s1+1):(s2-1)
                @t :apply_swap apply_local_gate_exact!(state, SWAPGate(), k, k+1, sim_params)
            end
        else
            # Nearest Neighbor: Direct
            @t :apply_local_gate_exact apply_local_gate_exact!(state, gate.op, s1, s2, sim_params)
        end
    else # :TDVP
        padding = 0
        win_start = max(1, s1 - padding)
        win_end = min(state.length, s2 + padding)
        @t :shift_orth_center MPSModule.shift_orthogonality_center!(state, win_start)
        win_len = win_end - win_start + 1
        @t :window_slice begin
            win_tensors = state.tensors[win_start:win_end]
            win_phys = state.phys_dims[win_start:win_end]
            short_state = MPS(win_len, win_tensors, win_phys, 1)
        end
        @t :construct_window_mpo short_mpo = construct_window_mpo(gate, win_start, win_end)
        
        # Use StrongMeasurementConfig to trigger the efficient Directed Circuit Sweep
        # instead of the Symmetric Hamiltonian Sweep triggered by TimeEvolutionConfig.
        gate_config = StrongMeasurementConfig(Observable[]; 
                                            max_bond_dim=sim_params.max_bond_dim, 
                                            truncation_threshold=sim_params.truncation_threshold)
        
        @t :two_site_tdvp two_site_tdvp!(short_state, short_mpo, gate_config)
    
        @t :window_writeback state.tensors[win_start:win_end] .= short_state.tensors
        state.orth_center = win_end
    end
end

# --- Main Runner ---

function run_digital_tjm(initial_state::MPS, circuit::DigitalCircuit, 
                            noise_model::Union{NoiseModel, Nothing}, 
                            sim_params::AbstractSimConfig;
                            alg_options::TJMOptions = TJMOptions(local_method=:TDVP, long_range_method=:TDVP))

    local ts = Timing.begin_scope!()
    try
        state = @t :deepcopy_state deepcopy(initial_state)
        layers, barrier_map = @t :process_circuit process_circuit(circuit)
        num_layers = length(layers)
        num_obs = length(sim_params.observables)

        sample_at_start, sample_after = _sample_plan(barrier_map, num_layers)
        num_steps = (sample_at_start ? 1 : 0) + length(sample_after)
        if num_steps == 0
            num_steps = 1
        end
    
    results = zeros(ComplexF64, num_obs, num_steps)
    bond_dims = zeros(Int, num_steps)
    current_meas_idx = 1
    
    function measure!(idx)
        @t :measure begin
            for (i, obs) in enumerate(sim_params.observables)
                results[i, idx] = @t :expect SimulationConfigs.expect(state, obs)
            end
            bond_dims[idx] = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
        end
    end
    
    if sample_at_start
        measure!(current_meas_idx)
        current_meas_idx += 1
    end
    
    for (l_idx, layer) in enumerate(layers)
        for gate in layer
            if length(gate.sites) == 1
                @t :apply_single_qubit_gate apply_single_qubit_gate!(state, gate)
            end
        end
        for gate in layer
            if length(gate.sites) == 2
                @t :apply_window apply_window!(state, gate, sim_params, alg_options)
                if !isnothing(noise_model) && !isempty(noise_model.processes)
                    s1, s2 = sort(gate.sites)
                    local_noise = @t :create_local_noise_model create_local_noise_model(noise_model, s1, s2)
                    if !isempty(local_noise.processes)
                        @t :apply_dissipation apply_dissipation(state, local_noise, 1.0, sim_params)
                        @t :stochastic_process stochastic_process!(state, local_noise, 1.0, sim_params)
                    else
                         @t :normalize MPSModule.normalize!(state)
                    end
                else
                     @t :normalize MPSModule.normalize!(state)
                end
            end
        end

        # Progress logging (overwrite line with \r)
        current_bond = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
        print("\r\tLayer $l_idx/$num_layers | Max Bond: $current_bond")
        flush(stdout)

        if l_idx in sample_after
            measure!(current_meas_idx)
            current_meas_idx += 1
        end
    end
    print("\n") # Newline after finishing all layers of this trajectory
    
    # If the circuit did not request sampling (no SAMPLE_OBSERVABLES barriers),
    # still produce at least one measurement for compatibility.
    if !sample_at_start && isempty(sample_after) && num_obs > 0
        measure!(1)
    end
        # If the circuit did not request sampling (no SAMPLE_OBSERVABLES barriers),
        # still produce at least one measurement for compatibility.
        if !sample_at_start && isempty(sample_after) && num_obs > 0
            measure!(1)
        end
    
    return state, results, bond_dims
    finally
        Timing.end_scope!(ts; header="DigitalTJM per-trajectory timing")
    end
end

function run_digital_tjm(initial_state::MPS, circuit::RepeatedDigitalCircuit,
                         noise_model::Union{NoiseModel, Nothing},
                         sim_params::AbstractSimConfig;
                         alg_options::TJMOptions = TJMOptions(local_method=:TDVP, long_range_method=:TDVP))
    local ts = Timing.begin_scope!()
    try
        state = @t :deepcopy_state deepcopy(initial_state)
        layers, barrier_map = @t :process_circuit process_circuit(circuit.step)
        step_layers = length(layers)
        repeats = circuit.repeats

        num_obs = length(sim_params.observables)
        sample_at_start, sample_after = _sample_plan(barrier_map, step_layers)

        num_steps = (sample_at_start ? 1 : 0) + repeats * length(sample_after)
        if num_steps == 0
            num_steps = 1
        end

        results = zeros(ComplexF64, num_obs, num_steps)
        bond_dims = zeros(Int, num_steps)
        current_meas_idx = 1

        function measure!(idx)
            @t :measure begin
                for (i, obs) in enumerate(sim_params.observables)
                    results[i, idx] = @t :expect SimulationConfigs.expect(state, obs)
                end
                bond_dims[idx] = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
            end
        end

        if sample_at_start
            measure!(current_meas_idx)
            current_meas_idx += 1
        end

        total_layers = repeats * step_layers
        for rep in 1:repeats
            for (l_idx, layer) in enumerate(layers)
                for gate in layer
                    if length(gate.sites) == 1
                        @t :apply_single_qubit_gate apply_single_qubit_gate!(state, gate)
                    end
                end
                for gate in layer
                    if length(gate.sites) == 2
                        @t :apply_window apply_window!(state, gate, sim_params, alg_options)
                        if !isnothing(noise_model) && !isempty(noise_model.processes)
                            s1, s2 = sort(gate.sites)
                            local_noise = @t :create_local_noise_model create_local_noise_model(noise_model, s1, s2)
                            if !isempty(local_noise.processes)
                                @t :apply_dissipation apply_dissipation(state, local_noise, 1.0, sim_params)
                                @t :stochastic_process stochastic_process!(state, local_noise, 1.0, sim_params)
                            else
                                @t :normalize MPSModule.normalize!(state)
                            end
                        else
                            @t :normalize MPSModule.normalize!(state)
                        end
                    end
                end

                global_layer = (rep - 1) * step_layers + l_idx
                current_bond = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
                print("\r\tLayer $global_layer/$total_layers | Max Bond: $current_bond")
                flush(stdout)

                if l_idx in sample_after
                    measure!(current_meas_idx)
                    current_meas_idx += 1
                end
            end
        end
        print("\n")

        if num_steps == 1 && num_obs > 0 && !sample_at_start && isempty(sample_after)
            measure!(1)
        end

        return state, results, bond_dims
    finally
        Timing.end_scope!(ts; header="DigitalTJM per-trajectory timing")
    end
end

end # module
