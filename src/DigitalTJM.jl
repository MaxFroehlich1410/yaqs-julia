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

export DigitalGate, DigitalCircuit, add_gate!, process_circuit, run_digital_tjm

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

function DigitalCircuit(n::Int)
    return DigitalCircuit(n, DigitalGate[], Vector{Vector{DigitalGate}}())
end

function add_gate!(circ::DigitalCircuit, op::AbstractOperator, sites::Vector{Int}; generator=nothing)
    push!(circ.gates, DigitalGate(op, sites, generator))
end

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
    op_mat = matrix(gate.op)
    A = mps.tensors[site]
    L, d, R = size(A)
    A_perm = reshape(permutedims(A, (2, 1, 3)), d, L*R)
    New_A_mat = op_mat * A_perm
    New_A = permutedims(reshape(New_A_mat, d, L, R), (2, 1, 3))
    mps.tensors[site] = New_A
end

function apply_window!(state::MPS, gate::DigitalGate, sim_params::AbstractSimConfig)
    s1, s2 = sort(gate.sites)
    padding = 1
    win_start = max(1, s1 - padding)
    win_end = min(state.length, s2 + padding)
    MPSModule.shift_orthogonality_center!(state, win_start)
    win_len = win_end - win_start + 1
    win_tensors = state.tensors[win_start:win_end]
    win_phys = state.phys_dims[win_start:win_end]
    short_state = MPS(win_len, win_tensors, win_phys, 1)
    short_mpo = construct_window_mpo(gate, win_start, win_end)
    gate_config = TimeEvolutionConfig(Observable[], 1.0; dt=1.0, 
                                      truncation_threshold=sim_params.truncation_threshold,
                                      max_bond_dim=sim_params.max_bond_dim)
    two_site_tdvp!(short_state, short_mpo, gate_config)
    state.tensors[win_start:win_end] .= short_state.tensors
    state.orth_center = win_start
end

# --- Main Runner ---

function run_digital_tjm(initial_state::MPS, circuit::DigitalCircuit, 
                            noise_model::Union{NoiseModel, Nothing}, 
                            sim_params::AbstractSimConfig)
    
    state = deepcopy(initial_state)
    layers, barrier_map = process_circuit(circuit)
    num_layers = length(layers)
    num_obs = length(sim_params.observables)
    sample_indices = Int[]
    
    if haskey(barrier_map, 0)
        for label in barrier_map[0]
            if uppercase(label) == "SAMPLE_OBSERVABLES"
                push!(sample_indices, 0)
                break 
            end
        end
    end
    for l in 1:num_layers
        if haskey(barrier_map, l)
            for label in barrier_map[l]
                if uppercase(label) == "SAMPLE_OBSERVABLES"
                    push!(sample_indices, l)
                    break
                end
            end
        end
    end
    if isempty(sample_indices)
        num_steps = 1
    else
        num_steps = length(sample_indices)
    end
    
    results = zeros(ComplexF64, num_obs, num_steps)
    current_meas_idx = 1
    
    function measure!(idx)
        for (i, obs) in enumerate(sim_params.observables)
            results[i, idx] = SimulationConfigs.expect(state, obs)
        end
    end
    
    if 0 in sample_indices
        measure!(current_meas_idx)
        current_meas_idx += 1
    end
    
    for (l_idx, layer) in enumerate(layers)
        for gate in layer
            if length(gate.sites) == 1
                apply_single_qubit_gate!(state, gate)
            end
        end
        for gate in layer
            if length(gate.sites) == 2
                apply_window!(state, gate, sim_params)
                if !isnothing(noise_model) && !isempty(noise_model.processes)
                    s1, s2 = sort(gate.sites)
                    local_noise = create_local_noise_model(noise_model, s1, s2)
                    if !isempty(local_noise.processes)
                        apply_dissipation(state, local_noise, 1.0, sim_params)
                        stochastic_process!(state, local_noise, 1.0, sim_params)
                    else
                         MPSModule.normalize!(state) 
                    end
                else
                     MPSModule.normalize!(state) 
                end
            end
        end
        if l_idx in sample_indices
            measure!(current_meas_idx)
            current_meas_idx += 1
        end
    end
    
    if isempty(sample_indices) && num_obs > 0
        measure!(1)
    end
    
    return state, results
end

end # module
