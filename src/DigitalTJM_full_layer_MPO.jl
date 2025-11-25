module DigitalTJM

using LinearAlgebra
using Base.Threads
using ..MPSModule
using ..MPOModule
using ..GateLibrary
using ..NoiseModule
using ..SimulationConfigs
using ..Algorithms
using ..DissipationModule
using ..StochasticProcessModule

export DigitalCircuit, add_gate!, run_digital_tjm, DigitalGate

# --- Circuit Representation ---

struct DigitalGate
    op::AbstractOperator
    sites::Vector{Int}
    # Optional: Custom generator for 2-qubit gates (Hamiltonian term)
    # If nothing, we assume the GateLibrary defines it or it's a standard unitary.
    generator::Union{Vector{Matrix{ComplexF64}}, Nothing} 
end

mutable struct DigitalCircuit
    num_qubits::Int
    layers::Vector{Vector{DigitalGate}} # Organized by "time steps"
    gates::Vector{DigitalGate}
    
    function DigitalCircuit(n::Int)
        new(n, Vector{Vector{DigitalGate}}(), Vector{DigitalGate}())
    end
end

function add_gate!(circ::DigitalCircuit, op::AbstractOperator, sites::Vector{Int}; generator=nothing)
    if isnothing(generator)
        try
            generator = GateLibrary.generator(op)
        catch e
            # If generator not defined, and it's a 2-qubit gate, this might be an issue later if using TJM.
            # But for 1-qubit gates, generator is not needed by DigitalTJM (uses contract).
            # We leave it as nothing.
        end
    end
    push!(circ.gates, DigitalGate(op, sites, generator))
end

# --- Layer Processing (Commutation/Parallelism) ---

"""
    process_circuit(circuit::DigitalCircuit)

Organizes gates into parallelizable layers.
Returns a tuple: (layers, barriers_indices)
`layers` is a Vector of Vectors of gates.
`barriers_indices` is a Dict mapping layer_index -> Vector{String} of barrier labels that occur AFTER that layer.
"""
function process_circuit(circ::DigitalCircuit)
    # If existing layers are populated, we might need to re-process to find barriers if not already handled.
    # But DigitalCircuit struct doesn't store barriers separately in layers usually.
    # We will re-process from flat list `circ.gates` to ensure correct barrier handling.
    
    last_layer_idx = zeros(Int, circ.num_qubits)
    layers = Vector{Vector{DigitalGate}}()
    
    # Map from layer index to list of barrier labels to execute AFTER that layer
    barrier_map = Dict{Int, Vector{String}}()
    
    for gate in circ.gates
        op = gate.op
        
        if isa(op, Barrier)
            # Barriers act as synchronization points.
            # They effectively "close" the current layers for the involved qubits.
            # And we attach the barrier label to the *current max layer* of these qubits.
            
            # Find the max layer among involved qubits
            # If sites is empty (global barrier), use max of all.
            sites = isempty(gate.sites) ? collect(1:circ.num_qubits) : gate.sites
            
            max_l = 0
            for s in sites
                max_l = max(max_l, last_layer_idx[s])
            end
            
            # If max_l is 0, it means barrier at start. We can mark it at 0.
            if !haskey(barrier_map, max_l)
                barrier_map[max_l] = String[]
            end
            push!(barrier_map[max_l], op.label)
            
            # All involved qubits must now be at least at max_l (effectively they wait).
            # But they are already at <= max_l.
            # We don't increment last_layer_idx for a barrier, just sync?
            # Actually, future gates on these qubits must start AFTER this barrier.
            # So effectively, they must start at max_l + 1.
            # But wait, if we have gates at max_l already, next gate goes to max_l+1 naturally?
            # Yes, but a barrier forces ALL wires to be at least max_l before proceeding?
            # No, usually barriers just segment.
            # For simplicity: We attach the label to `max_l` so we measure after `max_l` is done.
            # And we update `last_layer_idx` to `max_l` for all sites to ensure synchronization?
            
            for s in sites
                last_layer_idx[s] = max_l
            end
            
            continue
        end
        
        # Normal Gate
        start_layer = 1
        for s in gate.sites
            start_layer = max(start_layer, last_layer_idx[s] + 1)
        end
        
        while length(layers) < start_layer
            push!(layers, Vector{DigitalGate}())
        end
        
        push!(layers[start_layer], gate)
        
        for s in gate.sites
            last_layer_idx[s] = start_layer
        end
    end
    
    return layers, barrier_map
end

# --- MPO Construction ---

"""
    construct_generator_mpo(gate, L)

Constructs an MPO for the generator of a 2-qubit gate.
Assumes the generator is a single term H = A ⊗ B (bond dim 1).
Applies the scaling coefficient from the gate op (e.g. theta/2).
"""
function construct_generator_mpo(gate::DigitalGate, L::Int)
    @assert length(gate.sites) == 2
    @assert !isnothing(gate.generator) "Gate must have a generator defined for Digital TJM"
    
    s1, s2 = sort(gate.sites)
    gen = gate.generator
    
    # Get scaling coefficient
    coeff = GateLibrary.hamiltonian_coeff(gate.op)
    
    tensors = Vector{Array{ComplexF64, 4}}(undef, L)
    phys_dims = fill(2, L)
    
    for i in 1:L
        T = zeros(ComplexF64, 1, 2, 2, 1)
        
        if i == s1
            T[1, :, :, 1] = coeff * gen[1] # Scale first op
        elseif i == s2
            T[1, :, :, 1] = gen[2] # Second op (coeff applied once)
        else
            T[1, :, :, 1] = Matrix{ComplexF64}(I, 2, 2)
        end
        tensors[i] = T
    end
    
    return MPO(L, tensors, phys_dims)
end

# --- Application ---

function apply_single_qubit_gate!(mps::MPS, gate::DigitalGate)
    site = gate.sites[1]
    op_mat = matrix(gate.op)
    
    A = mps.tensors[site] # (L, d, R)
    L, d, R = size(A)
    A_perm = reshape(permutedims(A, (2, 1, 3)), d, L*R)
    
    New_A = op_mat * A_perm
    New_A = permutedims(reshape(New_A, d, L, R), (2, 1, 3))
    
    mps.tensors[site] = New_A
end

function apply_two_qubit_gate!(mps::MPS, gate::DigitalGate, sim_params::AbstractSimConfig)
    # sim_params can be TimeEvolutionConfig or StrongMeasurementConfig.
    # two_site_tdvp! expects TimeEvolutionConfig for parameters like dt and order.
    # If we have StrongMeasurementConfig, we might need to extract/adapt.
    # However, the Python code assumes TJM (TDVP) runs with certain integration parameters.
    # In Julia, StrongMeasurementConfig doesn't carry dt/order/krylov params.
    # The user likely passed TimeEvolutionConfig for digital simulation anyway.
    # Let's assume sim_params is compatible or convert.
    
    H_gate = construct_generator_mpo(gate, mps.length)
    
    # Create a temporary config for the step
    # Default dt=1.0 is standard for "Gate as Generator" if H is properly scaled.
    tmp_config = TimeEvolutionConfig(Observable[], 1.0; dt=1.0) 
    two_site_tdvp!(mps, H_gate, tmp_config)
end

# --- Helper: Observable Evaluation ---

function evaluate_observables!(state::MPS, sim_params::StrongMeasurementConfig, results::Matrix{ComplexF64}, col_idx::Int)
    for (obs_idx, obs) in enumerate(sim_params.observables)
        val = SimulationConfigs.expect(state, obs)
        results[obs_idx, col_idx] = val
    end
end

function evaluate_observables!(state::MPS, sim_params::TimeEvolutionConfig, results::Matrix{ComplexF64}, col_idx::Int)
    for (obs_idx, obs) in enumerate(sim_params.observables)
        val = SimulationConfigs.expect(state, obs)
        results[obs_idx, col_idx] = val
    end
end

# --- Main Loop ---

function run_digital_tjm(initial_state::MPS, circuit::DigitalCircuit, 
                         noise_model::Union{NoiseModel, Nothing}, 
                         sim_params::AbstractSimConfig)
    
    state = deepcopy(initial_state)
    layers, barrier_map = process_circuit(circuit)
    num_layers = length(layers)
    
    # Initialize Results Storage
    # If sim_params is StrongMeasurementConfig or TimeEvolutionConfig with observables
    num_obs = length(sim_params.observables)
    
    # Pre-calculate how many "SAMPLE_OBSERVABLES" barriers exist
    # If none, we might default to just final, or standard behavior.
    # But user request says "measure ONLY when Barrier labeled SAMPLE_OBSERVABLES".
    
    sample_indices = Int[]
    
    # Check layer 0 (before start)
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
    
    num_steps = length(sample_indices)
    
    # If no explicit barriers found, maybe default to final?
    # Or should we strictly follow "ONLY when barrier"?
    # Let's assume if no barriers, we output just final state (1 step).
    if num_steps == 0
        num_steps = 1
        # We will measure at the very end
    end
    
    results = zeros(ComplexF64, num_obs, num_steps)
    
    # Measurement Counter
    current_meas_idx = 1
    
    # Handle Layer 0 (Initial)
    if 0 in sample_indices
        evaluate_observables!(state, sim_params, results, current_meas_idx)
        current_meas_idx += 1
    end
    
    # Loop Layers
    for (layer_idx, layer) in enumerate(layers)
        single_q = [g for g in layer if length(g.sites) == 1]
        two_q = [g for g in layer if length(g.sites) == 2]
        
        # 1. Single Qubit Gates
        for gate in single_q
            apply_single_qubit_gate!(state, gate)
        end
        
        # 2. Two Qubit Gates (Even/Odd)
        even_gates = [g for g in two_q if min(g.sites...) % 2 != 0] # 1-based: 1, 3, 5 are "odd" starts
        odd_gates = [g for g in two_q if min(g.sites...) % 2 == 0]  # 1-based: 2, 4, 6...
        
        if !isempty(even_gates)
            H_even = construct_combined_mpo(even_gates, state.length)
            # Apply unitary
            tmp_config = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
            two_site_tdvp!(state, H_even, tmp_config)
            apply_noise_if_needed!(state, noise_model, even_gates, sim_params)
        end
        
        if !isempty(odd_gates)
            H_odd = construct_combined_mpo(odd_gates, state.length)
            tmp_config = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
            two_site_tdvp!(state, H_odd, tmp_config)
            apply_noise_if_needed!(state, noise_model, odd_gates, sim_params)
        end
        
        if isnothing(noise_model)
             MPSModule.normalize!(state)
        end
        
        # Check if we should sample after this layer
        if layer_idx in sample_indices
            evaluate_observables!(state, sim_params, results, current_meas_idx)
            current_meas_idx += 1
        end
    end
    
    # If no sample barriers were found at all, measure final
    if isempty(sample_indices) && num_obs > 0
        evaluate_observables!(state, sim_params, results, 1)
    end
    
    return state, results
end

function construct_combined_mpo(gates::Vector{DigitalGate}, L::Int)
    if isempty(gates)
        return MPO(L; identity=true)
    end
    
    H_total = MPO(L) 
    
    for gate in gates
        H_gate = construct_generator_mpo(gate, L)
        H_total = H_total + H_gate
    end
    
    MPOModule.truncate!(H_total; threshold=1e-12)
    return H_total
end

function apply_noise_if_needed!(state::MPS, noise_model::Union{NoiseModel, Nothing}, gates::Vector{DigitalGate}, sim_params)
    if isnothing(noise_model) return end
    
    dt = 1.0
    if isa(sim_params, TimeEvolutionConfig)
        dt = sim_params.dt # Usually 1.0 for gates
    end
    
    for gate in gates
        s1, s2 = sort(gate.sites)
        
        relevant_procs = Vector{AbstractNoiseProcess{ComplexF64}}()
        for p in noise_model.processes
            if issubset(p.sites, [s1, s2])
                push!(relevant_procs, p)
            end
        end
        
        if !isempty(relevant_procs)
            local_nm = NoiseModel(relevant_procs)
            sp = StochasticProcess(MPO(state.length), local_nm) 
            H_dissip = sp.H_eff
            
            # Dissipative evolution
            # Always use a local config for the step
            tmp = TimeEvolutionConfig(Observable[], 1.0; dt=dt)
            two_site_tdvp!(state, H_dissip, tmp)
            
            solve_jumps!(state, sp, dt) 
        end
    end
end

end # module
