module DigitalTJMV2

using LinearAlgebra
using Base.Threads
using ..MPSModule
using ..MPOModule
using ..GateLibrary
using ..NoiseModule
using ..SimulationConfigs
using ..Algorithms
using ..DigitalTJM: DigitalCircuit, DigitalGate, process_circuit

export run_digital_tjm_v2

# --- Helpers ---

"""
    construct_window_mpo(gate, window_start, window_end)

Constructs an MPO for the window [window_start, window_end].
The gate acts on `gate.sites`.
"""
function construct_window_mpo(gate::DigitalGate, window_start::Int, window_end::Int)
    L_window = window_end - window_start + 1
    tensors = Vector{Array{ComplexF64, 4}}(undef, L_window)
    phys_dims = fill(2, L_window)
    
    s1, s2 = sort(gate.sites)
    # Map global sites to window relative indices (1-based)
    rel_s1 = s1 - window_start + 1
    rel_s2 = s2 - window_start + 1
    
    gen = gate.generator
    coeff = GateLibrary.hamiltonian_coeff(gate.op)
    
    # Check if sites are within window (they MUST be)
    @assert rel_s1 >= 1 && rel_s2 <= L_window "Gate sites outside window"
    
    for i in 1:L_window
        # Identity default
        T = zeros(ComplexF64, 1, 2, 2, 1)
        T[1, :, :, 1] = Matrix{ComplexF64}(I, 2, 2)
        
        if i == rel_s1
            T[1, :, :, 1] = coeff * gen[1]
        elseif i == rel_s2
            T[1, :, :, 1] = gen[2]
        end
        
        tensors[i] = T
    end
    
    return MPO(L_window, tensors, phys_dims, 0)
end

"""
    apply_single_qubit_gate!(mps, gate)
"""
function apply_single_qubit_gate!(mps::MPS, gate::DigitalGate)
    site = gate.sites[1]
    op_mat = matrix(gate.op)
    
    # Ensure orthogonality center is at site (optional for correctness but good for numerics?)
    # Python code uses oe.contract on the tensor directly.
    # In Julia, if we modify a tensor, we should ideally respect canonical form or update it.
    # But for a local unitary, if we apply it to the orthogonality center, form is preserved.
    # If not at center, form is destroyed/needs update.
    # Python does NOT shift center for single qubit gates. It just contracts.
    # But later `canonical_form_lost = True` if using purely oe.contract? 
    # Python `apply_single_qubit_gate` just updates the tensor.
    # If the site was orthogonal, U*A is still orthogonal (unitary).
    # So we should be fine just updating.
    
    A = mps.tensors[site] # (L, d, R)
    L, d, R = size(A)
    
    # Contract: NewA[l, d', r] = Op[d', d] * A[l, d, r]
    # Permute A -> (d, L, R) -> (d, L*R)
    A_perm = reshape(permutedims(A, (2, 1, 3)), d, L*R)
    New_A_mat = op_mat * A_perm
    New_A = permutedims(reshape(New_A_mat, d, L, R), (2, 1, 3))
    
    mps.tensors[site] = New_A
end

"""
    apply_window!(state, gate, sim_params)

Applies a two-qubit gate using the windowing technique.
"""
function apply_window!(state::MPS, gate::DigitalGate, sim_params::AbstractSimConfig)
    s1, s2 = sort(gate.sites)
    
    # Window Logic: Python uses `window_size = 1` padding
    # Window = [min(s) - 1, max(s) + 1] (clipped)
    padding = 1
    win_start = max(1, s1 - padding)
    win_end = min(state.length, s2 + padding)
    
    # 1. Shift Orthogonality Center to Window Start
    # This ensures left env is Identity.
    MPSModule.shift_orthogonality_center!(state, win_start)
    
    # 2. Extract Window
    # We use views/copies. For MPS construction we need arrays.
    # We copy the tensors to avoid mutating the original state's tensors directly during TDVP
    # until we are done (or TDVP mutates them in place).
    # Since we put them back later, we can just take the subset.
    # Note: `two_site_tdvp!` modifies the MPS in place.
    # If we pass a new MPS struct pointing to the SAME tensor arrays, it modifies them.
    # This is fine, effectively "In-Place" on the window.
    # BUT: `two_site_tdvp!` expects a full chain.
    # We need a `short_state` that thinks it has length `win_len`.
    
    win_len = win_end - win_start + 1
    
    # We must be careful: if we share the arrays, `two_site_tdvp!` might resize them (svd truncation).
    # If `two_site_tdvp!` replaces `tensors[i]` with a new array (it does), 
    # then the original `state.tensors` won't point to the new one unless we update it.
    # So we can create a temporary MPS, run TDVP, then copy the tensors back.
    
    win_tensors = state.tensors[win_start:win_end] # Shallow copy of the vector of arrays
    win_phys = state.phys_dims[win_start:win_end]
    
    # Create Short MPS
    # Orth center is at 1 (relative) because we shifted global to win_start.
    short_state = MPS(win_len, win_tensors, win_phys, 1)
    
    # 3. Construct Short MPO
    short_mpo = construct_window_mpo(gate, win_start, win_end)
    
    # 4. Run TDVP
    # Use temporary config with dt=1.0 (gate application)
    # Python uses dt=1.0 for generator gates.
    gate_config = TimeEvolutionConfig(Observable[], 1.0; dt=1.0, 
                                      truncation_threshold=sim_params.truncation_threshold,
                                      max_bond_dim=sim_params.max_bond_dim)
    
    two_site_tdvp!(short_state, short_mpo, gate_config)
    
    # 5. Update Original State
    # The `short_state.tensors` have been updated (replaced with new arrays due to SVD/truncation).
    # We must put them back into `state`.
    state.tensors[win_start:win_end] .= short_state.tensors
    
    # 6. Update Orthogonality Center
    # `two_site_tdvp!` finishes with a backward sweep, so center ends at 1 (relative).
    # So global center is `win_start`.
    state.orth_center = win_start
end


# --- Main Runner ---

function run_digital_tjm_v2(initial_state::MPS, circuit::DigitalCircuit, 
                            noise_model::Union{NoiseModel, Nothing}, 
                            sim_params::AbstractSimConfig)
    
    # 1. Deepcopy Initial State
    state = deepcopy(initial_state)
    
    # 2. Process Circuit into Layers (Reuse V1 logic)
    layers, barrier_map = process_circuit(circuit)
    num_layers = length(layers)
    
    # 3. Setup Results
    num_obs = length(sim_params.observables)
    
    # Identify sample points
    sample_indices = Int[]
    
    # Check layer 0
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
    
    # Default to final measurement if no samples requested
    if isempty(sample_indices)
        num_steps = 1 # Final
    else
        num_steps = length(sample_indices)
    end
    
    results = zeros(ComplexF64, num_obs, num_steps)
    current_meas_idx = 1
    
    # Helper for measurement
    function measure!(idx)
        for (i, obs) in enumerate(sim_params.observables)
            results[i, idx] = SimulationConfigs.expect(state, obs)
        end
    end
    
    # Initial Measurement
    if 0 in sample_indices
        measure!(current_meas_idx)
        current_meas_idx += 1
    end
    
    # Loop Layers
    for (l_idx, layer) in enumerate(layers)
        
        # 1. Single Qubit Gates
        for gate in layer
            if length(gate.sites) == 1
                apply_single_qubit_gate!(state, gate)
            end
        end
        
        # 2. Two Qubit Gates (Windowed)
        # We iterate all 2-qubit gates in this layer.
        # Since they are in the same layer, they are disjoint. Order doesn't matter.
        for gate in layer
            if length(gate.sites) == 2
                apply_window!(state, gate, sim_params)
                
                # TODO: Noise Application would go here (locally)
            end
        end
        
        # Normalize (if noise free, TDVP preserves norm, but good practice)
        # MPSModule.normalize!(state) 
        
        # Measurement
        if l_idx in sample_indices
            measure!(current_meas_idx)
            current_meas_idx += 1
        end
    end
    
    # Final Measurement (if no barriers)
    if isempty(sample_indices) && num_obs > 0
        measure!(1)
    end
    
    return state, results
end

end # module

