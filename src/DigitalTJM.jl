module DigitalTJMV2

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
using ..DigitalTJM: DigitalCircuit, DigitalGate, process_circuit

export run_digital_tjm_v2

# --- Noise Helpers ---

"""
    create_local_noise_model(global_noise_model, sites...)

Creates a new NoiseModel containing only processes that act on the specified sites.
"""
function create_local_noise_model(noise_model::NoiseModel{T}, site1::Int, site2::Int) where T
    affected_sites = Set([site1, site2])
    
    local_procs = Vector{AbstractNoiseProcess{T}}()
    
    for proc in noise_model.processes
        # Check if process sites are subset of affected sites
        p_sites = proc.sites
        if length(p_sites) == 1
            if p_sites[1] == site1 || p_sites[1] == site2
                push!(local_procs, proc)
            end
        elseif length(p_sites) == 2
            # Check exact match (ignoring order if needed, but usually sorted)
            if (p_sites[1] == site1 && p_sites[2] == site2) || (p_sites[1] == site2 && p_sites[2] == site1)
                 push!(local_procs, proc)
            end
        end
    end
    
    return NoiseModel(local_procs)
end

"""
    solve_local_jumps!(mps, noise_model, dt)

Performs stochastic jumps based on the local noise model.
Optimized for local application without full H_eff construction.
"""
function solve_local_jumps!(mps::MPS{T}, noise_model::NoiseModel{T}, dt::Float64) where T
    # 1. Calculate Norm Squared
    # Since we are in a trajectory, the state might be unnormalized due to dissipation.
    norm_sq = real(MPSModule.scalar_product(mps, mps))
    
    # 2. Random Draw
    r = rand()
    
    if r > norm_sq
        # A jump occurs!
        rates = Float64[]
        jump_candidates = [] # Store (proc, op)
        
        for proc in noise_model.processes
            gamma = proc.strength
            
            val = 0.0
            if proc isa LocalNoiseProcess
                op = proc.matrix
                op_sq = op' * op
                
                if length(proc.sites) == 1
                    val = real(MPSModule.local_expect(mps, op_sq, proc.sites[1]))
                elseif length(proc.sites) == 2
                     val = real(MPSModule.local_expect_two_site(mps, op_sq, proc.sites[1], proc.sites[2]))
                end
                
                push!(rates, max(0.0, val * gamma * dt))
                push!(jump_candidates, (proc, op))
                
            elseif proc isa MPONoiseProcess
                L_mpo = proc.mpo
                
                # Calculate rate <psi| L_dag L |psi>
                # Efficiently: contract L|psi>, norm sq.
                # Or expect_mpo(L_dag L).
                # Let's use L|psi> contraction as it's simpler if we don't have L_dag*L precomputed.
                # L_mpo is MPO. mps is MPS.
                
                # Expectation:
                # We can use MPOModule.expect_mpo(L_dag * L, mps).
                # Or create temp_mps = L * mps.
                # temp_mps = contract_mpo_mps(L_mpo, mps) -- this creates uncompressed MPS (bond dim grows).
                # norm(temp_mps)^2 is the rate.
                # This is O(N * D^2 * d^2 * k^2).
                
                # Let's use contract_mpo_mps then norm.
                # Note: contract_mpo_mps might return MPS with large bond dimension.
                # We don't need to compress it just to get norm.
                # However, `contract_mpo_mps` in MPO.jl typically returns an MPS.
                
                # Wait, we have `expect_mpo(O, mps)`.
                # If we compute O = L_dag * L once?
                # But L is specific to this process.
                # Let's compute L_dag L on the fly or assume it's cheap enough.
                # Usually L is simple MPO (bond dim 2 or 4).
                # contract_mpo_mpo might be better.
                
                # Let's use `contract_mpo_mps` -> `norm`.
                
                # We need to make sure we don't modify `mps` here.
                # `contract_mpo_mps` creates new MPS.
                
                # Optimization: We only need norm. We don't need full MPS?
                # But let's stick to existing tools.
                
                # To avoid import cycle or overhead, we use MPOModule functions.
                # We assume MPOModule is available.
                
                temp_mps = MPOModule.contract_mpo_mps(L_mpo, mps)
                val = real(MPSModule.scalar_product(temp_mps, temp_mps))
                
                push!(rates, max(0.0, val * gamma * dt))
                push!(jump_candidates, (proc, L_mpo))
            end
        end
        
        total_rate = sum(rates)
        
        if total_rate > 0
            # Select jump
            r_jump = rand() * total_rate
            current_sum = 0.0
            chosen_idx = 0
            
            for (i, rate) in enumerate(rates)
                current_sum += rate
                if r_jump <= current_sum
                    chosen_idx = i
                    break
                end
            end
            
            if chosen_idx > 0
                (proc, op) = jump_candidates[chosen_idx]
                
                if proc isa LocalNoiseProcess
                    # ... (existing local logic)
                    if length(proc.sites) == 1
                       # ...
                       site = proc.sites[1]
                       T_ten = mps.tensors[site]
                       @tensor T_new[l, p, r] := op[p, k] * T_ten[l, k, r]
                       mps.tensors[site] = T_new
                   elseif length(proc.sites) == 2
                       # ... (existing 2-site logic)
                       site1, site2 = proc.sites
                       A1 = mps.tensors[site1]
                       A2 = mps.tensors[site2]
                       @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
                       
                       op_tensor = reshape(op, 2, 2, 2, 2)
                       @tensor Theta_prime[l, p1n, p2n, r] := op_tensor[p1n, p2n, p1, p2] * Theta[l, p1, p2, r]
                       
                       d1, d2 = size(Theta_prime, 2), size(Theta_prime, 3)
                       L_dim, R_dim = size(Theta_prime, 1), size(Theta_prime, 4)
                       
                       Mat = reshape(Theta_prime, L_dim * 2, 2 * R_dim)
                       F = svd(Mat)
                       rank = length(F.S)
                       mps.tensors[site1] = reshape(F.U, L_dim, 2, rank)
                       mps.tensors[site2] = reshape(Diagonal(F.S) * F.Vt, rank, 2, R_dim)
                   end
                elseif proc isa MPONoiseProcess
                    # Apply MPO Jump
                    # op is the MPO
                    new_mps = MPOModule.contract_mpo_mps(op, mps)
                    # Truncate to keep bond dimension reasonable
                    # Jumps increase bond dimension.
                    # We should truncate to max_bond_dim (from where? usually mps.max_bond_dim if stored, or sim_params)
                    # But solve_local_jumps! signature doesn't have sim_params or max_bond_dim.
                    # We'll rely on default truncation or just compression.
                    # MPSModule.truncate!(new_mps)
                    # Better: check if mps has bond dim info? No.
                    # Just compress.
                    MPSModule.truncate!(new_mps; threshold=1e-10)
                    
                    # Update mps tensors in-place (replace arrays)
                    # mps is mutable struct.
                    mps.tensors = new_mps.tensors
                    mps.phys_dims = new_mps.phys_dims
                    mps.orth_center = new_mps.orth_center
                    mps.length = new_mps.length
                end
            end
        end
    end
    
    # 3. Normalize
    # Digital simulation usually enforces normalization after steps
    MPSModule.normalize!(mps)
end

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
    win_len = win_end - win_start + 1
    
    win_tensors = state.tensors[win_start:win_end] # Shallow copy of the vector of arrays
    win_phys = state.phys_dims[win_start:win_end]
    
    # Create Short MPS
    # Orth center is at 1 (relative) because we shifted global to win_start.
    short_state = MPS(win_len, win_tensors, win_phys, 1)
    
    # 3. Construct Short MPO
    short_mpo = construct_window_mpo(gate, win_start, win_end)
    
    # 4. Run TDVP
    # Use temporary config with dt=1.0 (gate application)
    gate_config = TimeEvolutionConfig(Observable[], 1.0; dt=1.0, 
                                      truncation_threshold=sim_params.truncation_threshold,
                                      max_bond_dim=sim_params.max_bond_dim)
    
    two_site_tdvp!(short_state, short_mpo, gate_config)
    
    # 5. Update Original State
    state.tensors[win_start:win_end] .= short_state.tensors
    
    # 6. Update Orthogonality Center
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
        for gate in layer
            if length(gate.sites) == 2
                apply_window!(state, gate, sim_params)
                
                # Apply Noise
                if !isnothing(noise_model) && !isempty(noise_model.processes)
                    s1, s2 = sort(gate.sites)
                    local_noise = create_local_noise_model(noise_model, s1, s2)
                    
                    if !isempty(local_noise.processes)
                        # 1. Dissipation (dt=1.0)
                        apply_dissipation(state, local_noise, 1.0, sim_params)
                        
                        # 2. Stochastic Jump (dt=1.0)
                        solve_local_jumps!(state, local_noise, 1.0)
                    else
                         # If no noise, normalize to correct any drift
                         MPSModule.normalize!(state) 
                    end
                else
                     MPSModule.normalize!(state) 
                end
            end
        end
        
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
