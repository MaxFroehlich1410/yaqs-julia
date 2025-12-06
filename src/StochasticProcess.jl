module StochasticProcessModule

using LinearAlgebra
using Random
using TensorOperations
using ..MPSModule
using ..MPOModule
using ..NoiseModule
using ..SimulationConfigs
using ..Algorithms

export StochasticProcess, calculate_stochastic_factor, create_probability_distribution, stochastic_process!

"""
    StochasticProcess{T}

A lightweight wrapper around `NoiseModel` to maintain API compatibility.
"""
struct StochasticProcess{T}
    noise_model::NoiseModel{T}
end

# Constructor for backward compatibility (ignores H argument)
function StochasticProcess(hamiltonian::MPO{T}, noise_model::NoiseModel{T}) where T
    return StochasticProcess(noise_model)
end

"""
    calculate_stochastic_factor(state::MPS)

Calculate the stochastic factor dp = 1 - <psi|psi>.
This assumes the state norm has decayed due to dissipation.
"""
function calculate_stochastic_factor(state::MPS)
    norm_sq = real(MPSModule.scalar_product(state, state))
    return 1.0 - norm_sq
end

"""
    apply_mpo_jump!(state::MPS, mpo::MPO, sim_params)

Apply an MPO jump operator to the MPS in-place.
Contracts site-by-site (growing bond dimensions) and then truncates.
"""
function apply_mpo_jump!(state::MPS, mpo::MPO{T}, sim_params) where T
    @assert state.length == mpo.length
    
    # Contract site-by-site
    for i in 1:state.length
        A = state.tensors[i] # (La, Pi, Ra)
        W = mpo.tensors[i]   # (Lw, Po, Pi, Rw)
        
        # Contract over physical index Pi
        # A: [la, pi, ra]
        # W: [lw, po, pi, rw]
        # Result C: [la, lw, po, ra, rw]
        @tensor C[la, lw, po, ra, rw] := A[la, pi, ra] * W[lw, po, pi, rw]
        
        # Reshape to merge virtual bonds: (La*Lw, Po, Ra*Rw)
        la, lw, po, ra, rw = size(C)
        state.tensors[i] = reshape(C, la * lw, po, ra * rw)
    end
    
    # Truncate to control bond dimension growth
    # Use threshold from sim_params if available
    threshold = 1e-12
    max_bond = nothing
    if hasproperty(sim_params, :truncation_threshold)
        threshold = sim_params.truncation_threshold
    elseif hasproperty(sim_params, :threshold)
        threshold = sim_params.threshold
    end
    
    if hasproperty(sim_params, :bond_dim)
        max_bond = sim_params.bond_dim
    elseif hasproperty(sim_params, :max_bond_dim)
        max_bond = sim_params.max_bond_dim
    end
    
    MPSModule.truncate!(state; threshold=threshold, max_bond_dim=max_bond)
end

"""
    create_probability_distribution(state, noise_model, dt, sim_params)

Generate probabilities for all possible jumps.
"""
function create_probability_distribution(state::MPS, noise_model::NoiseModel, dt::Float64, sim_params::AbstractSimConfig)
    if isempty(noise_model.processes)
        return Float64[], Any[]
    end

    dp_m_list = Float64[]
    jump_candidates = Any[] # Stores (proc, op, type, sites)

    L = state.length

    # Iterate over sites to minimize orthogonality center shifts
    for site in 1:L
        # Shift OC to site (efficient access for local norms)
        if site > 1
             MPSModule.shift_orthogonality_center!(state, site)
        else
             MPSModule.shift_orthogonality_center!(state, 1)
        end

        # --- 1-site jumps at this site ---
        for proc in noise_model.processes
            if isa(proc, LocalNoiseProcess) && length(proc.sites) == 1 && proc.sites[1] == site
                gamma = proc.strength
                op = proc.matrix
                
                # Check if Pauli (efficient norm)
                if is_pauli(op)
                    # For Pauli L = sqrt(gamma)*P, L^dag L = gamma I.
                    nrm = norm(state.tensors[site])^2
                    dp_m = dt * gamma * nrm
                    push!(dp_m_list, dp_m)
                    push!(jump_candidates, (proc, op, "local_1", [site]))
                else
                    # Non-Pauli: apply and measure
                    T_orig = state.tensors[site]
                    @tensor T_new[l, p, r] := op[p, k] * T_orig[l, k, r]
                    nrm = norm(T_new)^2
                    dp_m = dt * gamma * nrm
                    push!(dp_m_list, dp_m)
                    push!(jump_candidates, (proc, op, "local_1", [site]))
                end
            end
        end

        # --- 2-site jumps starting at [site, site+1] ---
        if site < L
            for proc in noise_model.processes
                if isa(proc, LocalNoiseProcess) && length(proc.sites) == 2 && sort(proc.sites) == [site, site+1]
                    # Ensure we process this pair only once (when site == first site)
                     if proc.sites[1] == site
                        gamma = proc.strength
                        
                        if is_pauli_proc(proc)
                            nrm = norm(state.tensors[site])^2 # OC is at site
                            dp_m = dt * gamma * nrm
                            push!(dp_m_list, dp_m)
                            push!(jump_candidates, (proc, proc.matrix, "local_2", [site, site+1]))
                        else
                             # Non-Pauli: merge and apply
                             A1 = state.tensors[site]
                             A2 = state.tensors[site+1]
                             @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
                             
                             op = proc.matrix
                             op_ten = reshape(op, 2, 2, 2, 2)
                             @tensor Theta_new[l, p1n, p2n, r] := op_ten[p1n, p2n, p1, p2] * Theta[l, p1, p2, r]
                             
                             nrm = norm(Theta_new)^2
                             dp_m = dt * gamma * nrm
                             push!(dp_m_list, dp_m)
                             push!(jump_candidates, (proc, op, "local_2", [site, site+1]))
                        end
                    end
                end
            end
        end
    end
    
    # --- Long Range Jumps (Iterate processes directly) ---
    for proc in noise_model.processes
         if isa(proc, MPONoiseProcess)
             # MPO based Jump
             gamma = proc.strength
             
             # Check if it's a Pauli process with factors (Optimization)
             if is_pauli_proc(proc) && hasproperty(proc, :factors) && !isempty(proc.factors)
                 # Use global norm approx (assuming normalized state usually) or compute
                 # Ideally we should shift OC to sites and measure, but for global norm
                 # we can just use current norm if we assume state is unnormalized only due to dissipation
                 # But we might need to be careful.
                 # Let's use the same logic as before: current norm squared.
                 nrm = norm(state.tensors[state.orth_center])^2
                 dp_m = dt * gamma * nrm
                 push!(dp_m_list, dp_m)
                 push!(jump_candidates, (proc, proc.factors, "pauli_long_range", proc.sites))
             else
                 # General MPO Jump (e.g. Projector or Unitary-2pt components, or Pauli w/o factors)
                 # Need to apply MPO to get norm
                 temp_state = deepcopy(state)
                 apply_mpo_jump!(temp_state, proc.mpo, sim_params)
                 nrm = MPSModule.norm(temp_state)^2
                 dp_m = dt * gamma * nrm
                 push!(dp_m_list, dp_m)
                 push!(jump_candidates, (proc, proc.mpo, "mpo_general", proc.sites))
             end
             
         elseif isa(proc, LocalNoiseProcess) && length(proc.sites) == 2 && abs(proc.sites[2]-proc.sites[1]) > 1
             # Long range local process? Should be wrapped in MPO if it's not simple Pauli factors.
             # If it has factors, we can handle it.
             # Legacy support if needed, but MPONoiseProcess is preferred for long range.
         end
    end

    # Normalize probabilities
    total_dp = sum(dp_m_list)
    if total_dp == 0
        return Float64[], Any[]
    end
    
    probs = dp_m_list ./ total_dp
    return probs, jump_candidates
end

"""
    stochastic_process!(state, noise_model, dt, sim_params)

Perform a stochastic quantum jump.
"""
function stochastic_process!(state::MPS, noise_model::NoiseModel, dt::Float64, sim_params::AbstractSimConfig)
    dp = calculate_stochastic_factor(state)
    rng = Random.default_rng()
    
    if isempty(noise_model.processes) || rand(rng) >= dp
        # No Jump
        MPSModule.shift_orthogonality_center!(state, 1)
        MPSModule.normalize!(state) # Enforce norm 1
        return state
    end
    
    # Jump Occurs
    probs, candidates = create_probability_distribution(state, noise_model, dt, sim_params)
    
    if isempty(probs)
        MPSModule.shift_orthogonality_center!(state, 1)
        MPSModule.normalize!(state)
        return state
    end
    
    # Sample
    r = rand(rng)
    accum = 0.0
    chosen_idx = 1
    for (i, p) in enumerate(probs)
        accum += p
        if r <= accum
            chosen_idx = i
            break
        end
    end
    
    (proc, op, type, sites) = candidates[chosen_idx]
    
    if type == "local_1"
        site = sites[1]
        MPSModule.shift_orthogonality_center!(state, site)
        T = state.tensors[site]
        @tensor T_new[l, p, r] := op[p, k] * T[l, k, r]
        state.tensors[site] = T_new
        
    elseif type == "local_2"
        s1, s2 = sites
        MPSModule.shift_orthogonality_center!(state, s1)
        
        A1 = state.tensors[s1]
        A2 = state.tensors[s2]
        @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
        
        op_ten = reshape(op, 2, 2, 2, 2)
        @tensor Theta_new[l, p1n, p2n, r] := op_ten[p1n, p2n, p1, p2] * Theta[l, p1, p2, r]
        
        # Split back
        l_dim, _, _, r_dim = size(Theta_new)
        Mat = reshape(Theta_new, l_dim*2, 2*r_dim)
        F = svd(Mat)
        
        # Keep all singular values (no truncation for jump application)
        U, S, Vt = F.U, F.S, F.Vt
        rank = length(S)
        
        state.tensors[s1] = reshape(U, l_dim, 2, rank)
        state.tensors[s2] = reshape(Diagonal(S)*Vt, rank, 2, r_dim)
        
    elseif type == "mpo_pauli"
        # Apply factors if available
        if hasproperty(proc, :factors) && !isempty(proc.factors)
            MPSModule.shift_orthogonality_center!(state, proc.sites[1])
            state.tensors[proc.sites[1]] = permutedims(proc.factors[1] * permutedims(state.tensors[proc.sites[1]], (2,1,3)), (2,1,3))
            
            MPSModule.shift_orthogonality_center!(state, proc.sites[2])
            state.tensors[proc.sites[2]] = permutedims(proc.factors[2] * permutedims(state.tensors[proc.sites[2]], (2,1,3)), (2,1,3))
        else
            # Apply MPO
            apply_mpo_jump!(state, proc.mpo, sim_params)
        end
        
    elseif type == "mpo_general"
        apply_mpo_jump!(state, op, sim_params) # op is the mpo here
    end
    
    MPSModule.normalize!(state)
    return state
end



# --- Helpers ---

function is_pauli(op::AbstractMatrix)
    d = size(op, 1)
    # P'P = I
    PdagP = op' * op
    return isapprox(PdagP, Matrix(I, d, d), atol=1e-10)
end

function is_pauli_proc(proc::AbstractNoiseProcess)
    # Heuristic: check name
    if (occursin("pauli", proc.name) || occursin("crosstalk", proc.name)) && !occursin("projector", proc.name)
        return true
    end
    # Fallback to matrix check
    if hasproperty(proc, :matrix)
        return is_pauli(proc.matrix)
    end
    # Fallback to factors check
    if hasproperty(proc, :factors) && !isempty(proc.factors)
        return all(op -> is_pauli(op), proc.factors)
    end
    return false
end

end # module
