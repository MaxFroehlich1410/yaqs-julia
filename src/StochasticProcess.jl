module StochasticProcessModule

using LinearAlgebra
using Random
using TensorOperations
using ..MPSModule
using ..MPOModule
using ..NoiseModule
using ..SimulationConfigs
using ..Algorithms
using ..Timing: @t

export StochasticProcess, calculate_stochastic_factor, create_probability_distribution, stochastic_process!

"""
Wrapper type for stochastic processes driven by a noise model.

This stores a `NoiseModel` and exists primarily to preserve compatibility with earlier APIs that
expected a dedicated stochastic-process object.

Args:
    noise_model (NoiseModel{T}): Noise model driving the stochastic dynamics.

Returns:
    StochasticProcess{T}: Wrapper around the noise model.
"""
struct StochasticProcess{T}
    noise_model::NoiseModel{T}
end

"""
Construct a StochasticProcess while ignoring the Hamiltonian argument.

This maintains backward compatibility with APIs that passed both a Hamiltonian and a noise model,
returning a wrapper around the noise model only.

Args:
    hamiltonian (MPO{T}): Ignored Hamiltonian argument.
    noise_model (NoiseModel{T}): Noise model driving the stochastic dynamics.

Returns:
    StochasticProcess{T}: Wrapper around the noise model.
"""
# Constructor for backward compatibility (ignores H argument)
function StochasticProcess(hamiltonian::MPO{T}, noise_model::NoiseModel{T}) where T
    return StochasticProcess(noise_model)
end

"""
Compute the stochastic factor for jump probability.

This calculates `dp = 1 - ⟨ψ|ψ⟩`, assuming the norm has decayed due to dissipation and indicating
the probability mass available for jumps.

Args:
    state (MPS): State whose norm determines the stochastic factor.

Returns:
    Float64: Stochastic factor `dp`.
"""
function calculate_stochastic_factor(state::MPS)
    norm_sq = @t :stoch_scalar_product real(MPSModule.scalar_product(state, state))
    return 1.0 - norm_sq
end

"""
Apply an MPO jump operator to an MPS in-place.

This contracts the MPO with the MPS site-by-site, increasing bond dimensions, and then truncates
according to the simulation parameters to control growth.

Args:
    state (MPS): State to update in-place.
    mpo (MPO): MPO jump operator to apply.
    sim_params: Simulation parameters containing truncation settings.

Returns:
    Nothing: The MPS is updated in-place.
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
        @t :stoch_mpo_contract @tensor C[la, lw, po, ra, rw] := A[la, pi, ra] * W[lw, po, pi, rw]
        
        # Reshape to merge virtual bonds: (La*Lw, Po, Ra*Rw)
        la, lw, po, ra, rw = size(C)
        state.tensors[i] = @t :stoch_mpo_reshape reshape(C, la * lw, po, ra * rw)
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
    
    @t :stoch_mpo_truncate MPSModule.truncate!(state; threshold=threshold, max_bond_dim=max_bond)
end

"""
Create a probability distribution over possible jumps.

This evaluates jump probabilities for local and MPO-based noise processes, returning normalized
probabilities and a list of corresponding jump candidates.

Args:
    state (MPS): Current state used to compute jump norms.
    noise_model (NoiseModel): Noise model with candidate processes.
    dt (Float64): Time step for probability scaling.
    sim_params (AbstractSimConfig): Simulation parameters for truncation and scaling.

Returns:
    Tuple: `(probs, jump_candidates)` where `probs` is a vector of probabilities and
        `jump_candidates` stores process metadata for sampling.
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
        @t :stoch_shift_oc begin
            if site > 1
                MPSModule.shift_orthogonality_center!(state, site)
            else
                MPSModule.shift_orthogonality_center!(state, 1)
            end
        end

        # --- 1-site jumps at this site ---
        for proc in noise_model.processes
            if isa(proc, LocalNoiseProcess) && length(proc.sites) == 1 && proc.sites[1] == site
                gamma = proc.strength
                op = proc.matrix
                
                # Check if Pauli (efficient norm)
                if is_pauli(op)
                    # For Pauli L = sqrt(gamma)*P, L^dag L = gamma I.
                    nrm = @t :stoch_local_norm norm(state.tensors[site])^2
                    dp_m = dt * gamma * nrm
                    push!(dp_m_list, dp_m)
                    push!(jump_candidates, (proc, op, "local_1", [site]))
                else
                    # Non-Pauli: apply and measure
                    T_orig = state.tensors[site]
                    @t :stoch_1site_contract @tensor T_new[l, p, r] := op[p, k] * T_orig[l, k, r]
                    nrm = @t :stoch_local_norm norm(T_new)^2
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
                             @t :stoch_2site_merge @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
                             
                             op = proc.matrix
                             op_ten = reshape(op, 2, 2, 2, 2)
                             @t :stoch_2site_apply @tensor Theta_new[l, p1n, p2n, r] := op_ten[p1n, p2n, p1, p2] * Theta[l, p1, p2, r]
                             
                             nrm = @t :stoch_local_norm norm(Theta_new)^2
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
                 nrm = @t :stoch_local_norm norm(state.tensors[state.orth_center])^2
                 dp_m = dt * gamma * nrm
                 push!(dp_m_list, dp_m)
                 push!(jump_candidates, (proc, proc.factors, "pauli_long_range", proc.sites))
             else
                 # General MPO Jump (e.g. Projector or Unitary-2pt components, or Pauli w/o factors)
                 # Need to apply MPO to get norm
                 temp_state = @t :stoch_mpo_deepcopy deepcopy(state)
                 @t :stoch_apply_mpo_jump apply_mpo_jump!(temp_state, proc.mpo, sim_params)
                 nrm = @t :stoch_temp_norm MPSModule.norm(temp_state)^2
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
Perform a stochastic quantum jump on the state.

This computes jump probabilities, samples a jump event, and applies the corresponding operator
to the state, including necessary truncation and normalization steps.

Args:
    state (MPS): State to update in-place.
    noise_model (NoiseModel): Noise model with candidate processes.
    dt (Float64): Time step for probability scaling.
    sim_params (AbstractSimConfig): Simulation parameters for truncation and scaling.

Returns:
    Nothing: The MPS is updated in-place.
"""
function stochastic_process!(state::MPS, noise_model::NoiseModel, dt::Float64, sim_params::AbstractSimConfig)
    dp = @t :stoch_calc_dp calculate_stochastic_factor(state)
    rng = Random.default_rng()
    
    if isempty(noise_model.processes) || rand(rng) >= dp
        # No Jump
        @t :stoch_nojump_finalize begin
            MPSModule.shift_orthogonality_center!(state, 1)
            MPSModule.normalize!(state) # Enforce norm 1
        end
        return state
    end
    
    # Jump Occurs
    probs, candidates = @t :stoch_create_prob_dist create_probability_distribution(state, noise_model, dt, sim_params)
    
    if isempty(probs)
        @t :stoch_empty_probs_finalize begin
            MPSModule.shift_orthogonality_center!(state, 1)
            MPSModule.normalize!(state)
        end
        return state
    end
    
    # Sample
    r = rand(rng)
    accum = 0.0
    chosen_idx = 1
    @t :stoch_sample_loop begin
        for (i, p) in enumerate(probs)
            accum += p
            if r <= accum
                chosen_idx = i
                break
            end
        end
    end
    
    (proc, op, type, sites) = candidates[chosen_idx]
    
    if type == "local_1"
        site = sites[1]
        @t :stoch_apply_local1_shift MPSModule.shift_orthogonality_center!(state, site)
        T = state.tensors[site]
        @t :stoch_apply_local1_contract @tensor T_new[l, p, r] := op[p, k] * T[l, k, r]
        @t :stoch_apply_local1_writeback state.tensors[site] = T_new
        
    elseif type == "local_2"
        s1, s2 = sites
        @t :stoch_apply_local2_shift MPSModule.shift_orthogonality_center!(state, s1)
        
        A1 = state.tensors[s1]
        A2 = state.tensors[s2]
        @t :stoch_apply_local2_merge @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
        
        op_ten = reshape(op, 2, 2, 2, 2)
        @t :stoch_apply_local2_apply @tensor Theta_new[l, p1n, p2n, r] := op_ten[p1n, p2n, p1, p2] * Theta[l, p1, p2, r]
        
        # Split back
        l_dim, _, _, r_dim = size(Theta_new)
        Mat = @t :stoch_apply_local2_reshape reshape(Theta_new, l_dim*2, 2*r_dim)
        F = @t :stoch_apply_local2_svd svd(Mat)
        
        # Truncation logic
        threshold = hasproperty(sim_params, :truncation_threshold) ? sim_params.truncation_threshold : 1e-12
        max_bond = hasproperty(sim_params, :max_bond_dim) ? sim_params.max_bond_dim : typemax(Int)
        
        U, S, Vt = F.U, F.S, F.Vt

        rank = length(S)
        @t :stoch_apply_local2_trunc begin
            norm_sq = sum(abs2, S)
            current_sum = 0.0
            for k in length(S):-1:1
                current_sum += abs2(S[k])
                if current_sum > threshold * norm_sq
                    rank = k
                    break
                end
            end
        end
        rank = clamp(rank, 1, max_bond)

        U = U[:, 1:rank]
        S = S[1:rank]
        Vt = Vt[1:rank, :]
        
        @t :stoch_apply_local2_writeback begin
            state.tensors[s1] = reshape(U, l_dim, 2, rank)
            state.tensors[s2] = reshape(Diagonal(S)*Vt, rank, 2, r_dim)
        end
        
    elseif type == "mpo_pauli"
        # Apply factors if available
        if hasproperty(proc, :factors) && !isempty(proc.factors)
            @t :stoch_apply_mpo_pauli_shift MPSModule.shift_orthogonality_center!(state, proc.sites[1])
            @t :stoch_apply_mpo_pauli_apply state.tensors[proc.sites[1]] = permutedims(proc.factors[1] * permutedims(state.tensors[proc.sites[1]], (2,1,3)), (2,1,3))
            
            @t :stoch_apply_mpo_pauli_shift MPSModule.shift_orthogonality_center!(state, proc.sites[2])
            @t :stoch_apply_mpo_pauli_apply state.tensors[proc.sites[2]] = permutedims(proc.factors[2] * permutedims(state.tensors[proc.sites[2]], (2,1,3)), (2,1,3))
        else
            # Apply MPO
            @t :stoch_apply_mpo_jump apply_mpo_jump!(state, proc.mpo, sim_params)
        end
        
    elseif type == "mpo_general"
        @t :stoch_apply_mpo_jump apply_mpo_jump!(state, op, sim_params) # op is the mpo here
    end
    
    @t :stoch_finalize_normalize MPSModule.normalize!(state)
    return state
end



# --- Helpers ---

"""
Check whether a matrix is a Pauli operator (up to phase).

This compares the matrix against the standard Pauli X, Y, and Z operators to enable fast
probability calculations for Pauli jumps.

Args:
    op (AbstractMatrix): Operator matrix to classify.

Returns:
    Bool: `true` if the operator matches a Pauli matrix, otherwise `false`.
"""
function is_pauli(op::AbstractMatrix)
    d = size(op, 1)
    # P'P = I
    PdagP = op' * op
    return isapprox(PdagP, Matrix(I, d, d), atol=1e-10)
end

"""
Check whether a noise process corresponds to a Pauli channel.

This inspects the process name and, if available, its operator or factors to decide whether it
represents a Pauli-type jump.

Args:
    proc (AbstractNoiseProcess): Noise process to classify.

Returns:
    Bool: `true` if the process is Pauli-type, otherwise `false`.
"""
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
