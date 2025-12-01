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
    # Python: return 1 - state.norm(0)
    # Here state.norm() usually returns L2 norm.
    # We use 1 - <psi|psi> (squared norm) which corresponds to probability loss.
    norm_sq = real(MPSModule.scalar_product(state, state))
    return 1.0 - norm_sq
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

    # Iterate over sites to minimize orthogonality center shifts (matching Python logic)
    for site in 1:L
        # Shift OC to site (if not 1 or L, shift to site-1? Python shifts to site-1)
        # Python: if site not in {0, L}: state.shift_orthogonality_center_right(site - 1)
        # In Julia (1-based), if site > 1.
        # We ensure OC is close to site. 
        # Actually, for 1-site ops at `site`, we want OC at `site`.
        # For 2-site ops at `site, site+1`, we want OC at `site`.
        
        # Helper to shift efficiently
        if site > 1
             MPSModule.shift_orthogonality_center!(state, site)
        else
             MPSModule.shift_orthogonality_center!(state, 1)
        end

        # --- 1-site jumps at this site ---
        for proc in noise_model.processes
            if length(proc.sites) == 1 && proc.sites[1] == site
                gamma = proc.strength
                op = proc.matrix
                
                # Check if Pauli (efficient norm)
                if is_pauli(op)
                    # For Pauli L = sqrt(gamma)*P, L^dag L = gamma I.
                    # <psi|L^dag L|psi> = gamma <psi|psi> = gamma * norm_sq_local
                    # Since we shifted OC to site, norm(tensor[site])^2 is global norm squared.
                    # Note: state might be unnormalized.
                    nrm = norm(state.tensors[site])^2
                    dp_m = dt * gamma * nrm
                    push!(dp_m_list, dp_m)
                    push!(jump_candidates, (proc, op, "local_1", [site]))
                else
                    # Non-Pauli: apply and measure
                    # Copy tensor to avoid modifying state
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
                if length(proc.sites) == 2 && proc.sites[1] == site && proc.sites[2] == site + 1
                    gamma = proc.strength
                    
                    if is_pauli_proc(proc)
                        # Pauli 2-site: L^dag L = gamma I
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
                    
                elseif length(proc.sites) == 2 && sort(proc.sites) == [site, site+1] && proc.sites[1] != site
                    # Handle case where sites are defined as [site+1, site] but we are at site.
                    # Usually sites are sorted in NoiseModel.
                end
            end
        end
    end
    
    # --- Long Range Jumps ---
    # Python iterates noise_model again or handles them specially? 
    # Python code: if "mpo" in process... inside the loop?
    # Python loop iterates sites. Inside check processes.
    # If process is long-range, Python code checks `if len(process["sites"]) == 2 and process["sites"][0] == site`.
    # So we should handle long-range here too if they start at `site`.
    
    for site in 1:L
         for proc in noise_model.processes
             # Long range: sites=[i, j] with |i-j|>1
             if length(proc.sites) == 2 && abs(proc.sites[2] - proc.sites[1]) > 1 && proc.sites[1] == site
                 # Long Range Pauli via factors
                 if is_pauli_proc(proc) && hasproperty(proc, :factors) && !isempty(proc.factors)
                     gamma = proc.strength
                     # Pauli: L^dag L = gamma I
                     # Need global norm.
                     # We can just use norm at current OC (which might not be site, but we can shift or just use what we have)
                     # In loop above, we shifted.
                     MPSModule.shift_orthogonality_center!(state, site)
                     nrm = norm(state.tensors[site])^2
                     dp_m = dt * gamma * nrm
                     push!(dp_m_list, dp_m)
                     push!(jump_candidates, (proc, proc.factors, "pauli_long_range", proc.sites))
                 else
                     # MPO based (ignored as per requirements "only support... via local factors")
                     # But Python code supports it. 
                     # User said: "only supports: ... long-range Pauli noise (implemented via local factors, not MPOs)"
                 end
             end
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
        
        # Split back (Full Rank / No Truncation for jump)
        # Use existing split helper from algorithms but force high rank?
        # Python: "no truncation or full rank"
        # We can just use split_mps_tensor_svd with very loose config or manually svd
        
        l_dim, _, _, r_dim = size(Theta_new)
        Mat = reshape(Theta_new, l_dim*2, 2*r_dim)
        F = svd(Mat)
        
        # Keep all singular values
        U, S, Vt = F.U, F.S, F.Vt
        rank = length(S)
        
        state.tensors[s1] = reshape(U, l_dim, 2, rank)
        state.tensors[s2] = reshape(Diagonal(S)*Vt, rank, 2, r_dim)
        
    elseif type == "pauli_long_range"
        # Apply factors
        factors = op
        s1, s2 = sites # unsorted usually? NoiseModel should sort. 
        # Assume sorted or factors correspond to sites order.
        # factors is vector [op1, op2]
        
        # Apply s1
        MPSModule.shift_orthogonality_center!(state, s1)
        T1 = state.tensors[s1]
        op1 = factors[1]
        @tensor T1_new[l, p, r] := op1[p, k] * T1[l, k, r]
        state.tensors[s1] = T1_new
        
        # Apply s2
        MPSModule.shift_orthogonality_center!(state, s2)
        T2 = state.tensors[s2]
        op2 = factors[2]
        @tensor T2_new[l, p, r] := op2[p, k] * T2[l, k, r]
        state.tensors[s2] = T2_new
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
