module StochasticProcessModule

using LinearAlgebra
using Random
using ..MPSModule
using ..MPOModule
using ..NoiseModule
using ..Algorithms
using ..SimulationConfigs
using ..GateLibrary
using TensorOperations

export StochasticProcess, trajectory_evolution, solve_jumps!

"""
    StochasticProcess{T}

Holds the pre-calculated effective Hamiltonian and jump operators for a given noise model.
Does NOT hold the MPS state (to avoid re-creation).
"""
struct StochasticProcess{T}
    H_eff::MPO{T}
    jump_ops::Vector{Any} 
    pauli_decays::Vector{Tuple{Int, Float64}}
end

"""
    StochasticProcess(hamiltonian, noise_model, L)

Constructor that pre-calculates H_eff and jump_ops.
"""
function StochasticProcess(hamiltonian::MPO{T}, noise_model::NoiseModel{T}) where T
    H_eff = deepcopy(hamiltonian)
    jump_ops = []
    pauli_decays = Tuple{Int, Float64}[]
    L = hamiltonian.length
    
    one_site_terms = [zeros(T, 2, 2) for _ in 1:L] # Keep as Matrix
    has_one_site = false
    
    for proc in noise_model.processes
        gamma = proc.strength
        
        if proc isa LocalNoiseProcess
            L_op = proc.matrix
            # Convert to Matrix for arithmetic with one_site_terms
            term = -0.5im * gamma * Matrix(L_op' * L_op)
            
            if length(proc.sites) == 1
                s = proc.sites[1]
                one_site_terms[s] .+= term
                has_one_site = true
                push!(jump_ops, (proc, L_op, "local_1"))
            else
                # 2-site local
                s1, s2 = sort(proc.sites)
                if s2 - s1 == 1
                     push!(jump_ops, (proc, L_op, "local_2"))
                     mpo_term = construct_2site_mpo(L, s1, s2, term)
                     H_eff = H_eff + mpo_term
                else
                    error("Non-adjacent local process not supported without MPO")
                end
            end
            
        elseif proc isa MPONoiseProcess
            # Check for optimized Pauli/Crosstalk case
            if hasproperty(proc, :factors) && !isempty(proc.factors)
                # Optimization: Don't add to H_eff (handle via scalar decay)
                # Store jump ops as factors
                push!(jump_ops, (proc, proc.factors, "pauli_long_range"))
                
                # Store decay for manual application
                # Apply decay to the second site (standard convention for right-to-left sweep or just consistently)
                # But here we apply globally or to specific site.
                # Let's apply to the max site index to be safe with sweeps or just last site.
                target_site = maximum(proc.sites)
                push!(pauli_decays, (target_site, gamma))
                
            else
                # Standard MPO processing
                L_mpo = proc.mpo
                L_dag = adjoint_mpo(L_mpo)
                
                term_mpo = contract_mpo_mpo(L_dag, L_mpo)
                term_mpo = (-0.5im * gamma) * term_mpo
                
                H_eff = H_eff + term_mpo
                
                push!(jump_ops, (proc, L_mpo, "mpo"))
            end
        end
    end
    
    if has_one_site
        decay_mpo = construct_1site_sum_mpo(L, one_site_terms)
        H_eff = H_eff + decay_mpo
    end
    
    return StochasticProcess(H_eff, jump_ops, pauli_decays)
end

function adjoint_mpo(a::MPO{T}) where T
    L = a.length
    new_tensors = Vector{Array{T, 4}}(undef, L)
    for i in 1:L
        T_ten = a.tensors[i]
        T_conj = conj(permutedims(T_ten, (1, 3, 2, 4)))
        new_tensors[i] = T_conj
    end
    return MPO(L, new_tensors, a.phys_dims)
end

function construct_2site_mpo(L::Int, s1::Int, s2::Int, op::AbstractMatrix)
    op_tensor = reshape(Matrix(op), 2, 2, 2, 2) 
    op_perm = permutedims(op_tensor, (1, 3, 2, 4))
    op_mat = reshape(op_perm, 4, 4)
    
    F = svd(op_mat)
    U, S, Vt = F.U, F.S, F.Vt
    rank = length(S)
    
    sqS = sqrt.(Diagonal(S))
    W1_mat = U * sqS
    W2_mat = sqS * Vt
    
    tensors = Vector{Array{ComplexF64, 4}}(undef, L)
    for i in 1:L
        if i == s1
            tensors[i] = reshape(W1_mat, 1, 2, 2, rank)
        elseif i == s2
            tensors[i] = reshape(W2_mat, rank, 2, 2, 1)
        else
            tensors[i] = reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1)
        end
    end
    
    return MPO(L, tensors, fill(2, L))
end

function construct_1site_sum_mpo(L::Int, terms::Vector{<:AbstractMatrix})
    tensors = Vector{Array{ComplexF64, 4}}(undef, L)
    phys = fill(2, L)
    
    I_op = Matrix{ComplexF64}(I, 2, 2)
    Z_op = zeros(ComplexF64, 2, 2)
    
    for i in 1:L
        term = Matrix(terms[i])
        W_block = Matrix{Matrix{ComplexF64}}(undef, 2, 2)
        W_block[1, 1] = I_op
        W_block[1, 2] = term
        W_block[2, 1] = Z_op
        W_block[2, 2] = I_op
        
        if i == 1
            W_final = W_block[1:1, :]
        elseif i == L
            W_final = Matrix{Matrix{ComplexF64}}(undef, 2, 1)
            W_final[1, 1] = term
            W_final[2, 1] = I_op
        else
            W_final = W_block
        end
        
        r, c = size(W_final)
        T_ten = zeros(ComplexF64, r, 2, 2, c)
        for row in 1:r, col in 1:c
            T_ten[row, :, :, col] = W_final[row, col]
        end
        tensors[i] = T_ten
    end
    
    return MPO(L, tensors, phys)
end

"""
    solve_jumps!(mps::MPS, process::StochasticProcess, dt::Float64)

Perform the MCWF jump logic on `mps` using pre-calculated `process`.
"""
function solve_jumps!(mps::MPS, process::StochasticProcess, dt::Float64)
    # Norm is already decayed by H_eff AND by manual pauli_decays
    norm_sq = real(scalar_product(mps, mps))
    
    r = rand()
    
    if r > norm_sq
        # Jump!
        rates = Float64[]
        
        for (proc, op, type) in process.jump_ops
            val = 0.0
            gamma = proc.strength
            
            if type == "local_1"
                op_sq = op' * op
                val = real(local_expect(mps, op_sq, proc.sites[1]))
            elseif type == "local_2"
                op_sq = op' * op
                val = real(local_expect_two_site(mps, op_sq, proc.sites[1], proc.sites[2]))
            elseif type == "mpo"
                L_dag = adjoint_mpo(op)
                L_sq = contract_mpo_mpo(L_dag, op)
                val = real(expect_mpo(L_sq, mps))
            elseif type == "pauli_long_range"
                # For Pauli: L'L = I. Expectation is norm_sq.
                # However, we need the rate relative to the *current* state.
                # The jump probability for channel k is dt * <psi|L_k' L_k|psi> / norm_sq (conditioned on jump)
                # But here we accumulate raw rates to choose ONE jump.
                # rate_k = <psi|L_k' L_k|psi>.
                # Since we already applied decay, psi is unnormalized.
                # val = <psi|psi> = norm_sq.
                val = norm_sq
            end
            
            # rate = <psi|L'L|psi> * dt ?
            # Wait. The total jump probability P_jump = Sum(rates) * dt.
            # And P_jump = 1 - norm_sq.
            # If norm_sq ~ 1 - Sum(gamma)*dt (for pure decay).
            # Then 1 - norm_sq ~ Sum(gamma)*dt.
            # So rates should be roughly gamma * norm_sq? or gamma?
            # If psi is normalized, rate = gamma.
            # If psi is decayed, rate = gamma * norm_sq.
            # The standard MCWF algorithm:
            # Probability of jump in dt is dp = sum_k dp_k. dp_k = dt * <psi|L_k' L_k|psi>.
            # If NO jump, we normalize.
            # If JUMP, we choose k with prob dp_k / dp.
            # Here we are INSIDE "if r > norm_sq" (so jump occurred).
            # We need to pick k with prob proportional to dp_k.
            # dp_k = dt * gamma * <psi|L_op' L_op|psi> (if strength separated).
            # For Pauli, <psi|L'L|psi> = <psi|psi> = norm_sq.
            # So rate propto gamma * norm_sq.
            
            push!(rates, max(0.0, val * gamma * dt)) 
        end
        
        total_rate = sum(rates)
        if total_rate > 0
            r_jump = rand() * total_rate
            current_sum = 0.0
            chosen_idx = 1
            for (i, rate) in enumerate(rates)
                current_sum += rate
                if r_jump <= current_sum
                    chosen_idx = i
                    break
                end
            end
            
            # Apply Jump
            (proc, op, type) = process.jump_ops[chosen_idx]
            
            if type == "local_1"
                site = proc.sites[1]
                T_ten = mps.tensors[site]
                @tensor T_new[l, p, r] := op[p, k] * T_ten[l, k, r]
                mps.tensors[site] = T_new
            elseif type == "local_2"
                site1, site2 = proc.sites
                A1 = mps.tensors[site1]
                A2 = mps.tensors[site2]
                @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
                
                op_tensor = reshape(op, 2, 2, 2, 2)
                @tensor Theta_prime[l, p1n, p2n, r] := op_tensor[p1n, p2n, p1, p2] * Theta[l, p1, p2, r]
                
                # SVD Split
                l_dim, _, _, r_dim = size(Theta_prime)
                Mat = reshape(Theta_prime, l_dim*2, 2*r_dim)
                F = svd(Mat)
                
                # Keep full rank for jump
                rank = length(F.S)
                U = F.U
                S = F.S
                Vt = F.Vt
                
                mps.tensors[site1] = reshape(U, l_dim, 2, rank)
                mps.tensors[site2] = reshape(Diagonal(S)*Vt, rank, 2, r_dim)
                
            elseif type == "mpo"
                 new_mps = contract_mpo_mps(op, mps)
                 mps.tensors = new_mps.tensors
                 MPSModule.truncate!(mps)
                 
            elseif type == "pauli_long_range"
                # Apply local factors directly
                factors = op # op is factors vector
                sites = proc.sites
                s1, s2 = sort(sites) # Assuming factors are [sigma, tau] for s1, s2
                
                # Apply to s1
                T1 = mps.tensors[s1]
                op1 = factors[1]
                @tensor T1_new[l, p, r] := op1[p, k] * T1[l, k, r]
                mps.tensors[s1] = T1_new
                
                # Apply to s2
                T2 = mps.tensors[s2]
                op2 = factors[2]
                @tensor T2_new[l, p, r] := op2[p, k] * T2[l, k, r]
                mps.tensors[s2] = T2_new
            end
        end
    end
    
    MPSModule.normalize!(mps)
end

function trajectory_evolution(process::StochasticProcess, mps::MPS, t_final::Float64, dt::Float64; 
                              measure_interval::Int=1, observables=[])
    
    current_mps = deepcopy(mps)
    steps = floor(Int, t_final / dt)
    
    results = Dict{String, Vector{ComplexF64}}()
    for obs in observables
        results[obs.name] = ComplexF64[]
    end
    results["time"] = ComplexF64[]
    
    time = 0.0
    for step in 1:steps
        time += dt
        
        # 1. TDVP Step with H_eff
        dummy_obs = Vector{Observable{AbstractOperator}}()
        config = TimeEvolutionConfig(dummy_obs, dt; dt=dt, truncation_threshold=1e-10, max_bond_dim=64)
        
        if current_mps.length == 1
            single_site_tdvp!(current_mps, process.H_eff, config)
        else
            two_site_tdvp!(current_mps, process.H_eff, config)
        end
        
        # 2. Apply Pauli Decays (Manual non-unitary evolution)
        for (site, gamma) in process.pauli_decays
            factor = exp(-0.5 * dt * gamma)
            current_mps.tensors[site] .*= factor
        end
        
        # 3. Jumps
        solve_jumps!(current_mps, process, dt)
        
        # Measurements
        if step % measure_interval == 0
            push!(results["time"], ComplexF64(time))
            for obs in observables
                val = expect(current_mps, obs)
                push!(results[obs.name], val)
            end
        end
    end
    
    return results
end

end
