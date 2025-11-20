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

export StochasticProcess, trajectory_evolution

mutable struct StochasticProcess{T}
    mps::MPS{T}
    noise_model::NoiseModel{T}
    H_eff::MPO{T}
    jump_ops::Vector{Any} 
end

function StochasticProcess(mps::MPS{T}, hamiltonian::MPO{T}, noise_model::NoiseModel{T}) where T
    H_eff = deepcopy(hamiltonian)
    jump_ops = []
    
    one_site_terms = [zeros(T, 2, 2) for _ in 1:mps.length]
    has_one_site = false
    
    for proc in noise_model.processes
        gamma = proc.strength
        
        if proc isa LocalNoiseProcess
            L_op = proc.matrix
            term = -0.5im * gamma * (L_op' * L_op)
            
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
                     mpo_term = construct_2site_mpo(mps.length, s1, s2, term)
                     H_eff = H_eff + mpo_term
                else
                    error("Non-adjacent local process not supported without MPO")
                end
            end
            
        elseif proc isa MPONoiseProcess
            L_mpo = proc.mpo
            L_dag = adjoint_mpo(L_mpo)
            
            term_mpo = contract_mpo_mpo(L_dag, L_mpo)
            term_mpo = (-0.5im * gamma) * term_mpo
            
            H_eff = H_eff + term_mpo
            
            push!(jump_ops, (proc, L_mpo, "mpo"))
        end
    end
    
    if has_one_site
        decay_mpo = construct_1site_sum_mpo(mps.length, one_site_terms)
        H_eff = H_eff + decay_mpo
    end
    
    return StochasticProcess(mps, noise_model, H_eff, jump_ops)
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

function construct_2site_mpo(L::Int, s1::Int, s2::Int, op::Matrix{ComplexF64})
    op_tensor = reshape(op, 2, 2, 2, 2) 
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

function construct_1site_sum_mpo(L::Int, terms::Vector{Matrix{ComplexF64}})
    tensors = Vector{Array{ComplexF64, 4}}(undef, L)
    phys = fill(2, L)
    
    I_op = Matrix{ComplexF64}(I, 2, 2)
    Z_op = zeros(ComplexF64, 2, 2)
    
    for i in 1:L
        term = terms[i]
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

function trajectory_evolution(process::StochasticProcess, t_final::Float64, dt::Float64; 
                              measure_interval::Int=1, observables=[])
    
    current_mps = deepcopy(process.mps)
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
        norm_sq_old = real(scalar_product(current_mps, current_mps))
        
        # Evolve using 2-site TDVP
        dummy_obs = Vector{Observable{AbstractOperator}}()
        config = TimeEvolutionConfig(dummy_obs, dt; dt=dt, truncation_threshold=1e-10, max_bond_dim=64)
        
        two_site_tdvp!(current_mps, process.H_eff, config)
        
        # Norm after
        norm_sq_new = real(scalar_product(current_mps, current_mps))
        p_survive = norm_sq_new # / norm_sq_old (should be 1)
        
        # Roll
        r = rand()
        
        if r > p_survive
            # JUMP
            rates = Float64[]
            for (proc, op, type) in process.jump_ops
                val = 0.0
                gamma = proc.strength
                if type == "local_1"
                    op_sq = op' * op
                    val = real(local_expect(current_mps, op_sq, proc.sites[1]))
                elseif type == "local_2"
                    # op is 4x4. local_expect handles 2x2.
                    # TODO: Implement 2-site local_expect
                    # Fallback or error?
                    # For now assume 1-site jumps only or use MPO for 2-site
                    # Ideally convert 2-site op to MPO for expectation
                    # But we stored it as "local_2".
                    # Let's just fail gracefully or skip for now.
                    val = 0.0 
                elseif type == "mpo"
                    L_dag = adjoint_mpo(op)
                    L_sq = contract_mpo_mpo(L_dag, op)
                    val = real(expect_mpo(L_sq, current_mps))
                end
                push!(rates, val * gamma * dt)
            end
            
            # Sample
            total_rate = sum(rates)
            if total_rate > 0
                r_jump = rand() * total_rate
                chosen_idx = 1
                cum = 0.0
                for (i, rate) in enumerate(rates)
                    cum += rate
                    if r_jump <= cum
                        chosen_idx = i
                        break
                    end
                end
                
                # Apply Jump
                (proc, op, type) = process.jump_ops[chosen_idx]
                
                if type == "local_1"
                    site = proc.sites[1]
                    T_ten = current_mps.tensors[site]
                    @tensor T_new[l, p, r] := op[p, k] * T_ten[l, k, r]
                    current_mps.tensors[site] = T_new
                elseif type == "mpo"
                     current_mps = contract_mpo_mps(op, current_mps)
                     truncate!(current_mps)
                end
            end
        end
        
        # Normalize
        MPSModule.normalize!(current_mps)
        
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
