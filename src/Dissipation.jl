module DissipationModule

using LinearAlgebra
using TensorOperations
using StaticArrays
using ..MPSModule
using ..NoiseModule
using ..Decompositions
using ..SimulationConfigs

export apply_dissipation

function is_pauli(proc::LocalNoiseProcess)
    return proc.name in Set([
        "pauli_x", "pauli_y", "pauli_z",
        "crosstalk_xx", "crosstalk_yy", "crosstalk_zz",
        "crosstalk_xy", "crosstalk_yx", "crosstalk_zy",
        "crosstalk_zx", "crosstalk_yz", "crosstalk_xz"
    ])
end

function is_adjacent(proc::LocalNoiseProcess)
    return length(proc.sites) == 2 && abs(proc.sites[2] - proc.sites[1]) == 1
end

function is_longrange(proc::AbstractNoiseProcess)
    return length(proc.sites) == 2 && abs(proc.sites[2] - proc.sites[1]) > 1
end

"""
    apply_dissipation(mps::MPS, noise_model::NoiseModel, dt::Float64, sim_params)

Apply dissipation to the MPS using the Trotterized non-unitary operators from the noise model.
Sweeps Right-to-Left, applying 1-site and 2-site operators and shifting the orthogonality center.
"""
function apply_dissipation(mps::MPS{T}, noise_model::Union{NoiseModel{T}, Nothing}, dt::Float64, sim_params) where T
    # 1. Check if noise is present
    if isnothing(noise_model) || all(p.strength == 0 for p in noise_model.processes)
        # Shift orthogonality center to Left (1)
        for i in mps.length:-1:2
            MPSModule.shift_orthogonality_center!(mps, i - 1)
        end
        return
    end

    processed_projector_pairs = Set{Tuple{Tuple{Int, Int}, String}}()

    # Sweep Right-to-Left
    for i in mps.length:-1:1
        
        # 1. Apply 1-site dissipators on site i
        for proc in noise_model.processes
            if proc isa LocalNoiseProcess && length(proc.sites) == 1 && proc.sites[1] == i
                gamma = proc.strength
                if is_pauli(proc)
                    factor = exp(-0.5 * dt * gamma)
                    mps.tensors[i] .*= factor
                else
                    L_op = proc.matrix
                    mat = L_op' * L_op
                    op = exp(-0.5 * dt * gamma * mat) # (2,2)
                    
                    # Contract: op[p_new, p_old] * T[l, p_old, r]
                    T_ten = mps.tensors[i]
                    @tensor T_new[l, p, r] := op[p, k] * T_ten[l, k, r]
                    mps.tensors[i] = T_new
                end
            end
        end

        # 2. Apply 2-site dissipators on (i-1, i)
        if i > 1
            processes_here = [
                p for p in noise_model.processes 
                if length(p.sites) == 2 && maximum(p.sites) == i
            ]
            
            for proc in processes_here
                gamma = proc.strength
                
                if proc isa LocalNoiseProcess
                     if is_pauli(proc)
                        factor = exp(-0.5 * dt * gamma)
                        mps.tensors[i] .*= factor 
                        
                     elseif is_longrange(proc)
                        handle_longrange_scalar!(mps, proc, i, processed_projector_pairs, processes_here, dt)
                        
                     else
                        # General 2-site local (adjacent)
                        L_op = proc.matrix 
                        mat = L_op' * L_op
                        op = exp(-0.5 * dt * gamma * mat) # 4x4
                        
                        # Merge i-1 and i
                        A = mps.tensors[i-1] # (l, p1, k)
                        B = mps.tensors[i]   # (k, p2, r)
                        
                        @tensor C[l, p1, p2, r] := A[l, p1, k] * B[k, p2, r]
                        
                        # Apply Op
                        # Op is 4x4 acting on (p1, p2)
                        op_tensor = reshape(op, 2, 2, 2, 2) # (p1', p2', p1, p2)
                        @tensor C_new[l, p1_new, p2_new, r] := op_tensor[p1_new, p2_new, p1, p2] * C[l, p1, p2, r]
                        
                        # Split (Keep B Right Canonical approx, put S on B)
                        # We use standard SVD: C = U S V'.
                        # A = U. B = S V'.
                        
                        l_dim, _, _, r_dim = size(C_new)
                        C_split = reshape(C_new, l_dim * 2, 2 * r_dim)
                        F = svd(C_split)
                        U, S, Vt = F.U, F.S, F.Vt
                        
                        # Truncation
                        # Using params.threshold if available, else 1e-12
                        threshold = hasproperty(sim_params, :truncation_threshold) ? sim_params.truncation_threshold : (hasproperty(sim_params, :threshold) ? sim_params.threshold : 1e-12)
                        
                        # Calculate kept rank
                        norm_sq = sum(abs2, S)
                        current_sum = 0.0
                        rank = length(S)
                        for k in length(S):-1:1
                            current_sum += abs2(S[k])
                            if current_sum > threshold * norm_sq
                                rank = k
                                break
                            end
                        end
                        # Ensure at least 1
                        rank = max(1, rank)
                        
                        U = U[:, 1:rank]
                        S = S[1:rank]
                        Vt = Vt[1:rank, :]
                        
                        mps.tensors[i-1] = reshape(U, l_dim, 2, rank)
                        mps.tensors[i] = reshape(Diagonal(S) * Vt, rank, 2, r_dim)
                     end
                
                elseif proc isa MPONoiseProcess
                    handle_longrange_scalar!(mps, proc, i, processed_projector_pairs, processes_here, dt)
                end
            end
        end

        # Shift center left
        if i > 1
             MPSModule.shift_orthogonality_center!(mps, i - 1)
        end
    end
end

function handle_longrange_scalar!(mps, proc, i, processed_set, processes_here, dt)
    nm = proc.name
    
    if startswith(nm, "projector_")
        # Group +/- branches
        # Remove "projector_plus_" or "projector_minus_"
        base = replace(replace(nm, "projector_plus_" => ""), "projector_minus_" => "")
        sites_key = tuple(sort(proc.sites)...)
        pair_key = (sites_key, base)
        
        if pair_key in processed_set
            return
        end
        
        # Find mate
        mates = [q for q in processes_here
                 if tuple(sort(q.sites)...) == sites_key &&
                 endswith(q.name, base) &&
                 startswith(q.name, "projector_")]
                 
        if length(mates) != 2
            error("Incomplete projector pair for long-range channel $base on $sites_key")
        end
        
        gamma_pair = mates[1].strength + mates[2].strength
        mps.tensors[i] .*= exp(-0.5 * dt * gamma_pair)
        
        push!(processed_set, pair_key)
        
    elseif startswith(nm, "unitary2pt_") || startswith(nm, "unitary_gauss_")
        gamma_comp = proc.strength
        mps.tensors[i] .*= exp(-0.5 * dt * gamma_comp)
    else
        error("Non-Pauli long-range processes not fully implemented")
    end
end

end

