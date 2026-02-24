module DissipationModule

using LinearAlgebra
using TensorOperations
using StaticArrays
using ..MPSModule
using ..NoiseModule
using ..Decompositions
using ..SimulationConfigs
using ..Timing: @t

export apply_dissipation

"""
Check whether a local noise process is Pauli-type.

This matches the process name against supported Pauli and crosstalk Pauli channels to determine
whether the dissipation can be applied as a simple scalar decay.

Args:
    proc (LocalNoiseProcess): Noise process to classify.

Returns:
    Bool: `true` if the process is Pauli-type, otherwise `false`.
"""
function is_pauli(proc::LocalNoiseProcess)
    return proc.name in Set([
        "pauli_x", "pauli_y", "pauli_z",
        "crosstalk_xx", "crosstalk_yy", "crosstalk_zz",
        "crosstalk_xy", "crosstalk_yx", "crosstalk_zy",
        "crosstalk_zx", "crosstalk_yz", "crosstalk_xz"
    ])
end

"""
Check whether a local noise process acts on adjacent sites.

This returns true when the process has two sites and their indices differ by exactly one.

Args:
    proc (LocalNoiseProcess): Noise process to classify.

Returns:
    Bool: `true` if the process targets adjacent sites, otherwise `false`.
"""
function is_adjacent(proc::LocalNoiseProcess)
    return length(proc.sites) == 2 && abs(proc.sites[2] - proc.sites[1]) == 1
end

"""
Check whether a noise process is long-range.

This returns true when the process acts on two sites that are separated by more than one bond.

Args:
    proc (AbstractNoiseProcess): Noise process to classify.

Returns:
    Bool: `true` if the process is long-range, otherwise `false`.
"""
function is_longrange(proc::AbstractNoiseProcess)
    return length(proc.sites) == 2 && abs(proc.sites[2] - proc.sites[1]) > 1
end

"""
Apply dissipative evolution to an MPS using a noise model.

This performs a right-to-left sweep, applying one-site and two-site non-unitary operators derived
from the noise model, and shifts the orthogonality center as it progresses.

Args:
    mps (MPS): State to update in-place.
    noise_model (Union{NoiseModel, Nothing}): Noise model or `nothing` to skip dissipation.
    dt (Float64): Time step for the dissipative evolution.
    sim_params: Simulation configuration with truncation settings.

Returns:
    Nothing: The MPS is updated in-place.
"""
function apply_dissipation(mps::MPS{T}, noise_model::Union{NoiseModel{T}, Nothing}, dt::Float64, sim_params) where T
    # 1. Check if noise is present
    if isnothing(noise_model) || all(p.strength == 0 for p in noise_model.processes)
        # Shift orthogonality center to Left (1)
        @t :dissipation_shift_only begin
            for i in mps.length:-1:2
                MPSModule.shift_orthogonality_center!(mps, i - 1)
            end
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
                    factor = @t :dissipation_exp_scalar exp(-0.5 * dt * gamma)
                    @t :dissipation_1site_pauli_scale mps.tensors[i] .*= factor
                else
                    L_op = proc.matrix
                    mat = @t :dissipation_1site_matmul (L_op' * L_op)
                    op = @t :dissipation_1site_expm exp(-0.5 * dt * gamma * mat) # (2,2)
                    
                    # Contract: op[p_new, p_old] * T[l, p_old, r]
                    T_ten = mps.tensors[i]
                    @t :dissipation_1site_contract @tensor T_new[l, p, r] := op[p, k] * T_ten[l, k, r]
                    @t :dissipation_1site_writeback mps.tensors[i] = T_new
                end
            end
        end

        # 2. Apply 2-site dissipators on (i-1, i)
        if i > 1
            processes_here = @t :dissipation_collect_2site_processes begin
                [p for p in noise_model.processes if length(p.sites) == 2 && maximum(p.sites) == i]
            end
            
            for proc in processes_here
                gamma = proc.strength
                
                if proc isa LocalNoiseProcess
                     if is_pauli(proc)
                        factor = @t :dissipation_exp_scalar exp(-0.5 * dt * gamma)
                        @t :dissipation_2site_pauli_scale mps.tensors[i] .*= factor 
                        
                     elseif is_longrange(proc)
                        @t :dissipation_longrange_scalar handle_longrange_scalar!(mps, proc, i, processed_projector_pairs, processes_here, dt)
                        
                     else
                        # General 2-site local (adjacent)
                        L_op = proc.matrix 
                        mat = @t :dissipation_2site_matmul (L_op' * L_op)
                        op = @t :dissipation_2site_expm exp(-0.5 * dt * gamma * mat) # 4x4
                        
                        # Merge i-1 and i
                        A = mps.tensors[i-1] # (l, p1, k)
                        B = mps.tensors[i]   # (k, p2, r)
                        
                        @t :dissipation_2site_merge @tensor C[l, p1, p2, r] := A[l, p1, k] * B[k, p2, r]
                        
                        # Apply Op
                        # Op is 4x4 acting on (p1, p2)
                        op_tensor = reshape(op, 2, 2, 2, 2) # (p1', p2', p1, p2)
                        @t :dissipation_2site_apply @tensor C_new[l, p1_new, p2_new, r] := op_tensor[p1_new, p2_new, p1, p2] * C[l, p1, p2, r]
                        
                        # Split (Keep B Right Canonical approx, put S on B)
                        # We use standard SVD: C = U S V'.
                        # A = U. B = S V'.
                        
                        l_dim, _, _, r_dim = size(C_new)
                        C_split = @t :dissipation_2site_reshape reshape(C_new, l_dim * 2, 2 * r_dim)
                        F = @t :dissipation_2site_svd svd(C_split)
                        U, S, Vt = F.U, F.S, F.Vt
                        
                        # Truncation
                        # Using params.threshold if available, else 1e-12
                        threshold = hasproperty(sim_params, :truncation_threshold) ? sim_params.truncation_threshold : (hasproperty(sim_params, :threshold) ? sim_params.threshold : 1e-12)
                        max_bond = hasproperty(sim_params, :max_bond_dim) ? sim_params.max_bond_dim : typemax(Int)
                        
                        # Calculate kept rank
                        rank = length(S)
                        @t :dissipation_2site_trunc begin
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
                        # Ensure at least 1 and at most max_bond
                        rank = clamp(rank, 1, max_bond)
                        
                        U = U[:, 1:rank]
                        S = S[1:rank]
                        Vt = Vt[1:rank, :]
                        
                        @t :dissipation_2site_writeback begin
                            mps.tensors[i-1] = reshape(U, l_dim, 2, rank)
                            mps.tensors[i] = reshape(Diagonal(S) * Vt, rank, 2, r_dim)
                        end
                     end
                
                elseif proc isa MPONoiseProcess
                    @t :dissipation_longrange_scalar handle_longrange_scalar!(mps, proc, i, processed_projector_pairs, processes_here, dt)
                end
            end
        end

        # Shift center left
        if i > 1
             @t :center MPSModule.shift_orthogonality_center!(mps, i - 1)
        end
    end
end

"""
Handle scalar-only long-range dissipation updates for a site.

This applies scalar decay factors for long-range noise processes, handling paired projector channels
and other supported long-range process types.

Args:
    mps: MPS state to update in-place.
    proc: Noise process being handled.
    i: Site index currently being processed.
    processed_set: Set tracking already-processed projector pairs.
    processes_here: Collection of processes active at the current site.
    dt: Time step for the dissipative evolution.

Returns:
    Nothing: The MPS tensor at site `i` is scaled in-place.

Raises:
    ErrorException: If a projector pair is incomplete or the process type is unsupported.
"""
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
        factor = @t :dissipation_longrange_exp_scalar exp(-0.5 * dt * gamma_pair)
        @t :dissipation_longrange_scale mps.tensors[i] .*= factor
        
        push!(processed_set, pair_key)
        
    elseif startswith(nm, "unitary2pt_") || startswith(nm, "unitary_gauss_")
        gamma_comp = proc.strength
        factor = @t :dissipation_longrange_exp_scalar exp(-0.5 * dt * gamma_comp)
        @t :dissipation_longrange_scale mps.tensors[i] .*= factor
    
    elseif startswith(nm, "crosstalk_") || startswith(nm, "pauli_")
        # Standard Pauli MPO dissipation (L†L = I, so it's just scalar decay)
        gamma = proc.strength
        factor = @t :dissipation_longrange_exp_scalar exp(-0.5 * dt * gamma)
        @t :dissipation_longrange_scale mps.tensors[i] .*= factor
    else
        error("Non-Pauli long-range processes not fully implemented for: $nm")
    end
end

end

