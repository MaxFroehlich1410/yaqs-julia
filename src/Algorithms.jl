module Algorithms

using LinearAlgebra
using TensorOperations
using KrylovKit
using ..MPSModule
using ..MPOModule
using ..SimulationConfigs
using ..Decompositions

export single_site_tdvp!, two_site_tdvp!

# --- Krylov Subspace Methods ---

"""
    expm_krylov(A_func, v, dt, k)

Compute exp(-im * dt * A) * v using Krylov subspace.
"""
function expm_krylov(A_func::Function, v::AbstractArray{T}, dt::Number, k::Int) where T
    norm_v = norm(v)
    if norm_v == 0
        return v
    end
        
    t_val = -1im * dt
    val, info = exponentiate(A_func, t_val, v; tol=1e-12, krylovdim=k, maxiter=1, ishermitian=false)
    return val
end

# --- Environment Helpers ---

function make_identity_env(dim::Int, mpo_dim::Int)
    E = zeros(ComplexF64, dim, mpo_dim, dim)
    for i in 1:dim
        E[i, 1, i] = 1.0
    end
    return E
end

function update_left_environment(A, W, E_left)
    @tensor T1[bra_l, mpo_l, p_in, ket_r] := E_left[bra_l, mpo_l, k] * A[k, p_in, ket_r]
    @tensor T2[bra_l, ket_r, p_out, mpo_r] := T1[bra_l, k_ml, k_pin, ket_r] * W[k_ml, p_out, k_pin, mpo_r]
    @tensor E_next[bra_r, mpo_r, ket_r] := T2[k_bl, ket_r, k_pout, mpo_r] * conj(A[k_bl, k_pout, bra_r])
    return E_next
end

function update_right_environment(A, W, E_right)
    @tensor T1[ket_l, p_in, bra_r, mpo_r] := A[ket_l, p_in, k] * E_right[bra_r, mpo_r, k]
    @tensor T2[mpo_l, p_out, ket_l, bra_r] := W[mpo_l, p_out, k_pin, k_mr] * T1[ket_l, k_pin, bra_r, k_mr]
    @tensor E_next[bra_l, mpo_l, ket_l] := T2[mpo_l, k_pout, ket_l, k_br] * conj(A[bra_l, k_pout, k_br])
    return E_next
end

# --- Projectors ---

function project_site(A, L, R, W)
    @tensor T1[bra_l, mpo_l, p_in, ket_r] := L[bra_l, mpo_l, k] * A[k, p_in, ket_r]
    @tensor T2[bra_l, ket_r, p_out, mpo_r] := T1[bra_l, k_ml, k_pin, ket_r] * W[k_ml, p_out, k_pin, mpo_r]
    @tensor A_new[bra_l, p_out, bra_r] := T2[bra_l, k_kr, p_out, k_mr] * R[bra_r, k_mr, k_kr]
    return A_new
end

function project_bond(C, L, R)
    @tensor T1[bra_l, mpo, ket_r] := L[bra_l, mpo, k] * C[k, ket_r]
    @tensor C_new[bra_l, bra_r] := T1[bra_l, k_mpo, k_kr] * R[bra_r, k_mpo, k_kr]
    return C_new
end

# --- SVD Helper ---

function split_mps_tensor_svd(Theta, l_virt, p1, p2, r_virt, config)
    # Reshape for SVD (L*p1, p2*R)
    Mat = reshape(Theta, l_virt*p1, p2*r_virt)
    F = svd(Mat)
    
    # Truncation
    discarded_sq = 0.0
    keep_rank = length(F.S)
    threshold = config.truncation_threshold
    min_keep = 2
    
    for k in length(F.S):-1:1
        discarded_sq += F.S[k]^2
        if discarded_sq >= threshold
            keep_rank = max(k, min_keep)
            break
        end
    end
    
    keep = clamp(keep_rank, 1, config.max_bond_dim)
    
    U = F.U[:, 1:keep]
    S = F.S[1:keep]
    Vt = F.Vt[1:keep, :]
    
    return U, S, Vt, keep
end

# --- Main Dispatch Functions ---

function single_site_tdvp!(state::MPS, H::MPO, config::TimeEvolutionConfig)
    # Hamiltonian Simulation: Symmetric Sweep (Forward + Backward) with dt/2
    _tdvp_sweep_hamiltonian_1site!(state, H, config)
end

function single_site_tdvp!(state::MPS, H::MPO, config::Union{MeasurementConfig, StrongMeasurementConfig})
    # Circuit Simulation: Single Forward Sweep with dt=2 logic
    _tdvp_sweep_circuit_1site!(state, H, config)
end

function two_site_tdvp!(state::MPS, H::MPO, config::TimeEvolutionConfig)
    # Hamiltonian Simulation: Symmetric Sweep
    _tdvp_sweep_hamiltonian_2site!(state, H, config)
end

function two_site_tdvp!(state::MPS, H::MPO, config::Union{MeasurementConfig, StrongMeasurementConfig})
    # Circuit Simulation: Single Forward Sweep
    _tdvp_sweep_circuit_2site!(state, H, config)
end

# --- Implementation of Sweeps ---

# 1. Hamiltonian 1-Site (Forward + Backward)
function _tdvp_sweep_hamiltonian_1site!(state, H, config)
    shift_orthogonality_center!(state, 1)
    L = state.length
    dt = config.dt
    
    # Init Envs
    E_left, E_right = _init_envs(state, H)
    
    # Forward (1 -> L)
    for i in 1:L
        W = H.tensors[i]
        func_site(x) = project_site(x, E_left[i], E_right[i+1], W)
        
        # Evolve Site (dt/2)
        state.tensors[i] = expm_krylov(func_site, state.tensors[i], dt/2, 25)
        
        if i < L
            l, p, r = size(state.tensors[i])
            A_mat = reshape(state.tensors[i], l*p, r)
            F = qr(A_mat)
            Q_mat = Matrix(F.Q)
            state.tensors[i] = reshape(Q_mat, l, p, size(Q_mat, 2))
            R_mat = Matrix(F.R)
            
            E_left[i+1] = update_left_environment(state.tensors[i], W, E_left[i])
            
            # Evolve Bond Backward (-dt/2)
            func_bond(x) = project_bond(x, E_left[i+1], E_right[i+1])
            C_new = expm_krylov(func_bond, R_mat, -dt/2, 25)
            
            @tensor Next[l, p, r] := C_new[l, k] * state.tensors[i+1][k, p, r]
            state.tensors[i+1] = Next
        end
    end
    
    # Backward (L -> 1)
    for i in L:-1:1
        W = H.tensors[i]
        func_site(x) = project_site(x, E_left[i], E_right[i+1], W)
        
        state.tensors[i] = expm_krylov(func_site, state.tensors[i], dt/2, 25)
        
        if i > 1
            l, p, r = size(state.tensors[i])
            A_mat = reshape(state.tensors[i], l, p*r)
            F = lq(A_mat)
            Q_mat = Matrix(F.Q)
            state.tensors[i] = reshape(Q_mat, size(Q_mat, 1), p, r)
            L_mat = Matrix(F.L)
            
            E_right[i] = update_right_environment(state.tensors[i], W, E_right[i+1])
            
            func_bond(x) = project_bond(x, E_left[i], E_right[i])
            C_new = expm_krylov(func_bond, L_mat, -dt/2, 25)
            
            @tensor Prev[l, p, r] := state.tensors[i-1][l, p, k] * C_new[k, r]
            state.tensors[i-1] = Prev
        end
    end
    println("max bond dim: ", write_max_bond_dim(state))
end

# 2. Circuit 1-Site (Forward Only, dt=2 logic)
function _tdvp_sweep_circuit_1site!(state, H, config)
    shift_orthogonality_center!(state, 1)
    L = state.length
    
    # Circuit Logic: dt starts at 2.0
    dt = 2.0
    
    E_left, E_right = _init_envs(state, H)
    
    # Forward Sweep (1 -> L-1)
    for i in 1:(L-1)
        W = H.tensors[i]
        func_site(x) = project_site(x, E_left[i], E_right[i+1], W)
        
        # Evolve Site (0.5 * dt = 1.0)
        state.tensors[i] = expm_krylov(func_site, state.tensors[i], 0.5 * dt, 25)
        
        l, p, r = size(state.tensors[i])
        A_mat = reshape(state.tensors[i], l*p, r)
        F = qr(A_mat)
        state.tensors[i] = reshape(Matrix(F.Q), l, p, size(F.Q, 2))
        R_mat = Matrix(F.R)
        
        E_left[i+1] = update_left_environment(state.tensors[i], W, E_left[i])
        
        # Evolve Bond (-0.5 * dt = -1.0)
        func_bond(x) = project_bond(x, E_left[i+1], E_right[i+1])
        C_new = expm_krylov(func_bond, R_mat, -0.5 * dt, 25)
        
        @tensor Next[l, p, r] := C_new[l, k] * state.tensors[i+1][k, p, r]
        state.tensors[i+1] = Next
    end
    
    # Final Site Update (dt becomes 1.0)
    dt = 1.0
    W = H.tensors[L]
    func_site_last(x) = project_site(x, E_left[L], E_right[L+1], W)
    state.tensors[L] = expm_krylov(func_site_last, state.tensors[L], dt, 25)
    
    # No Backward Sweep
end

# 3. Hamiltonian 2-Site
function _tdvp_sweep_hamiltonian_2site!(state, H, config)
    shift_orthogonality_center!(state, 1)
    L = state.length
    dt = config.dt
    E_left, E_right = _init_envs(state, H)
    
    # Forward Sweep (1 -> L-2)
    # Evolve 2-site by dt/2, Split Right, Evolve Bond/RightSite back by -dt/2
    for i in 1:(L-2)
        _two_site_update_forward!(state, H, E_left, E_right, i, dt/2, config, true)
    end
    
    # Edge Step (L-1)
    if L >= 2
        # Evolve 2-site by FULL dt. Split Left. NO backward evolution.
        _two_site_update_edge_hamiltonian!(state, H, E_left, E_right, L-1, dt, config)
    end
    
    # Backward Sweep (L-2 -> 1)
    # Python: Evolve RightSite back -dt/2, Merge, Evolve dt/2, Split Left
    for i in (L-2):-1:1
        _two_site_update_backward_precorrect!(state, H, E_left, E_right, i, dt/2, config)
    end
end

# 4. Circuit 2-Site (Forward Only)
function _tdvp_sweep_circuit_2site!(state, H, config)
    shift_orthogonality_center!(state, 1)
    L = state.length
    
    E_left, E_right = _init_envs(state, H)
    
    # Circuit Logic: 
    # Bulk dt=2.0 (split into +1.0, -1.0)
    # Edge dt=1.0 (just +1.0)
    
    # Forward Sweep (1 -> L-2)
    for i in 1:(L-2)
        # Evolve +1.0, Back -1.0
        _two_site_update_forward!(state, H, E_left, E_right, i, 1.0, config, true)
    end
    
    # Edge Step (L-1)
    # Evolve +1.0
    _two_site_update_edge_circuit!(state, H, E_left, E_right, L-1, 1.0, config)
    
    # No Backward
end

# --- Helpers for 2-Site ---

function _two_site_update_forward!(state, H, E_left, E_right, i, dt_step, config, evolve_back)
    A1 = state.tensors[i]
    A2 = state.tensors[i+1]
    @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
    
    W1 = H.tensors[i]
    W2 = H.tensors[i+1]
    @tensor W_merge[l, p1o, p1i, p2o, p2i, r] := W1[l, p1o, p1i, k] * W2[k, p2o, p2i, r]
    W_perm = permutedims(W_merge, (1, 2, 4, 3, 5, 6))
    l1, p1o, p1i, b1 = size(W1)
    l2, p2o, p2i, b2 = size(W2)
    W_group = reshape(W_perm, l1, p1o*p2o, p1i*p2i, b2)
    
    l_theta, p1, p2, r_theta = size(Theta)
    Theta_group = reshape(Theta, l_theta, p1*p2, r_theta)
    
    # Evolve Theta
    func_two(x) = project_site(x, E_left[i], E_right[i+2], W_group)
    Theta_new = expm_krylov(func_two, Theta_group, dt_step, 25)
    
    # Split (Move Center Right: Keep S with V)
    Theta_split = reshape(Theta_new, l_theta, p1, p2, r_theta)
    U, S, Vt, keep = split_mps_tensor_svd(Theta_split, l_theta, p1, p2, r_theta, config)
    
    # Assign Left (U) -> Left Canonical
    state.tensors[i] = reshape(U, l_theta, p1, keep)
    
    # Update Left Env for next site
    E_left[i+1] = update_left_environment(state.tensors[i], W1, E_left[i])
    
    # Form Right (S*V) -> Center
    A2_temp = reshape(Diagonal(S) * Vt, keep, p2, r_theta)
    
    if evolve_back
        func_back(x) = project_site(x, E_left[i+1], E_right[i+2], W2)
        A2_new = expm_krylov(func_back, A2_temp, -dt_step, 25)
        state.tensors[i+1] = A2_new
    else
        state.tensors[i+1] = A2_temp
    end
end

function _two_site_update_edge_hamiltonian!(state, H, E_left, E_right, i, dt_step, config)
    # Edge Step: Evolve by dt_step. Split Left (Center moves to L-1).
    
    A1 = state.tensors[i]
    A2 = state.tensors[i+1]
    @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
    
    W1 = H.tensors[i]
    W2 = H.tensors[i+1]
    @tensor W_merge[l, p1o, p1i, p2o, p2i, r] := W1[l, p1o, p1i, k] * W2[k, p2o, p2i, r]
    W_perm = permutedims(W_merge, (1, 2, 4, 3, 5, 6))
    l1, p1o, p1i, b1 = size(W1)
    l2, p2o, p2i, b2 = size(W2)
    W_group = reshape(W_perm, l1, p1o*p2o, p1i*p2i, b2)
    
    l_theta, p1, p2, r_theta = size(Theta)
    Theta_group = reshape(Theta, l_theta, p1*p2, r_theta)
    
    # Evolve Theta (Full dt)
    func_two(x) = project_site(x, E_left[i], E_right[i+2], W_group)
    Theta_new = expm_krylov(func_two, Theta_group, dt_step, 25)
    
    # Split Left (Move Center Left: Keep S with U)
    Theta_split = reshape(Theta_new, l_theta, p1, p2, r_theta)
    U, S, Vt, keep = split_mps_tensor_svd(Theta_split, l_theta, p1, p2, r_theta, config)
    
    # Assign Right (Vt) -> Right Canonical
    state.tensors[i+1] = reshape(Vt, keep, p2, r_theta)
    
    # Update Right Env
    E_right[i+1] = update_right_environment(state.tensors[i+1], W2, E_right[i+2])
    
    # Assign Left (U*S) -> Center
    state.tensors[i] = reshape(U * Diagonal(S), l_theta, p1, keep)
    
    # No Backward evolution at edge
end

function _two_site_update_backward_precorrect!(state, H, E_left, E_right, i, dt_step, config)
    # Python Backward Loop Logic:
    # 1. Evolve Right Site (i+1) by -dt_step (Pre-correction)
    # 2. Merge (i, i+1)
    # 3. Evolve Theta by +dt_step
    # 4. Split Left
    
    # 1. Pre-correct Right Site (state[i+1])
    W2 = H.tensors[i+1]
    # Note: state[i+1] is currently Right Canonical (from prev step split).
    # Is it okay to evolve it? Yes, we treat it as the "Center" of the previous bond that needs time adjustment.
    
    func_back(x) = project_site(x, E_left[i+1], E_right[i+2], W2)
    state.tensors[i+1] = expm_krylov(func_back, state.tensors[i+1], -dt_step, 25)
    
    # 2. Merge
    A1 = state.tensors[i]
    A2 = state.tensors[i+1]
    @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
    
    W1 = H.tensors[i]
    @tensor W_merge[l, p1o, p1i, p2o, p2i, r] := W1[l, p1o, p1i, k] * W2[k, p2o, p2i, r]
    W_perm = permutedims(W_merge, (1, 2, 4, 3, 5, 6))
    l1, p1o, p1i, b1 = size(W1)
    l2, p2o, p2i, b2 = size(W2)
    W_group = reshape(W_perm, l1, p1o*p2o, p1i*p2i, b2)
    
    l_theta, p1, p2, r_theta = size(Theta)
    Theta_group = reshape(Theta, l_theta, p1*p2, r_theta)
    
    # 3. Evolve Theta
    func_two(x) = project_site(x, E_left[i], E_right[i+2], W_group)
    Theta_new = expm_krylov(func_two, Theta_group, dt_step, 25)
    
    # 4. Split Left (Center moves to i)
    Theta_split = reshape(Theta_new, l_theta, p1, p2, r_theta)
    U, S, Vt, keep = split_mps_tensor_svd(Theta_split, l_theta, p1, p2, r_theta, config)
    
    # Assign Right (Vt) -> Right Canonical
    state.tensors[i+1] = reshape(Vt, keep, p2, r_theta)
    
    # Update Right Env
    E_right[i+1] = update_right_environment(state.tensors[i+1], W2, E_right[i+2])
    
    # Assign Left (U*S) -> Center
    state.tensors[i] = reshape(U * Diagonal(S), l_theta, p1, keep)
end

function _two_site_update_edge_circuit!(state, H, E_left, E_right, i, dt_step, config)
    # Circuit Edge: Evolve by dt_step. Split Right. No Back.
    # Note: Python splits "right" here!
    # "state.tensors[i], state.tensors[i+1] = split_mps_tensor(..., "right", ...)"
    # So Center stays at Right (i+1).
    
    A1 = state.tensors[i]
    A2 = state.tensors[i+1]
    @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
    
    W1 = H.tensors[i]
    W2 = H.tensors[i+1]
    @tensor W_merge[l, p1o, p1i, p2o, p2i, r] := W1[l, p1o, p1i, k] * W2[k, p2o, p2i, r]
    W_perm = permutedims(W_merge, (1, 2, 4, 3, 5, 6))
    l1, p1o, p1i, b1 = size(W1)
    l2, p2o, p2i, b2 = size(W2)
    W_group = reshape(W_perm, l1, p1o*p2o, p1i*p2i, b2)
    
    l_theta, p1, p2, r_theta = size(Theta)
    Theta_group = reshape(Theta, l_theta, p1*p2, r_theta)
    
    # Evolve Theta
    func_two(x) = project_site(x, E_left[i], E_right[i+2], W_group)
    Theta_new = expm_krylov(func_two, Theta_group, dt_step, 25)
    
    # Split Right (Center moves to i+1)
    Theta_split = reshape(Theta_new, l_theta, p1, p2, r_theta)
    U, S, Vt, keep = split_mps_tensor_svd(Theta_split, l_theta, p1, p2, r_theta, config)
    
    # Assign Left (U) -> Left Canonical
    state.tensors[i] = reshape(U, l_theta, p1, keep)
    
    # Update Left Env
    E_left[i+1] = update_left_environment(state.tensors[i], W1, E_left[i])
    
    # Assign Right (S*V) -> Center
    state.tensors[i+1] = reshape(Diagonal(S) * Vt, keep, p2, r_theta)
end

function _init_envs(state, H)
    L = state.length
    E_right = Vector{Array{ComplexF64, 3}}(undef, L+1)
    r_bond = size(state.tensors[L], 3)
    r_mpo = size(H.tensors[L], 4)
    E_right[L+1] = make_identity_env(r_bond, r_mpo)
    
    for i in L:-1:2
        E_right[i] = update_right_environment(state.tensors[i], H.tensors[i], E_right[i+1])
    end
    
    E_left = Vector{Array{ComplexF64, 3}}(undef, L+1)
    l_bond = size(state.tensors[1], 1)
    l_mpo = size(H.tensors[1], 1)
    E_left[1] = make_identity_env(l_bond, l_mpo)
    
    return E_left, E_right
end

end # module
