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
Wrapper around KrylovKit.exponentiate.
"""
function expm_krylov(A_func::Function, v::AbstractArray{T}, dt::Number, k::Int) where T
    norm_v = norm(v)
    if norm_v == 0
        return v
    end
        
    t_val = -1im * dt
    # Using KrylovKit
    # exponentiate(A, t, x0) -> exp(t*A) * x0
    val, info = exponentiate(A_func, t_val, v; tol=1e-10, krylovdim=k, maxiter=1, ishermitian=false)
    
    return val
end



# --- Environment Helpers ---

"""
    make_identity_env(dim::Int, mpo_dim::Int)

Create an identity environment tensor of shape (dim, mpo_dim, dim).
It is non-zero only for mpo index = 1 (assuming MPO boundary is 1).
"""
function make_identity_env(dim::Int, mpo_dim::Int)
    E = zeros(ComplexF64, dim, mpo_dim, dim)
    for i in 1:dim
        E[i, 1, i] = 1.0
    end
    return E
end

function update_left_environment(A, W, E_left)
    # A: (L, P, R) [Ket]
    # W: (L_w, P_out, P_in, R_w)
    # E_left: (Bra_L, MPO_L, Ket_L)
    
    # Contract E_left * A
    # E[bra_l, mpo_l, ket_l] * A[ket_l, p_in, ket_r]
    # -> T1[bra_l, mpo_l, p_in, ket_r]
    @tensor T1[bra_l, mpo_l, p_in, ket_r] := E_left[bra_l, mpo_l, k] * A[k, p_in, ket_r]
    
    # Contract T1 * W
    # T1[bra_l, mpo_l, p_in, ket_r] * W[mpo_l, p_out, p_in, mpo_r]
    # -> T2[bra_l, ket_r, p_out, mpo_r]
    @tensor T2[bra_l, ket_r, p_out, mpo_r] := T1[bra_l, k_ml, k_pin, ket_r] * W[k_ml, p_out, k_pin, mpo_r]
    
    # Contract T2 * conj(A) (Bra)
    # T2[bra_l, ket_r, p_out, mpo_r] * conj(A)[bra_l, p_out, bra_r]
    # -> E_next[bra_r, mpo_r, ket_r]
    @tensor E_next[bra_r, mpo_r, ket_r] := T2[k_bl, ket_r, k_pout, mpo_r] * conj(A[k_bl, k_pout, bra_r])
    
    return E_next
end

function update_right_environment(A, W, E_right)
    # A: (L, P, R) [Ket]
    # W: (L_w, P_out, P_in, R_w)
    # E_right: (Bra_R, MPO_R, Ket_R)
    
    # Contract A * E_right
    # A[ket_l, p_in, ket_r] * E[bra_r, mpo_r, ket_r]
    # -> T1[ket_l, p_in, bra_r, mpo_r]
    @tensor T1[ket_l, p_in, bra_r, mpo_r] := A[ket_l, p_in, k] * E_right[bra_r, mpo_r, k]
    
    # Contract W * T1
    # W[mpo_l, p_out, p_in, mpo_r] * T1[ket_l, p_in, bra_r, mpo_r]
    # -> T2[mpo_l, p_out, ket_l, bra_r]
    @tensor T2[mpo_l, p_out, ket_l, bra_r] := W[mpo_l, p_out, k_pin, k_mr] * T1[ket_l, k_pin, bra_r, k_mr]
    
    # Contract T2 * conj(A)
    # T2[mpo_l, p_out, ket_l, bra_r] * conj(A)[bra_l, p_out, bra_r]
    # -> E_next[bra_l, mpo_l, ket_l]
    @tensor E_next[bra_l, mpo_l, ket_l] := T2[mpo_l, k_pout, ket_l, k_br] * conj(A[bra_l, k_pout, k_br])
    
    return E_next
end

# --- Projectors (Matrix-Free Operators) ---

function project_site(A, L, R, W)
    # L: (Bra_L, MPO_L, Ket_L)
    # R: (Bra_R, MPO_R, Ket_R)
    # W: (MPO_L, P_out, P_in, MPO_R)
    # A: (Ket_L, P_in, Ket_R)
    
    # L * A
    @tensor T1[bra_l, mpo_l, p_in, ket_r] := L[bra_l, mpo_l, k] * A[k, p_in, ket_r]
    
    # T1 * W
    @tensor T2[bra_l, ket_r, p_out, mpo_r] := T1[bra_l, k_ml, k_pin, ket_r] * W[k_ml, p_out, k_pin, mpo_r]
    
    # T2 * R
    @tensor A_new[bra_l, p_out, bra_r] := T2[bra_l, k_kr, p_out, k_mr] * R[bra_r, k_mr, k_kr]
    
    return A_new
end

function project_bond(C, L, R)
    # C: (Ket_L, Ket_R)
    # L: (Bra_L, MPO, Ket_L)
    # R: (Bra_R, MPO, Ket_R)
    
    # L * C
    @tensor T1[bra_l, mpo, ket_r] := L[bra_l, mpo, k] * C[k, ket_r]
    
    # T1 * R
    @tensor C_new[bra_l, bra_r] := T1[bra_l, k_mpo, k_kr] * R[bra_r, k_mpo, k_kr]
    
    return C_new
end

# --- TDVP Sweeps ---

"""
    single_site_tdvp!(state, H, config)

Perform 1-site TDVP.
"""
function single_site_tdvp!(state::MPS, H::MPO, config::TimeEvolutionConfig)
    dt = config.dt
    L = state.length
    
    # Init Environments
    # Right Envs
    E_right = Vector{Array{ComplexF64, 3}}(undef, L+1)
    
    r_bond_dim = size(state.tensors[L], 3)
    r_mpo_dim = size(H.tensors[L], 4)
    E_right[L+1] = make_identity_env(r_bond_dim, r_mpo_dim)
    
    for i in L:-1:2
        E_right[i] = update_right_environment(state.tensors[i], H.tensors[i], E_right[i+1])
    end
    
    E_left = Vector{Array{ComplexF64, 3}}(undef, L+1)
    
    l_bond_dim = size(state.tensors[1], 1)
    l_mpo_dim = size(H.tensors[1], 1)
    E_left[1] = make_identity_env(l_bond_dim, l_mpo_dim)
    
    # Sweep Right (1 -> L)
    for i in 1:L
        # 1. Evolve Site Forward
        W = H.tensors[i]
        A = state.tensors[i]
        
        func_site(x) = project_site(x, E_left[i], E_right[i+1], W)
        A_new = expm_krylov(func_site, A, dt/2, 25) 
        
        state.tensors[i] = A_new
        
        if i < L
            # 2. Split / QR to move center right
            l, p, r = size(A_new)
            A_mat = reshape(A_new, l*p, r)
            F = qr(A_mat) # Julia's qr defaults to thin but returns full struct.
           
            

            # 2. Split / QR to move center right
            l, p, r = size(A_new)
            A_mat = reshape(A_new, l*p, r)
            F = qr(A_mat)
            
            # Extract Thin Q and R matching numpy.linalg.qr(mode='reduced')
            # K = min(l*p, r)
            # Q is (l*p, K)
            # R is (K, r)
            
            K = min(l*p, r)
            Q_thin = Matrix(F.Q)[:, 1:K]
            R_thin = Matrix(F.R)[1:K, :]
            
            # Update Site i
            state.tensors[i] = reshape(Q_thin, l, p, K)
            
            # Update Left Env
            E_left[i+1] = update_left_environment(state.tensors[i], W, E_left[i])
            
            # Evolve Bond Backward (-dt/2)
            func_bond(x) = project_bond(x, E_left[i+1], E_right[i+1]) # Bond is between i and i+1
            C_new = expm_krylov(func_bond, R_thin, -dt/2, 25)
            
            # Absorb C into next site (i+1)
            Next = state.tensors[i+1]
            # Next is (r, p_next, r_next). C_new is (K, r).
            # We contract C_new[k_new, k_old] * Next[k_old, p, r]
            @tensor Next_new[l, p, r] := C_new[l, k] * Next[k, p, r]
            state.tensors[i+1] = Next_new

        end
    end
    
    # Sweep Left (L -> 1)
    for i in L:-1:1
        # Evolve Site Forward (+dt/2)
        W = H.tensors[i]
        A = state.tensors[i]
        
        func_site(x) = project_site(x, E_left[i], E_right[i+1], W)
        A_new = expm_krylov(func_site, A, dt/2, 25)
        state.tensors[i] = A_new
        
        if i > 1
            # Split / LQ to move center left
            l, p, r = size(A_new)
            A_mat = reshape(A_new, l, p*r)
            F = lq(A_mat)
            
            # Extract Thin L and Q matching numpy logic (if we were doing QR on transpose)
            # LQ in Julia: L is (l, K), Q is (K, p*r). K = min(l, p*r).
            # This effectively preserves the bond dimension if it fits.
            
            K = min(l, p*r)
            L_thin = Matrix(F.L)[:, 1:K]
            Q_thin = Matrix(F.Q)[1:K, :]
            
            state.tensors[i] = reshape(Q_thin, K, p, r)
            
            # Update Right Env
            E_right[i] = update_right_environment(state.tensors[i], W, E_right[i+1])
            
            # Evolve Bond Backward (-dt/2)
            func_bond(x) = project_bond(x, E_left[i], E_right[i]) # Bond between i-1 and i
            C_new = expm_krylov(func_bond, L_thin, -dt/2, 25)
            
            # Absorb into i-1
            Prev = state.tensors[i-1]
            # Prev is (l_prev, p_prev, l). C_new is (l, K).
            # We contract Prev[..., l] * C_new[l, K] -> result (..., K)
            # Wait. C_new is (l, K)?
            # L_thin was (l, K).
            # So C_new is (l, K).
            # Prev is (l_prev, p_prev, l).
            # Result (l_prev, p_prev, K).
            @tensor Prev_new[l, p, r] := Prev[l, p, k] * C_new[k, r]
            state.tensors[i-1] = Prev_new
        end
    end
    
end


"""
    two_site_tdvp!(state, H, config)

Perform 2-site TDVP.
"""
function two_site_tdvp!(state::MPS, H::MPO, config::TimeEvolutionConfig)
    dt = config.dt
    L = state.length
    
    # Init Envs
    E_right = Vector{Array{ComplexF64, 3}}(undef, L+1)
    
    # E_right[L+1] boundary
    r_bond_dim = size(state.tensors[L], 3)
    r_mpo_dim = size(H.tensors[L], 4)
    E_right[L+1] = make_identity_env(r_bond_dim, r_mpo_dim)
    
    for i in L:-1:2
        E_right[i] = update_right_environment(state.tensors[i], H.tensors[i], E_right[i+1])
    end
    
    E_left = Vector{Array{ComplexF64, 3}}(undef, L+1)
    # E_left[1] boundary
    l_bond_dim = size(state.tensors[1], 1)
    l_mpo_dim = size(H.tensors[1], 1)
    E_left[1] = make_identity_env(l_bond_dim, l_mpo_dim)
    
    # Forward Sweep
    for i in 1:(L-1)
        # 1. Merge Sites A[i] and A[i+1]
        A1 = state.tensors[i]
        A2 = state.tensors[i+1]
        @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
        
        # 2. Merge MPO W[i] and W[i+1]
        W1 = H.tensors[i]
        W2 = H.tensors[i+1]
        # W1: (L, p1o, p1i, B)
        # W2: (B, p2o, p2i, R)
        # Result: (L, p1o, p1i, p2o, p2i, R)
        @tensor W_merge[l, p1o, p1i, p2o, p2i, r] := W1[l, p1o, p1i, k] * W2[k, p2o, p2i, r]
        
        # Reshape Theta -> (L, (p1,p2), R)
        l_theta, p1, p2, r_theta = size(Theta)
        Theta_group = reshape(Theta, l_theta, p1*p2, r_theta)
        
        # Reshape W_merge -> (L, (p1o, p2o), (p1i, p2i), R)
        # Requires permuting (l, p1o, p1i, p2o, p2i, r) -> (l, p1o, p2o, p1i, p2i, r)
        W_perm = permutedims(W_merge, (1, 2, 4, 3, 5, 6))
        W_group = reshape(W_perm, size(W1, 1), p1*p2, p1*p2, size(W2, 4))
        
        # 3. Evolve Theta (+dt/2)
        func_two_site(x) = project_site(x, E_left[i], E_right[i+2], W_group)
        Theta_new = expm_krylov(func_two_site, Theta_group, dt/2, 25)
        
        # 4. Split Theta (SVD) -> A1_new, S, A2_new
        # Reshape back to (L, p1, p2, R)
        Theta_split = reshape(Theta_new, l_theta, p1, p2, r_theta)
        
        # SVD
        # Group (L, p1) and (p2, R)
        Mat = reshape(Theta_split, l_theta*p1, p2*r_theta)
        F = svd(Mat)
        
        # Truncation
        truncated_weight = 0.0
        keep_rank = length(F.S)
        threshold = config.truncation_threshold
        
        # Calculate keep rank based on threshold
        for k in length(F.S):-1:1
            w = F.S[k]^2
            if truncated_weight + w > threshold
                keep_rank = k
                break
            end
            truncated_weight += w
            keep_rank = k - 1
        end
        
        max_D = config.max_bond_dim
        keep = clamp(keep_rank, 1, max_D)
        
        U = F.U[:, 1:keep]
        S = F.S[1:keep]
        Vt = F.Vt[1:keep, :]
        
        # 5. Assign A1_new (left canonical U)
        state.tensors[i] = reshape(U, l_theta, p1, keep)
        
        # Update Left Env (for i+1)
        E_left[i+1] = update_left_environment(state.tensors[i], W1, E_left[i])
        
        # 6. Form A2 (Right) = S * Vt
        # A2_temp is effectively the "Bond-Center" tensor for the next step
        A2_temp = reshape(Diagonal(S) * Vt, keep, p2, r_theta)
        
        # Evolve A2 Backward (-dt/2) ONLY if not at the last bond
        if i < L - 1
            func_site_back(x) = project_site(x, E_left[i+1], E_right[i+2], W2)
            A2_new = expm_krylov(func_site_back, A2_temp, -dt/2, 25)
            state.tensors[i+1] = A2_new
        else
            # At the edge, A2 stays at t+dt/2
            state.tensors[i+1] = A2_temp
        end
    end
    
    # Backward Sweep (L-1 -> 1)
    
    for i in (L-1):-1:1
        # 1. Merge
        A1 = state.tensors[i]
        A2 = state.tensors[i+1]
        @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
        
        W1 = H.tensors[i]
        W2 = H.tensors[i+1]
        @tensor W_merge[l, p1o, p1i, p2o, p2i, r] := W1[l, p1o, p1i, k] * W2[k, p2o, p2i, r]
        l_theta, p1, p2, r_theta = size(Theta)
        Theta_group = reshape(Theta, l_theta, p1*p2, r_theta)
        W_perm = permutedims(W_merge, (1, 2, 4, 3, 5, 6))
        W_group = reshape(W_perm, size(W1, 1), p1*p2, p1*p2, size(W2, 4))
        
        # Evolve (+dt/2)
        func_two_site(x) = project_site(x, E_left[i], E_right[i+2], W_group)
        Theta_new = expm_krylov(func_two_site, Theta_group, dt/2, 25)
        
        # Split (SVD)
        Theta_split = reshape(Theta_new, l_theta, p1, p2, r_theta)
        Mat = reshape(Theta_split, l_theta*p1, p2*r_theta)
        F = svd(Mat)
        
        # Truncation
        truncated_weight = 0.0
        keep_rank = length(F.S)
        threshold = config.truncation_threshold
        
        for k in length(F.S):-1:1
            w = F.S[k]^2
            if truncated_weight + w > threshold
                keep_rank = k
                break
            end
            truncated_weight += w
            keep_rank = k - 1
        end
        
        max_D = config.max_bond_dim
        keep = clamp(keep_rank, 1, max_D)
        
        U = F.U[:, 1:keep]
        S = F.S[1:keep]
        Vt = F.Vt[1:keep, :]
        
        # Assign A2 (Right Canonical Vt)
        state.tensors[i+1] = reshape(Vt, keep, p2, r_theta)
        
        # Update Right Env (for i)
        E_right[i+1] = update_right_environment(state.tensors[i+1], W2, E_right[i+2])
        
        # Form A1 = U * S
        A1_temp = reshape(U * Diagonal(S), l_theta, p1, keep)
        
        # Evolve A1 Backward (-dt/2) ONLY if not at the first bond
        if i > 1
            func_site_back(x) = project_site(x, E_left[i], E_right[i+1], W1)
            A1_new = expm_krylov(func_site_back, A1_temp, -dt/2, 25)
            state.tensors[i] = A1_new
        else
             # At the edge, A1 stays at t+dt (dt/2 total)
             state.tensors[i] = A1_temp
        end
    end
    
end

end # module

