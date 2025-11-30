module Decompositions

using LinearAlgebra
using TensorOperations

export right_qr, left_qr, two_site_svd

"""
    right_qr(tensor::AbstractArray{T,3})

QR Decomposition shifting orthogonality center to the Right.
Input Layout: (Left, Phys, Right)
Operation: Splits (Left, Phys) from (Right).
Returns: (Q, R)
- Q: (Left, Phys, NewBond)
- R: (NewBond, Right)
"""
function right_qr(tensor::AbstractArray{T,3}) where T
    l, p, r = size(tensor)
    
    # Reshape to Matrix: (Left * Phys) x Right
    # Julia is Column Major: Left is fastest, then Phys.
    # reshape(T, l*p, r) groups (l, p) into rows.
    mat = reshape(tensor, l * p, r)
    
    Q_fact = qr(mat)
    Q = Matrix(Q_fact.Q)
    R = Matrix(Q_fact.R)
    
    new_bond = size(Q, 2)
    
    # Q back to tensor: (Left, Phys, NewBond)
    Q_tensor = reshape(Q, l, p, new_bond)
    
    return Q_tensor, R
end

"""
    left_qr(tensor::AbstractArray{T,3})

QR Decomposition shifting orthogonality center to the Left.
Input Layout: (Left, Phys, Right)
Operation: Splits (Left) from (Phys, Right).
Returns: (R, Q)
- R: (Left, NewBond)
- Q: (NewBond, Phys, Right)
"""
function left_qr(tensor::AbstractArray{T,3}) where T
    l, p, r = size(tensor)
    
    # Reshape to Matrix: Left x (Phys * Right)
    # Since Phys is middle index, we can't natively reshape (l) x (p,r) unless we permute.
    # Or we use LQ on (Left * Phys) x Right? No.
    
    # We need rows to be Left, cols to be (Phys, Right).
    # reshape(tensor, l, p*r) works!
    # Julia's reshape merges adjacent dimensions efficiently.
    mat = reshape(tensor, l, p * r)
    
    # LQ Decomposition: M = L * Q
    # Julia has `lq`.
    F = lq(mat)
    L_mat = Matrix(F.L)
    Q_mat = Matrix(F.Q)
    
    new_bond = size(Q_mat, 1)
    
    # Q_mat shape: (NewBond, Phys*Right)
    # Reshape Q to (NewBond, Phys, Right)
    Q_tensor = reshape(Q_mat, new_bond, p, r)
    
    return L_mat, Q_tensor
end

"""
    two_site_svd(A, B, threshold; max_bond_dim=nothing)

Perform SVD on two adjacent MPS tensors A and B to truncate bond dimension.
Layouts:
- A: (Left_A, Phys_A, Bond)
- B: (Bond, Phys_B, Right_B)

Returns updated A_new, B_new.
"""
function two_site_svd(A::AbstractArray{T,3}, B::AbstractArray{T,3}, threshold::Real; max_bond_dim::Union{Int, Nothing}=nothing) where T
    l_a, p_a, r_a = size(A)
    l_b, p_b, r_b = size(B)
    
    # A connects to B via r_a == l_b
    @assert r_a == l_b
    
    # Contract Theta: (Left_A, Phys_A, Phys_B, Right_B)
    # A[l, pa, k] * B[k, pb, r]
    @tensor theta[l, pa, pb, r] := A[l, pa, k] * B[k, pb, r]
    
    # SVD separation
    # Group (Left_A, Phys_A) as Row
    # Group (Phys_B, Right_B) as Col
    # Note: SVD will return U, S, V' such that M = U * S * V'
    # U columns correspond to Row indices.
    # V' rows correspond to Col indices.
    
    theta_mat = reshape(theta, l_a * p_a, p_b * r_b)
    
    F = svd(theta_mat)
    U, S, Vt = F.U, F.S, F.Vt
    
    # Truncation Logic
    # Sum of squared discarded singular values <= threshold
    
    # Python logic:
    # discard = 0.0
    # keep = len(s_vec)
    # min_keep = 2
    # for idx, s in enumerate(reversed(s_vec)):
    #     discard += s**2
    #     if discard >= threshold:
    #         keep = max(len(s_vec) - idx, min_keep)
    #         break
    
    discarded_sq = 0.0
    keep_dim = length(S)
    min_keep = 2  # Python uses 2 to prevent pathological dimension-1 truncation
    
    # Calculate truncation
    # Note: svd returns sorted singular values (descending)
    # Python: enumerate(reversed(s_vec)) gives idx=0 for smallest, idx=1 for second-smallest, etc.
    # Julia: k=length(S) for smallest, k=length(S)-1 for second-smallest, etc.
    # Mapping: Python idx corresponds to Julia k = length(S) - idx
    # When Python sets keep = len(s_vec) - idx, Julia sets keep_dim = k
    
    # Check from end (smallest to largest)
    for k in length(S):-1:1
        discarded_sq += S[k]^2
        if discarded_sq >= threshold
            # Python: keep = max(len(s_vec) - idx, min_keep)
            # Julia: keep_dim = max(k, min_keep) where k = len(S) - idx
            keep_dim = max(k, min_keep)
            break
        end
    end
    
    # Enforce max_bond_dim
    if !isnothing(max_bond_dim)
        keep_dim = min(keep_dim, max_bond_dim)
    end
    
    # Ensure at least min_keep?
    keep_dim = max(keep_dim, 1)
    
    # Truncate
    U_trunc = U[:, 1:keep_dim]
    S_trunc = S[1:keep_dim]
    Vt_trunc = Vt[1:keep_dim, :]
    
    # Reconstruct A_new: U_trunc -> (Left_A, Phys_A, NewBond)
    A_new = reshape(U_trunc, l_a, p_a, keep_dim)
    
    # Reconstruct B_new: S * Vt -> (NewBond, Phys_B, Right_B)
    SV = Diagonal(S_trunc) * Vt_trunc
    B_new = reshape(SV, keep_dim, p_b, r_b)
    
    return A_new, B_new
end

end # module

