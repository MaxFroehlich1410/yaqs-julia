module MPOModule

using TensorOperations
using LinearAlgebra
using Base.Threads
using ..MPSModule
using ..GateLibrary

export MPO, contract_mpo_mps, expect_mpo, contract_mpo_mpo, init_ising, init_general_hamiltonian
export orthogonalize!, truncate!
export apply_zipup!
export apply_variational!, mpo_from_two_qubit_gate_matrix

abstract type AbstractTensorNetwork end

"""
    MPO{T} <: AbstractTensorNetwork

Matrix Product Operator (MPO) using Column-Major memory layout.

# Layout
Tensors are stored with indices: `(Left_Bond, Phys_Out, Phys_In, Right_Bond)`.
- `Left_Bond`: Connection to left neighbor MPO tensor.
- `Phys_Out`: Outgoing physical index (upper leg, bra).
- `Phys_In`: Incoming physical index (lower leg, ket).
- `Right_Bond`: Connection to right neighbor MPO tensor.

# Fields
- `tensors::Vector{Array{T, 4}}`: The MPO tensors.
- `phys_dims::Vector{Int}`: Physical dimension of each site.
- `length::Int`: Number of sites.
- `orth_center::Int`: Orthogonality center (1-based). 0 if unknown/mixed.
"""
mutable struct MPO{T<:Number} <: AbstractTensorNetwork
    tensors::Vector{Array{T, 4}}
    phys_dims::Vector{Int}
    length::Int
    orth_center::Int

    function MPO{T}(length::Int, tensors::Vector{Array{T, 4}}, phys_dims::Vector{Int}, orth_center::Int=0) where T
        new{T}(tensors, phys_dims, length, orth_center)
    end
end

# Outer constructor
function MPO(length::Int, tensors::Vector{Array{T, 4}}, phys_dims::Vector{Int}, orth_center::Int=0) where T
    return MPO{T}(length, tensors, phys_dims, orth_center)
end

# --- Constructors ---

"""
    MPO(length::Int; identity=false, physical_dimensions=nothing)

Initialize an MPO. 
"""
function MPO(length::Int; 
             identity::Bool=false,
             physical_dimensions::Union{Vector{Int}, Int, Nothing}=nothing)

    # 1. Handle Physical Dimensions
    phys_dims = if isnothing(physical_dimensions)
        fill(2, length)
    elseif isa(physical_dimensions, Int)
        fill(physical_dimensions, length)
    else
        physical_dimensions
    end
    @assert length == Base.length(phys_dims)

    T = ComplexF64 # Default type
    tensors = Vector{Array{T, 4}}(undef, length)

    if identity
        for i in 1:length
            d = phys_dims[i]
            # Identity MPO Tensor: (Left=1, Phys_Out=d, Phys_In=d, Right=1)
            id_mat = Matrix{T}(I, d, d)
            tensors[i] = reshape(id_mat, 1, d, d, 1)
        end
        return MPO(length, tensors, phys_dims, 0)
    else
        for i in 1:length
            d = phys_dims[i]
            tensors[i] = zeros(T, 1, d, d, 1)
        end
        return MPO(length, tensors, phys_dims, 0)
    end
end

"""
    init_ising(length::Int, J, g) -> MPO
"""
function init_ising(length::Int, J::Union{Real, Vector{<:Real}}, g::Union{Real, Vector{<:Real}})
    # Operators
    X_op = Matrix(matrix(XGate()))
    Z_op = Matrix(matrix(ZGate()))
    I_op = Matrix(I, 2, 2)
    Zero_op = zeros(ComplexF64, 2, 2)

    tensors = Vector{Array{ComplexF64, 4}}(undef, length)
    phys_dims = fill(2, length)

    get_J(i) = isa(J, Vector) ? (i <= Base.length(J) ? J[i] : 0.0) : J
    get_g(i) = isa(g, Vector) ? g[i] : g
    
    function block_to_tensor(W_block)
        rows, cols = size(W_block) # Left, Right bond dims
        T = zeros(ComplexF64, rows, 2, 2, cols)
        for r in 1:rows, c in 1:cols
            T[r, :, :, c] .= W_block[r, c]
        end
        return T
    end

    if length == 1
         val_g = get_g(1)
         tensors[1] = reshape(-val_g * X_op, 1, 2, 2, 1)
    else
        # Site 1
        val_J = get_J(1)
        val_g = get_g(1)
        
        # W_left = [I, -J*Z, -g*X] (1x3)
        W_left = Matrix{Matrix{ComplexF64}}(undef, 1, 3)
        W_left[1, 1] = I_op
        W_left[1, 2] = -val_J * Z_op
        W_left[1, 3] = -val_g * X_op
        
        tensors[1] = block_to_tensor(W_left)
        
        # Bulk
        for i in 2:(length-1)
            val_Ji = get_J(i)
            val_gi = get_g(i)
            
            # W_bulk 3x3
            W_bulk = Matrix{Matrix{ComplexF64}}(undef, 3, 3)
            # Fill with Zeros first
            for r in 1:3, c in 1:3
                W_bulk[r,c] = Zero_op
            end
            
            W_bulk[1, 1] = I_op
            W_bulk[1, 2] = -val_Ji * Z_op
            W_bulk[1, 3] = -val_gi * X_op
            W_bulk[2, 3] = Z_op
            W_bulk[3, 3] = I_op
            
            tensors[i] = block_to_tensor(W_bulk)
        end
        
        # Site L
        val_gL = get_g(length)
        # W_right = [-gL*X; Z; I] (3x1)
        W_right = Matrix{Matrix{ComplexF64}}(undef, 3, 1)
        W_right[1, 1] = -val_gL * X_op
        W_right[2, 1] = Z_op
        W_right[3, 1] = I_op
        
        tensors[length] = block_to_tensor(W_right)
    end

    return MPO(length, tensors, phys_dims, 0)
end


"""
    init_general_hamiltonian(length::Int, Jxx, Jyy, Jzz, hx, hy, hz) -> MPO

Construct MPO for general Hamiltonian:
H = sum_i (Jxx_i X_i X_{i+1} + Jyy_i Y_i Y_{i+1} + Jzz_i Z_i Z_{i+1})
  + sum_i (hx_i X_i + hy_i Y_i + hz_i Z_i)

Parameters can be Vectors (site-dependent) or Scalars (uniform).
"""
function init_general_hamiltonian(length::Int, Jxx, Jyy, Jzz, hx, hy, hz)
    X_op = Matrix(matrix(XGate()))
    Y_op = Matrix(matrix(YGate()))
    Z_op = Matrix(matrix(ZGate()))
    I_op = Matrix(I, 2, 2)
    Zero_op = zeros(ComplexF64, 2, 2)
    
    tensors = Vector{Array{ComplexF64, 4}}(undef, length)
    phys_dims = fill(2, length)

    # Helper to get parameter at site i
    get_param(p, i) = isa(p, Vector) ? (i <= Base.length(p) ? p[i] : 0.0) : p

    # Block to tensor helper
    function block_to_tensor(W_block)
        rows, cols = size(W_block)
        T = zeros(ComplexF64, rows, 2, 2, cols)
        for r in 1:rows, c in 1:cols
            T[r, :, :, c] .= W_block[r, c]
        end
        return T
    end

    for i in 1:length
        # Current site params
        # Couplings J are for bond (i, i+1). Only valid for i < L.
        val_Jxx = (i < length) ? get_param(Jxx, i) : 0.0
        val_Jyy = (i < length) ? get_param(Jyy, i) : 0.0
        val_Jzz = (i < length) ? get_param(Jzz, i) : 0.0
        
        # Fields h are for site i.
        val_hx = get_param(hx, i)
        val_hy = get_param(hy, i)
        val_hz = get_param(hz, i)
        
        # Build local field term
        loc_field = val_hx * X_op + val_hy * Y_op + val_hz * Z_op

        # Matrix size 5x5
        # 1: Identity (from Left)
        # 2: X (waiting for X)
        # 3: Y (waiting for Y)
        # 4: Z (waiting for Z)
        # 5: Identity (to Right)
        
        W = Matrix{Matrix{ComplexF64}}(undef, 5, 5)
        for r in 1:5, c in 1:5; W[r,c] = Zero_op; end
        
        # Row 1 (Input from Identity)
        W[1, 1] = I_op
        W[1, 2] = val_Jxx * X_op
        W[1, 3] = val_Jyy * Y_op
        W[1, 4] = val_Jzz * Z_op
        W[1, 5] = loc_field
        
        # Completing interactions
        W[2, 5] = X_op
        W[3, 5] = Y_op
        W[4, 5] = Z_op
        
        # Identity passthrough
        W[5, 5] = I_op
        
        # Extract correct block based on boundary
        if i == 1
            # First site: Row 1 only
            # But wait, if L=1, it's just [1, 5] -> loc_field.
            if length == 1
                 tensors[i] = reshape(loc_field, 1, 2, 2, 1)
            else
                 # Top row (1, :)
                 W_row = W[1:1, :]
                 tensors[i] = block_to_tensor(W_row)
            end
        elseif i == length
            # Last site: Column 5 only
            W_col = W[:, 5:5]
            tensors[i] = block_to_tensor(W_col)
        else
            # Bulk
            tensors[i] = block_to_tensor(W)
        end
    end

    return MPO(length, tensors, phys_dims, 0)
end


# --- Optimization & Compression ---

"""
    orthogonalize!(mpo::MPO, new_center::Int)

Shift the orthogonality center of the MPO to `new_center`.
Uses QR/LQ decomposition on the "effective" physical dimension (d_out * d_in).
"""
function orthogonalize!(mpo::MPO{T}, new_center::Int) where T
    # If current center is unknown (0), we must assume nothing and start sweeping.
    # To be safe, if 0, we usually sweep from 1->new_center or L->new_center fully.
    # Here we assume we want to establish a center. 
    # Strategy: Sweep 1 -> new_center (Left Canonicalize 1..nc-1)
    # Then Sweep L -> new_center (Right Canonicalize nc+1..L)
    # This guarantees the form regardless of initial state.
    
    # 1. Left Canonicalize: 1 -> new_center - 1
    for i in 1:(new_center - 1)
        A = mpo.tensors[i] # (L, do, di, R)
        L, dout, din, R = size(A)
        
        # Reshape to (L * dout * din, R)
        # Treat (dout, din) as one thick physical index
        Mat = reshape(A, L * dout * din, R)
        
        F = qr(Mat)
        Q = Matrix(F.Q)
        R_mat = Matrix(F.R)
        
        r_new = size(R_mat, 1)
        
        mpo.tensors[i] = reshape(Q, L, dout, din, r_new)
        
        # Absorb R into next
        Next = mpo.tensors[i+1] # (R, do', di', R')
        do_next, di_next, R_next = size(Next, 2), size(Next, 3), size(Next, 4)
        
        Next_mat = reshape(Next, R, do_next * di_next * R_next)
        New_Next = R_mat * Next_mat
        mpo.tensors[i+1] = reshape(New_Next, r_new, do_next, di_next, R_next)
    end
    
    # 2. Right Canonicalize: L -> new_center + 1
    for i in mpo.length:-1:(new_center + 1)
        A = mpo.tensors[i]
        L, dout, din, R = size(A)
        
        # Reshape to (L, dout * din * R)
        Mat = reshape(A, L, dout * din * R)
        
        F = lq(Mat)
        L_mat = Matrix(F.L)
        Q = Matrix(F.Q)
        
        r_new = size(L_mat, 2)
        
        mpo.tensors[i] = reshape(Q, r_new, dout, din, R)
        
        # Absorb L into prev
        Prev = mpo.tensors[i-1]
        L_prev, do_prev, di_prev = size(Prev, 1), size(Prev, 2), size(Prev, 3)
        
        Prev_mat = reshape(Prev, L_prev * do_prev * di_prev, L)
        New_Prev = Prev_mat * L_mat
        mpo.tensors[i-1] = reshape(New_Prev, L_prev, do_prev, di_prev, r_new)
    end
    
    mpo.orth_center = new_center
end

"""
    truncate!(mpo::MPO; threshold=1e-12, max_bond_dim=nothing)

Compress the MPO by truncating small singular values.
"""
function truncate!(mpo::MPO{T}; threshold::Float64=1e-12, max_bond_dim::Union{Int, Nothing}=nothing) where T
    # Sweep 1 -> L-1
    # First ensure we are at 1
    orthogonalize!(mpo, 1)
    
    total_err = 0.0
    
    for i in 1:(mpo.length - 1)
        A = mpo.tensors[i]   # (L, do, di, k)
        B = mpo.tensors[i+1] # (k, do', di', R)
        
        L, do1, di1, k = size(A)
        _, do2, di2, R = size(B)
        
        # Reshape A: (L*do1*di1, k)
        Amat = reshape(A, L*do1*di1, k)
        # Reshape B: (k, do2*di2*R)
        Bmat = reshape(B, k, do2*di2*R)
        
        Theta = Amat * Bmat # (Left_Phys, Right_Phys)
        
        F = svd(Theta)
        U, S, Vt = F.U, F.S, F.Vt
        
        norm_sq = dot(S, S)
        keep = length(S)
        
        # Threshold truncation
        csum = 0.0
        for idx in length(S):-1:1
            csum += S[idx]^2
            if csum > threshold * norm_sq
                keep = idx
                break
            end
            if idx == 1; keep = 1; end
        end
        
        if !isnothing(max_bond_dim)
            keep = min(keep, max_bond_dim)
        end
        
        # Update error
        total_err += sum(S[keep+1:end].^2)
        
        U_trunc = U[:, 1:keep]
        S_trunc = S[1:keep]
        Vt_trunc = Vt[1:keep, :]
        
        # Update A (Left Canonical)
        mpo.tensors[i] = reshape(U_trunc, L, do1, di1, keep)
        
        # Update B (Center)
        SV = Diagonal(S_trunc) * Vt_trunc
        mpo.tensors[i+1] = reshape(SV, keep, do2, di2, R)
    end
    
    mpo.orth_center = mpo.length
    return total_err
end


# --- Application (MPO x MPS) ---

"""
    contract_mpo_mps(w::MPO, psi::MPS) -> MPS

Apply MPO `w` to MPS `psi`.
"""
function contract_mpo_mps(w::MPO, psi::MPS)
    @assert w.length == psi.length
    
    L = w.length
    new_tensors = Vector{Array{ComplexF64, 3}}(undef, L)
    
    # Threaded if L is large? No, allocation inside threads is tricky.
    for i in 1:L
        W = w.tensors[i] # (L_w, P_out, P_in, R_w)
        A = psi.tensors[i] # (L_a, P_in, R_a)
        
        # C[l_w, l_a, p_out, r_w, r_a]
        @tensor C[l_w, l_a, p_out, r_w, r_a] := W[l_w, p_out, k, r_w] * A[l_a, k, r_a]
        
        l_w, l_a, p_out, r_w, r_a = size(C)
        new_left = l_w * l_a
        new_right = r_w * r_a
        
        # Merge bonds: (l_w, l_a) -> L', (r_w, r_a) -> R'
        new_tensor = reshape(C, new_left, p_out, new_right)
        new_tensors[i] = new_tensor
    end
    
    # Resulting MPS usually has high bond dimension.
    # Center is undefined/mixed (0).
    return MPS(L, new_tensors, psi.phys_dims, 0)
end

# --- Application (MPO x MPS) with Zip-up compression ---

@inline function _zipup_keep(S::AbstractVector{<:Real}, chi_max::Int, svd_min::Real)
    # TenPy's truncation supports multiple constraints. For zip-up we primarily need:
    # - keep at most `chi_max`
    # - drop singular values below `svd_min` (absolute cutoff)
    k = min(length(S), chi_max)
    if svd_min > 0
        # Find the last index with S[idx] >= svd_min
        last_good = 0
        @inbounds for i in 1:k
            if S[i] >= svd_min
                last_good = i
            end
        end
        k = max(last_good, 1)
    end
    return k
end

"""
    apply_zipup!(psi::MPS, W::MPO; chi_max, svd_min=0.0, m_temp=2, trunc_weight=1.0) -> trunc_err

Apply an MPO `W` to an MPS `psi` **in-place** using the *zip-up* method
(Stoudenmire & White, 2010; TenPy `MPO.apply_zipup`).

This performs a left-to-right sweep, contracting each site tensor with the MPO and
compressing immediately via an SVD. Intermediate SVDs use **relaxed truncation**
parameters:

- `chi_max_zip = m_temp * chi_max`
- `svd_min_zip = trunc_weight * svd_min`

The resulting MPS is (approximately) left-canonical with `psi.orth_center = psi.length`.

Notes:
- Only **finite** boundary conditions are supported (requires `size(W.tensors[1],1)==1`
  and `size(W.tensors[end],4)==1`).
- This implementation is optimized for the repo’s tensor layouts:
  - MPS: `(χL, d, χR)`
  - MPO: `(Dl, d_out, d_in, Dr)`

Returns `trunc_err` as the discarded weight `sum_{i>χ} s_i^2` accumulated across SVDs.
"""
function apply_zipup!(psi::MPSModule.MPS{T},
                      W::MPO{T};
                      chi_max::Int,
                      svd_min::Real=0.0,
                      m_temp::Int=2,
                      trunc_weight::Real=1.0) where {T<:Number}
    @assert psi.length == W.length "MPS and MPO lengths must match"
    L = psi.length
    @assert L >= 1
    @assert size(W.tensors[1], 1) == 1 "apply_zipup! currently supports finite MPOs (left boundary Dl=1)."
    @assert size(W.tensors[end], 4) == 1 "apply_zipup! currently supports finite MPOs (right boundary Dr=1)."

    chi_relax = max(1, m_temp * chi_max)
    svd_min_relax = trunc_weight * svd_min

    trunc_err = 0.0

    if L == 1
        A = psi.tensors[1] # (1, d_in, 1)
        W1 = W.tensors[1]  # (1, d_out, d_in, 1)
        d_out = size(W1, 2)
        @views W1s = W1[1, :, :, 1]  # (d_out, d_in)
        @views A1s = A[1, :, 1]      # (d_in,)
        @tensor T1[pout] := W1s[pout, pin] * A1s[pin]
        psi.tensors[1] = reshape(Array(T1), 1, d_out, 1)
        psi.orth_center = 1
        return trunc_err
    end

    # Carry tensor: (χ_prev, Dl, χL_site) with Dl = MPO left bond of the current site
    # Initialized after the first SVD.
    carry = Array{T,3}(undef, 0, 0, 0)

    for i in 1:L
        A = psi.tensors[i] # (χL, d_in, χR)
        Wi = W.tensors[i]  # (Dl, d_out, d_in, Dr)

        Dl = size(Wi, 1)
        d_out = size(Wi, 2)
        d_in = size(Wi, 3)
        Dr = size(Wi, 4)
        @assert size(A, 2) == d_in "MPO/MPS physical dimensions mismatch at site $i"

        if i == 1
            # T4[χL(=1), pout, Dr, χR]
            @views Wis = Wi[1, :, :, :]  # (d_out, d_in, Dr)
            @tensor T4[aL, pout, rW, aR] := A[aL, pin, aR] * Wis[pout, pin, rW]
            Mat = reshape(Array(T4), size(A, 1) * d_out, Dr * size(A, 3))

            F = svd(Mat)
            keep = _zipup_keep(F.S, chi_relax, svd_min_relax)
            trunc_err += sum(abs2, @view(F.S[(keep + 1):end]))

            U = @view(F.U[:, 1:keep])
            S = @view(F.S[1:keep])
            Vt = @view(F.Vt[1:keep, :])

            psi.tensors[i] = reshape(Array(U), size(A, 1), d_out, keep)

            # carry := (diag(S) * Vt) reshaped to (keep, Dr, χR)
            Vt_full = Array(Vt)  # keep × (Dr*χR)
            carry = reshape(Vt_full, keep, Dr, size(A, 3))
            @views carry .*= reshape(S, keep, 1, 1)
        elseif i == L
            # Last site: contract carry with current tensors, right boundary Dr=1 and χR=1.
            @assert Dr == 1
            @assert size(A, 3) == 1

            χprev = size(carry, 1)
            @views Wis = Wi[:, :, :, 1]   # (Dl, d_out, d_in)
            @views As = A[:, :, 1]        # (χL, d_in)
            @tensor T2[cp, pout] := carry[cp, lW, aL] * As[aL, pin] * Wis[lW, pout, pin]
            Mat = reshape(Array(T2), χprev * d_out, 1)

            F = svd(Mat)
            keep = _zipup_keep(F.S, 1, svd_min_relax)  # boundary enforces keep=1
            trunc_err += sum(abs2, @view(F.S[(keep + 1):end]))

            U = @view(F.U[:, 1:keep])
            psi.tensors[i] = reshape(Array(U), χprev, d_out, 1)
        else
            χprev = size(carry, 1)
            @tensor T4[cp, pout, rW, aR] := carry[cp, lW, aL] * A[aL, pin, aR] * Wi[lW, pout, pin, rW]
            Mat = reshape(Array(T4), χprev * d_out, Dr * size(A, 3))

            F = svd(Mat)
            keep = _zipup_keep(F.S, chi_relax, svd_min_relax)
            trunc_err += sum(abs2, @view(F.S[(keep + 1):end]))

            U = @view(F.U[:, 1:keep])
            S = @view(F.S[1:keep])
            Vt = @view(F.Vt[1:keep, :])

            psi.tensors[i] = reshape(Array(U), χprev, d_out, keep)

            Vt_full = Array(Vt)
            carry = reshape(Vt_full, keep, Dr, size(A, 3))
            @views carry .*= reshape(S, keep, 1, 1)
        end
    end

    psi.orth_center = psi.length
    return trunc_err
end

# --- Application (MPO x MPS) with variational compression (TenPy-style) ---

@inline function _variational_keep(S::AbstractVector{<:Real}, chi_max::Int, trunc::Real, svd_min::Real)
    # Match the repo-wide convention: `trunc` is a relative discarded-weight tolerance (>=0).
    # We emulate the logic used in `Decompositions.two_site_svd` / `DigitalTJM.apply_local_gate_exact!`:
    # accumulate from the smallest singular values, and once the cumulative discarded weight
    # would exceed `trunc`, keep the current value (hence the actual discarded weight stays < trunc).
    kχ = min(length(S), chi_max)
    # absolute cutoff (drop tiny/zero singular values)
    if svd_min > 0
        last_good = 0
        @inbounds for i in 1:kχ
            if S[i] >= svd_min
                last_good = i
            end
        end
        kχ = max(last_good, 1)
    end

    if trunc <= 0
        return kχ
    end

    total_sq = sum(abs2, S)
    discarded_sq = 0.0
    keep_dim = kχ
    min_keep = 2
    @inbounds for k in length(S):-1:1
        discarded_sq += S[k]^2
        frac = (total_sq == 0.0) ? 0.0 : (discarded_sq / total_sq)
        if frac >= trunc
            keep_dim = max(k, min_keep)
            break
        end
    end
    keep_dim = min(keep_dim, kχ)
    keep_dim = max(keep_dim, 1)
    return keep_dim
end

@inline function _theta_overlap_diff(theta_new::AbstractArray{T,4}, theta_old::AbstractArray{T,4}) where {T<:Number}
    # Return 1 - |<new|old>|/(||new||*||old||), in [0, 1] (up to roundoff)
    num = dot(vec(theta_new), vec(theta_old))  # dot() conjugates the first argument for complex arrays
    den = norm(vec(theta_new)) * norm(vec(theta_old))
    if den == 0
        return 0.0
    end
    return 1.0 - abs(num) / den
end

@inline function _update_left_env(Lenv::Array{T,3},
                                 Abra::Array{T,3},
                                 Wi::Array{T,4},
                                 Aket::Array{T,3}) where {T<:Number}
    # Lenv[aL, wL, bL] -> Lnew[aR, wR, bR]
    @tensor Lnew[aR, wR, bR] := Lenv[aL, wL, bL] * conj(Abra[aL, pout, aR]) * Wi[wL, pout, pin, wR] * Aket[bL, pin, bR]
    return Array(Lnew)
end

@inline function _update_right_env(Rnext::Array{T,3},
                                  Abra::Array{T,3},
                                  Wi::Array{T,4},
                                  Aket::Array{T,3}) where {T<:Number}
    # Rnext[aR, wR, bR] -> Rcur[aL, wL, bL]
    @tensor Rcur[aL, wL, bL] := conj(Abra[aL, pout, aR]) * Wi[wL, pout, pin, wR] * Aket[bL, pin, bR] * Rnext[aR, wR, bR]
    return Array(Rcur)
end

function _build_right_envs(psi::MPSModule.MPS{T}, W::MPO{T}, psi0::MPSModule.MPS{T}) where {T<:Number}
    L = psi.length
    Renv = Vector{Array{T,3}}(undef, L + 1)
    Renv[L + 1] = fill(one(T), 1, 1, 1)
    for i in L:-1:1
        Renv[i] = _update_right_env(Renv[i + 1], psi.tensors[i], W.tensors[i], psi0.tensors[i])
    end
    return Renv
end

function _build_left_envs(psi::MPSModule.MPS{T}, W::MPO{T}, psi0::MPSModule.MPS{T}) where {T<:Number}
    L = psi.length
    Lenv = Vector{Array{T,3}}(undef, L + 1)
    Lenv[1] = fill(one(T), 1, 1, 1)
    for i in 1:L
        Lenv[i + 1] = _update_left_env(Lenv[i], psi.tensors[i], W.tensors[i], psi0.tensors[i])
    end
    return Lenv
end

"""
    apply_variational!(psi::MPS, W::MPO; chi_max, trunc=0.0, svd_min=eps(), min_sweeps=1, max_sweeps=4, tol_theta_diff=1e-12) -> (trunc_err, sweeps, last_theta_diff)

Apply an MPO `W` to an MPS `psi` **in-place** using a TenPy-style *variational MPO application*.

This finds an MPS `phi` (stored back into `psi`) which best approximates `W|psi0⟩`, where `psi0`
is the input state (captured internally once), by alternating least-squares sweeps with two-site
updates and SVD-based truncation.

Tensor layouts (repo convention):
- MPS: `(χL, d, χR)`
- MPO: `(Dl, d_out, d_in, Dr)`

Truncation:
- `chi_max`: hard cap on MPS bond dimension
- `trunc`: relative discarded-weight tolerance (>=0), matching the rest of this repo’s conventions
- `svd_min`: absolute cutoff for singular values (drop exact zeros / tiny values)
"""
function apply_variational!(psi::MPSModule.MPS{T},
                            W::MPO{T};
                            chi_max::Int,
                            trunc::Real=0.0,
                            svd_min::Real=eps(Float64),
                            min_sweeps::Int=1,
                            max_sweeps::Int=4,
                            tol_theta_diff::Real=1.0e-12) where {T<:Number}
    @assert psi.length == W.length "MPS and MPO lengths must match"
    @assert psi.length >= 1
    @assert trunc >= 0.0

    L = psi.length
    # Capture input state (ket) once, as in TenPy.
    psi0 = deepcopy(psi)

    # Handle L=1 directly
    if L == 1
        A = psi.tensors[1]
        W1 = W.tensors[1]
        d_out = size(W1, 2)
        @tensor T1[pout] := W1[1, pout, pin, 1] * A[1, pin, 1]
        psi.tensors[1] = reshape(Array(T1), 1, d_out, 1)
        psi.orth_center = 1
        MPSModule.normalize!(psi)
        return (0.0, 0, 0.0)
    end

    trunc_err_total = 0.0
    last_max_diff = Inf
    sweeps_done = 0

    for sweep in 1:max_sweeps
        sweeps_done = sweep
        max_diff = 0.0

        # --- Left-to-right sweep ---
        Renv = _build_right_envs(psi, W, psi0)             # environments for cuts i..L
        Lenv = fill(one(T), 1, 1, 1)                       # cut before site 1

        for i in 1:(L - 1)
            # old theta from current bra (for convergence diagnostics)
            A0b = psi.tensors[i]
            A1b = psi.tensors[i + 1]
            @tensor theta_old[aL, p0, p1, aR] := A0b[aL, p0, k] * A1b[k, p1, aR]
            theta_old_arr = Array(theta_old)

            Wi0 = W.tensors[i]
            Wi1 = W.tensors[i + 1]
            K0 = psi0.tensors[i]
            K1 = psi0.tensors[i + 1]
            Rcut = Renv[i + 2]

            @tensor theta[aL, p0o, p1o, aR2] :=
                Lenv[aL, wL, bL] *
                Wi0[wL, p0o, p0i, wM] * K0[bL, p0i, bM] *
                Wi1[wM, p1o, p1i, wR] * K1[bM, p1i, bR] *
                Rcut[aR2, wR, bR]
            theta_arr = Array(theta)

            χL = size(theta_arr, 1)
            d0 = size(theta_arr, 2)
            d1 = size(theta_arr, 3)
            χR = size(theta_arr, 4)
            M = reshape(theta_arr, χL * d0, d1 * χR)
            F = svd(M)
            keep = _variational_keep(F.S, chi_max, trunc, svd_min)
            trunc_err_total += sum(abs2, @view(F.S[(keep + 1):end]))

            U = @view(F.U[:, 1:keep])
            S = @view(F.S[1:keep])
            Vt = @view(F.Vt[1:keep, :])

            psi.tensors[i] = reshape(Array(U), χL, d0, keep)
            psi.tensors[i + 1] = reshape(Array(Diagonal(S) * Array(Vt)), keep, d1, χR)
            psi.orth_center = i + 1

            @tensor theta_new[aL, p0, p1, aR] := psi.tensors[i][aL, p0, k] * psi.tensors[i + 1][k, p1, aR]
            max_diff = max(max_diff, _theta_overlap_diff(Array(theta_new), theta_old_arr))

            # update Lenv to cut before site i+1 using UPDATED bra tensor at site i
            Lenv = _update_left_env(Lenv, psi.tensors[i], Wi0, K0)
        end

        # --- Right-to-left sweep ---
        Lenvs = _build_left_envs(psi, W, psi0)             # environments for cuts 1..i-1
        Rcut = fill(one(T), 1, 1, 1)                       # cut after site L

        for i in (L - 1):-1:1
            # old theta from current bra
            A0b = psi.tensors[i]
            A1b = psi.tensors[i + 1]
            @tensor theta_old[aL, p0, p1, aR] := A0b[aL, p0, k] * A1b[k, p1, aR]
            theta_old_arr = Array(theta_old)

            Wi0 = W.tensors[i]
            Wi1 = W.tensors[i + 1]
            K0 = psi0.tensors[i]
            K1 = psi0.tensors[i + 1]
            Lcut = Lenvs[i]

            @tensor theta[aL, p0o, p1o, aR2] :=
                Lcut[aL, wL, bL] *
                Wi0[wL, p0o, p0i, wM] * K0[bL, p0i, bM] *
                Wi1[wM, p1o, p1i, wR] * K1[bM, p1i, bR] *
                Rcut[aR2, wR, bR]
            theta_arr = Array(theta)

            χL = size(theta_arr, 1)
            d0 = size(theta_arr, 2)
            d1 = size(theta_arr, 3)
            χR = size(theta_arr, 4)
            M = reshape(theta_arr, χL * d0, d1 * χR)
            F = svd(M)
            keep = _variational_keep(F.S, chi_max, trunc, svd_min)
            trunc_err_total += sum(abs2, @view(F.S[(keep + 1):end]))

            U = @view(F.U[:, 1:keep])
            S = @view(F.S[1:keep])
            Vt = @view(F.Vt[1:keep, :])

            psi.tensors[i] = reshape(Array(U), χL, d0, keep)
            psi.tensors[i + 1] = reshape(Array(Diagonal(S) * Array(Vt)), keep, d1, χR)
            psi.orth_center = i

            @tensor theta_new[aL, p0, p1, aR] := psi.tensors[i][aL, p0, k] * psi.tensors[i + 1][k, p1, aR]
            max_diff = max(max_diff, _theta_overlap_diff(Array(theta_new), theta_old_arr))

            # update Rcut to include UPDATED bra tensor at site i+1 for next step (moving left)
            Rcut = _update_right_env(Rcut, psi.tensors[i + 1], Wi1, K1)
        end

        last_max_diff = max_diff
        if sweep >= min_sweeps && max_diff <= tol_theta_diff
            break
        end
    end

    MPSModule.normalize!(psi)
    return (trunc_err_total, sweeps_done, last_max_diff)
end

"""
    mpo_from_two_qubit_gate_matrix(U::AbstractMatrix{ComplexF64}, i::Int, j::Int, L::Int; d::Int=2) -> MPO

Build an MPO (length `L`) for a 2-site operator `U` acting on sites `i` and `j` (1-based),
identity elsewhere, using an operator-Schmidt (SVD) decomposition.

If `i > j`, the qubit order in `U` is swapped by conjugating with SWAP such that the returned MPO
still represents the operator acting on the originally requested ordered pair.
"""
function mpo_from_two_qubit_gate_matrix(U::AbstractMatrix{ComplexF64},
                                       i::Int,
                                       j::Int,
                                       L::Int;
                                       d::Int=2)
    @assert 1 ≤ i ≤ L && 1 ≤ j ≤ L && i != j
    @assert size(U, 1) == d^2 && size(U, 2) == d^2

    ii, jj = i, j
    Uuse = Matrix(U)
    if ii > jj
        # U' = SWAP * U * SWAP to flip qubit order
        swap = zeros(ComplexF64, d^2, d^2)
        # For qubits d=2 this is the usual swap; for general d it swaps |a,b> -> |b,a>.
        for a in 0:(d - 1), b in 0:(d - 1)
            in_idx = a * d + b + 1
            out_idx = b * d + a + 1
            swap[out_idx, in_idx] = 1.0
        end
        Uuse = swap * Uuse * swap
        ii, jj = jj, ii
    end
    @assert ii < jj

    # Convention: U_tensor[p1_out, p2_out, p1_in, p2_in]
    Uten = reshape(Uuse, d, d, d, d)
    # Matrix M[(p1_out,p1_in), (p2_out,p2_in)]
    X = permutedims(Uten, (1, 3, 2, 4))  # (p1_out,p1_in,p2_out,p2_in)
    M = reshape(X, d * d, d * d)
    F = svd(M)

    # Drop near-zero singular values (numerical rank)
    s = F.S
    r = 0
    @inbounds for k in 1:length(s)
        if s[k] > 1.0e-14
            r = k
        end
    end
    r = max(r, 1)

    tensors = Vector{Array{ComplexF64,4}}(undef, L)
    for sidx in 1:L
        tensors[sidx] = zeros(ComplexF64, 1, d, d, 1) # overwritten below
        @inbounds for p in 1:d
            tensors[sidx][1, p, p, 1] = 1.0
        end
    end

    Wi = zeros(ComplexF64, 1, d, d, r)
    @inbounds for α in 1:r
        Avec = F.U[:, α] * sqrt(s[α])
        Aα = reshape(Avec, d, d)
        Wi[1, :, :, α] .= Aα
    end
    tensors[ii] = Wi

    if jj > ii + 1
        Wmid = zeros(ComplexF64, r, d, d, r)
        @inbounds for α in 1:r, p in 1:d
            Wmid[α, p, p, α] = 1.0
        end
        for sidx in (ii + 1):(jj - 1)
            tensors[sidx] = Wmid
        end
    end

    Wj = zeros(ComplexF64, r, d, d, 1)
    @inbounds for α in 1:r
        Bvec = sqrt(s[α]) * F.Vt[α, :]
        Bα = reshape(Bvec, d, d)
        Wj[α, :, :, 1] .= Bα
    end
    tensors[jj] = Wj

    return MPO(L, tensors, fill(d, L), 0)
end

# --- Expectation Value ---

"""
    expect_mpo(w::MPO, psi::MPS) -> ComplexF64

Compute <ψ|W|ψ>.
"""
function expect_mpo(w::MPO, psi::MPS)
    @assert w.length == psi.length
    
    # E[bra_bond, mpo_bond, ket_bond]
    E = ones(ComplexF64, 1, 1, 1)
    
    for i in 1:w.length
        A = psi.tensors[i]       # (L_k, p, R_k)
        W = w.tensors[i]         # (L_w, po, pi, R_w)
        
        # println("Debug Expect i=$i. E size: ", size(E))
        
        # E * A -> T1
        # E: (lb, lw, lk)
        # A: (lk, pi, rk)
        # Sum lk.
        @tensor T1[lb, lw, pi, rk] := E[lb, lw, c] * A[c, pi, rk]
        
        # T1 * W -> T2
        # T1: (lb, lw, pi, rk)
        # W: (lw, po, pi, rw)
        # Sum lw, pi.
        @tensor T2[lb, rk, po, rw] := T1[lb, c_lw, c_pi, rk] * W[c_lw, po, c_pi, rw]
        
        # T2 * conj(A) -> E_new
        # T2: (lb, rk, po, rw)
        # A_conj: (lb, po, rb) (from conj(A[lb, po, rb]))
        # Sum lb, po.
        @tensor E_new[rb, rw, rk] := T2[c_lb, rk, c_po, rw] * conj(A[c_lb, c_po, rb])
        
        E = E_new
        # println("Debug Expect i=$i. E_new: ", E)
    end
    
    return E[1, 1, 1]
end

# --- MPO-MPO Multiplication & Addition ---

"""
    Base.:+(a::MPO, b::MPO) -> MPO

Exact addition of two MPOs by direct sum of bonds.
"""
function Base.:+(a::MPO{T}, b::MPO{T}) where T
    @assert a.length == b.length
    L = a.length
    
    new_tensors = Vector{Array{T, 4}}(undef, L)
    
    for i in 1:L
        A = a.tensors[i]
        B = b.tensors[i]
        
        la, po, pi, ra = size(A)
        lb, _, _, rb = size(B)
        
        # Block Diagonal Construction
        if i == 1
            # Row [A B]
            C = zeros(T, 1, po, pi, ra+rb)
            C[1, :, :, 1:ra] = A[1, :, :, :]
            C[1, :, :, ra+1:end] = B[1, :, :, :]
        elseif i == L
            # Col [A; B]
            C = zeros(T, la+lb, po, pi, 1)
            C[1:la, :, :, 1] = A[:, :, :, 1]
            C[la+1:end, :, :, 1] = B[:, :, :, 1]
        else
            # Diag [A 0; 0 B]
            C = zeros(T, la+lb, po, pi, ra+rb)
             C[1:la, :, :, 1:ra] = A
             C[la+1:end, :, :, ra+1:end] = B
        end
        
        new_tensors[i] = C
    end
    
    return MPO(L, new_tensors, a.phys_dims, 0)
end

function Base.:*(c::Number, mpo::MPO{T}) where T
    new_tensors = copy(mpo.tensors)
    new_tensors[1] = c .* new_tensors[1]
    return MPO(mpo.length, new_tensors, mpo.phys_dims, mpo.orth_center)
end

Base.:*(mpo::MPO, c::Number) = c * mpo

"""
    contract_mpo_mpo(a::MPO, b::MPO) -> MPO

Contract A * B (A acting on B).
"""
function contract_mpo_mpo(a::MPO, b::MPO)
    @assert a.length == b.length
    L = a.length
    new_tensors = Vector{Array{ComplexF64, 4}}(undef, L)
    
    for i in 1:L
        A = a.tensors[i] # (La, Po, Pi, Ra)
        B = b.tensors[i] # (Lb, Po', Pi', Rb)
        
        # Contract Pi of A with Po' of B
        @tensor C[la, lb, po, pi, ra, rb] := A[la, po, k, ra] * B[lb, k, pi, rb]
        
        la, lb, po, pi, ra, rb = size(C)
        new_left = la * lb
        new_right = ra * rb
        
        new_tensors[i] = reshape(C, new_left, po, pi, new_right)
    end
    
    return MPO(L, new_tensors, a.phys_dims, 0)
end

end # module
