module MPOModule

using TensorOperations
using LinearAlgebra
using Base.Threads
using ..MPSModule
using ..GateLibrary

export MPO, contract_mpo_mps, expect_mpo, contract_mpo_mpo, init_ising, init_general_hamiltonian
export orthogonalize!, truncate!

"""
Abstract supertype for tensor network containers.

This provides a common parent for MPS and MPO types used throughout the library.

Args:
    None

Returns:
    AbstractTensorNetwork: Abstract type for tensor network containers.
"""
abstract type AbstractTensorNetwork end

"""
Represent a matrix product operator with an optional orthogonality center.

This stores rank-4 tensors in `(Left_Bond, Phys_Out, Phys_In, Right_Bond)` layout and tracks an
optional orthogonality center for canonicalization. It is the core operator container used for
Hamiltonians and channel representations.

Args:
    tensors (Vector{Array{T,4}}): MPO tensors in `(Dl, d_out, d_in, Dr)` layout.
    phys_dims (Vector{Int}): Physical dimensions per site.
    length (Int): Number of sites in the chain.
    orth_center (Int): Orthogonality center index (1-based), or 0 if unknown.

Returns:
    MPO{T}: Matrix product operator container.
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

"""
Construct an MPO with type inference for the element type.

This outer constructor infers `T` from the provided tensors and delegates to the inner constructor.

Args:
    length (Int): Number of sites in the chain.
    tensors (Vector{Array{T,4}}): MPO tensors in `(Dl, d_out, d_in, Dr)` layout.
    phys_dims (Vector{Int}): Physical dimensions per site.
    orth_center (Int): Orthogonality center index (1-based), or 0 if unknown.

Returns:
    MPO{T}: Matrix product operator container.
"""
function MPO(length::Int, tensors::Vector{Array{T, 4}}, phys_dims::Vector{Int}, orth_center::Int=0) where T
    return MPO{T}(length, tensors, phys_dims, orth_center)
end

# --- Constructors ---

"""
Initialize an MPO, optionally as an identity operator.

This constructs an MPO with the specified physical dimensions, creating either identity tensors
or zero tensors depending on the `identity` flag.

Args:
    length (Int): Number of sites in the chain.
    identity (Bool): Whether to initialize as the identity MPO.
    physical_dimensions (Union{Vector{Int}, Int, Nothing}): Physical dimensions per site or scalar.

Returns:
    MPO: Initialized matrix product operator.

Raises:
    AssertionError: If the physical dimension length does not match `length`.
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
Construct a transverse-field Ising Hamiltonian MPO.

This builds an MPO for `H = -∑ J_i Z_i Z_{i+1} - ∑ g_i X_i` with scalar or site-dependent parameters.

Args:
    length (Int): Number of sites.
    J (Union{Real, Vector{Real}}): Ising coupling(s) per bond.
    g (Union{Real, Vector{Real}}): Transverse field(s) per site.

Returns:
    MPO: Hamiltonian MPO for the Ising model.
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
    
    """
    Convert a block matrix of operators into an MPO tensor.

    This expands a matrix of 2x2 operator blocks into a rank-4 MPO tensor with the corresponding
    left and right bond dimensions.

    Args:
        W_block: Matrix of operator blocks.

    Returns:
        Array{ComplexF64,4}: MPO tensor built from the block matrix.
    """
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
Construct an MPO for a general XYZ Hamiltonian with fields.

This builds an MPO for `H = ∑ (Jxx_i X_i X_{i+1} + Jyy_i Y_i Y_{i+1} + Jzz_i Z_i Z_{i+1}) + ∑ (hx_i X_i + hy_i Y_i + hz_i Z_i)`,
accepting scalar or site-dependent parameters.

Args:
    length (Int): Number of sites.
    Jxx: XX coupling(s) per bond.
    Jyy: YY coupling(s) per bond.
    Jzz: ZZ coupling(s) per bond.
    hx: X field(s) per site.
    hy: Y field(s) per site.
    hz: Z field(s) per site.

Returns:
    MPO: Hamiltonian MPO for the general model.
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
    """
    Convert a block matrix of operators into an MPO tensor.

    This expands a matrix of 2x2 operator blocks into a rank-4 MPO tensor with the corresponding
    left and right bond dimensions.

    Args:
        W_block: Matrix of operator blocks.

    Returns:
        Array{ComplexF64,4}: MPO tensor built from the block matrix.
    """
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
Shift the orthogonality center of an MPO.

This canonicalizes the MPO by sweeping with QR/LQ decompositions on the combined physical index
`(d_out * d_in)` so that the orthogonality center is positioned at `new_center`.

Args:
    mpo (MPO): MPO to orthogonalize in-place.
    new_center (Int): Target orthogonality center index.

Returns:
    Nothing: The MPO tensors and `orth_center` are updated in-place.
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
Truncate an MPO by compressing internal bonds.

This sweeps left-to-right, merges neighboring tensors, performs an SVD, and truncates singular
values based on the provided threshold and optional maximum bond dimension.

Args:
    mpo (MPO): MPO to truncate in-place.
    threshold (Float64): Truncation threshold on discarded weight.
    max_bond_dim (Union{Int, Nothing}): Optional maximum bond dimension to keep.

Returns:
    Float64: Total truncation error accumulated over the sweep.
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
Apply an MPO to an MPS and return the resulting MPS.

This contracts each MPO tensor with the corresponding MPS tensor, expanding the bond dimensions
accordingly. The resulting MPS has an undefined/mixed orthogonality center.

Args:
    w (MPO): Operator to apply.
    psi (MPS): State to transform.

Returns:
    MPS: Resulting state after applying the MPO.

Raises:
    AssertionError: If the MPO and MPS lengths do not match.
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

# --- Expectation Value ---

"""
Compute the expectation value of an MPO in an MPS state.

This contracts the bra and ket MPS with the MPO to evaluate ⟨ψ|W|ψ⟩ as a scalar.

Args:
    w (MPO): Operator MPO.
    psi (MPS): State MPS.

Returns:
    ComplexF64: Expectation value ⟨ψ|W|ψ⟩.

Raises:
    AssertionError: If the MPO and MPS lengths do not match.
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
Add two MPOs by direct-sum bond expansion.

This constructs an exact MPO representing `a + b` by block-diagonalizing the bond spaces at each
site, increasing the bond dimensions as needed.

Args:
    a (MPO): First MPO.
    b (MPO): Second MPO.

Returns:
    MPO: Sum of the two MPOs.

Raises:
    AssertionError: If the MPO lengths do not match.
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

"""
Scale an MPO by a scalar factor.

This multiplies the first tensor of the MPO by the scalar, which is equivalent to scaling the
entire operator.

Args:
    c (Number): Scalar factor.
    mpo (MPO): MPO to scale.

Returns:
    MPO: Scaled MPO.
"""
function Base.:*(c::Number, mpo::MPO{T}) where T
    new_tensors = copy(mpo.tensors)
    new_tensors[1] = c .* new_tensors[1]
    return MPO(mpo.length, new_tensors, mpo.phys_dims, mpo.orth_center)
end

"""
Scale an MPO by a scalar factor (right multiplication).

This forwards to the left-multiplication definition so `mpo * c` and `c * mpo` behave identically.

Args:
    mpo (MPO): MPO to scale.
    c (Number): Scalar factor.

Returns:
    MPO: Scaled MPO.
"""
Base.:*(mpo::MPO, c::Number) = c * mpo

"""
Contract two MPOs to form their product.

This contracts the output physical index of `a` with the input physical index of `b` at each site,
producing an MPO representing the operator product `a * b`.

Args:
    a (MPO): Left MPO operator.
    b (MPO): Right MPO operator.

Returns:
    MPO: MPO representing the product operator.

Raises:
    AssertionError: If the MPO lengths do not match.
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
