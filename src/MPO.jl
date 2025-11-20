module MPOModule

using TensorOperations
using LinearAlgebra
using Base.Threads
using ..MPSModule
using ..GateLibrary

export MPO, contract_mpo_mps, expect_mpo, contract_mpo_mpo, init_ising, init_heisenberg

abstract type AbstractTensorNetwork end

"""
    MPO{T} <: AbstractTensorNetwork

Matrix Product Operator (MPO) using Column-Major memory layout.

# Layout
Tensors are stored with indices: `(Left_Bond, Phys_Out, Phys_In, Right_Bond)`.
- `Left_Bond`: Connection to left neighbor MPO tensor.
- `Phys_Out`: Outgoing physical index (upper leg).
- `Phys_In`: Incoming physical index (lower leg, connects to MPS).
- `Right_Bond`: Connection to right neighbor MPO tensor.

# Fields
- `tensors::Vector{Array{T, 4}}`: The MPO tensors.
- `phys_dims::Vector{Int}`: Physical dimension of each site.
- `length::Int`: Number of sites.
"""
mutable struct MPO{T<:Number} <: AbstractTensorNetwork
    tensors::Vector{Array{T, 4}}
    phys_dims::Vector{Int}
    length::Int

    function MPO(length::Int, tensors::Vector{Array{T, 4}}, phys_dims::Vector{Int}) where T
        new{T}(tensors, phys_dims, length)
    end
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
            # Element is 1 if Phys_Out == Phys_In, else 0.
            
            # Construct Identity Matrix for Physical indices
            # We want Id[p_out, p_in]
            id_mat = Matrix{T}(I, d, d)
            
            # Reshape to (1, d, d, 1)
            tensor = reshape(id_mat, 1, d, d, 1)
            tensors[i] = tensor
        end
    else
        # Default initialization (empty/zeros? or maybe just allocated)
        # Following MPS pattern, maybe zeros? But MPO usually requires specific construction.
        # Let's initialize to zeros if not identity.
        for i in 1:length
            d = phys_dims[i]
            tensors[i] = zeros(T, 1, d, d, 1)
        end
    end

    return MPO(length, tensors, phys_dims)
end

"""
    init_ising(length::Int, J::Union{Real, Vector{<:Real}}, g::Union{Real, Vector{<:Real}}) -> MPO

Initialize the Ising model MPO.
H = sum(-J Z_i Z_{i+1} - g X_i)
Supports site-dependent J and g.
If J is a vector, J[i] is the coupling between i and i+1.
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
    
    # Helper to convert Block Matrix of Matrices to (Left, P_out, P_in, Right)
    function block_to_tensor(W_block)
        rows, cols = size(W_block) # Left, Right bond dims
        T = zeros(ComplexF64, rows, 2, 2, cols)
        for r in 1:rows
            for c in 1:cols
                op = W_block[r, c]
                T[r, :, :, c] = op
            end
        end
        return T
    end

    if length == 1
         val_g = get_g(1)
         tensors[1] = reshape(-val_g * X_op, 1, 2, 2, 1)
    else
        # Left Boundary (Site 1)
        # [ I, -J1 Z, -g1 X ]
        val_J = get_J(1)
        val_g = get_g(1)
        
        W_left = Matrix{Matrix{ComplexF64}}(undef, 1, 3)
        W_left[1, 1] = I_op
        W_left[1, 2] = -val_J * Z_op
        W_left[1, 3] = -val_g * X_op
        tensors[1] = block_to_tensor(W_left)
        
        # Bulk (Sites 2 to L-1)
        for i in 2:(length-1)
            val_Ji = get_J(i)
            val_gi = get_g(i)
            
            W_bulk = Matrix{Matrix{ComplexF64}}(undef, 3, 3)
            W_bulk .= [Zero_op for _ in 1:3, _ in 1:3]
            
            W_bulk[1, 1] = I_op
            W_bulk[1, 2] = -val_Ji * Z_op
            W_bulk[1, 3] = -val_gi * X_op
            W_bulk[2, 3] = Z_op
            W_bulk[3, 3] = I_op
            
            tensors[i] = block_to_tensor(W_bulk)
        end
        
        # Right Boundary (Site L)
        # [ -gL X ]
        # [    Z  ]
        # [    I  ]
        val_gL = get_g(length)
        # J[L] is not used (open boundary)
        
        W_right = Matrix{Matrix{ComplexF64}}(undef, 3, 1)
        W_right[1, 1] = -val_gL * X_op
        W_right[2, 1] = Z_op
        W_right[3, 1] = I_op
        tensors[length] = block_to_tensor(W_right)
    end

    return MPO(length, tensors, phys_dims)
end

"""
    init_heisenberg(length::Int, Jx::Float64, Jy::Float64, Jz::Float64, h::Float64) -> MPO

Initialize the Heisenberg model MPO.
"""
function init_heisenberg(length::Int, Jx::Float64, Jy::Float64, Jz::Float64, h::Float64)
    X_op = Matrix(matrix(XGate()))
    Y_op = Matrix(matrix(YGate()))
    Z_op = Matrix(matrix(ZGate()))
    I_op = Matrix(I, 2, 2)
    Zero_op = zeros(ComplexF64, 2, 2)
    
    tensors = Vector{Array{ComplexF64, 4}}(undef, length)
    phys_dims = fill(2, length)

    # Bulk (5x5)
    # [ I  -Jx X  -Jy Y  -Jz Z  -h Z ]
    # [ 0    0      0      0      X  ]
    # [ 0    0      0      0      Y  ]
    # [ 0    0      0      0      Z  ]
    # [ 0    0      0      0      I  ]
    
    W_bulk = Matrix{Matrix{ComplexF64}}(undef, 5, 5)
    W_bulk .= [Zero_op for _ in 1:5, _ in 1:5]
    
    W_bulk[1, 1] = I_op
    W_bulk[1, 2] = -Jx * X_op
    W_bulk[1, 3] = -Jy * Y_op
    W_bulk[1, 4] = -Jz * Z_op
    W_bulk[1, 5] = -h * Z_op
    W_bulk[2, 5] = X_op
    W_bulk[3, 5] = Y_op
    W_bulk[4, 5] = Z_op
    W_bulk[5, 5] = I_op
    
    # Left (1x5)
    W_left = Matrix{Matrix{ComplexF64}}(undef, 1, 5)
    W_left[1, 1] = I_op
    W_left[1, 2] = -Jx * X_op
    W_left[1, 3] = -Jy * Y_op
    W_left[1, 4] = -Jz * Z_op
    W_left[1, 5] = -h * Z_op
    
    # Right (5x1)
    # [ 0, X, Y, Z, I ]^T
    W_right = Matrix{Matrix{ComplexF64}}(undef, 5, 1)
    W_right[1, 1] = Zero_op
    W_right[2, 1] = X_op
    W_right[3, 1] = Y_op
    W_right[4, 1] = Z_op
    W_right[5, 1] = I_op

    function block_to_tensor(W_block)
        rows, cols = size(W_block)
        T = zeros(ComplexF64, rows, 2, 2, cols)
        for r in 1:rows
            for c in 1:cols
                op = W_block[r, c]
                T[r, :, :, c] = op
            end
        end
        return T
    end

    if length == 1
        # Single site: -h Z  (Heisenberg usually sums interactions, only field remains)
        tensors[1] = reshape(-h * Z_op, 1, 2, 2, 1)
    else
        tensors[1] = block_to_tensor(W_left)
        for i in 2:(length-1)
            tensors[i] = block_to_tensor(W_bulk)
        end
        tensors[length] = block_to_tensor(W_right)
    end

    return MPO(length, tensors, phys_dims)
end


# --- Application (MPO x MPS) ---

"""
    contract_mpo_mps(w::MPO, psi::MPS) -> MPS

Apply MPO `w` to MPS `psi`, resulting in a new MPS (with increased bond dimension).
Output MPS is NOT truncated.
"""
function contract_mpo_mps(w::MPO, psi::MPS)
    @assert w.length == psi.length
    
    L = w.length
    new_tensors = Vector{Array{ComplexF64, 3}}(undef, L)
    
    for i in 1:L
        W = w.tensors[i] # (L_w, P_out, P_in, R_w)
        A = psi.tensors[i] # (L_a, P_in, R_a)
        
        # Contraction:
        # We contract W and A over the common physical index `P_in`.
        # Output indices: (L_w, L_a), P_out, (R_w, R_a)
        # We need to merge (L_w, L_a) -> New_Left and (R_w, R_a) -> New_Right
        
        # Tensor operations:
        # W[l_w, p_out, p_in, r_w]
        # A[l_a, p_in, r_a]
        # Result C[l_w, l_a, p_out, r_w, r_a]
        
        @tensor C[l_w, l_a, p_out, r_w, r_a] := W[l_w, p_out, k, r_w] * A[l_a, k, r_a]
        
        # Reshape to merge bonds
        l_w, l_a, p_out, r_w, r_a = size(C)
        new_left = l_w * l_a
        new_right = r_w * r_a
        
        # Reshape (L_w, L_a, P_out, R_w, R_a) -> (New_Left, P_out, New_Right)
        # Note: Reshape in Julia is Column-Major. 
        # (dim1, dim2, ...) -> first index changes fastest.
        # We want merged index `new_left` to combine `l_w` and `l_a`.
        # If we reshape directly, we get (l_w + l_a*... ). 
        # We just need to be consistent.
        # Merging (l_w, l_a) -> size l_w*l_a is standard reshaping of first two dims.
        
        new_tensor = reshape(C, new_left, p_out, new_right)
        new_tensors[i] = new_tensor
    end
    
    return MPS(L, new_tensors, psi.phys_dims)
end

# --- Expectation Value ---

"""
    expect_mpo(w::MPO, psi::MPS) -> ComplexF64

Compute <ψ|W|ψ>.
Efficiently contracts the network: <ψ| (W |ψ>).
Complexity: O(L * D^3 * d^2) roughly.
"""
function expect_mpo(w::MPO, psi::MPS)
    @assert w.length == psi.length
    
    # Initialize Environment E (Left Boundary)
    # Structure: (MPS_Bond, MPO_Bond, MPS_Bond') -> (Bra_Bond, MPO_Bond, Ket_Bond)
    # Actually, let's trace indices carefully.
    # We are computing <psi | W | psi>.
    # Left-to-Right sweep.
    # E_left connects:
    # 1. Conjugated MPS (Bra) Left Bond
    # 2. MPO Left Bond
    # 3. MPS (Ket) Left Bond
    
    # Initial E is 1x1x1 (Assuming bond dim 1 at boundaries)
    E = ones(ComplexF64, 1, 1, 1)
    
    for i in 1:w.length
        A = psi.tensors[i]       # (L_a, P, R_a)  [Ket]
        W = w.tensors[i]         # (L_w, P_out, P_in, R_w) [Operator]
                                 # Note: P_out connects to Bra, P_in connects to Ket.
        
        # A_conj (Bra): conj(A)  # (L_a_conj, P_bra, R_a_conj)
        
        # Contract E with Ket (A)
        # E indices: (l_bra, l_w, l_ket)
        # A indices: (l_ket, p_in, r_ket)
        # Result T1: (l_bra, l_w, p_in, r_ket)
        @tensor T1[l_bra, l_w, p_in, r_ket] := E[l_bra, l_w, k] * A[k, p_in, r_ket]
        
        # Contract T1 with MPO (W)
        # T1: (l_bra, l_w, p_in, r_ket)
        # W: (l_w, p_out, p_in, r_w)
        # Shared: l_w, p_in
        # Result T2: (l_bra, r_ket, p_out, r_w)
        @tensor T2[l_bra, r_ket, p_out, r_w] := T1[l_bra, k_lw, k_pin, r_ket] * W[k_lw, p_out, k_pin, r_w]
        
        # Contract T2 with Bra (A_conj)
        # T2: (l_bra, r_ket, p_out, r_w)
        # A_conj: (l_bra, p_out, r_bra) -> indices (k_lbra, k_pout, r_bra)
        # Shared: l_bra, p_out (which is Bra's physical index)
        # Result E_new: (r_bra, r_w, r_ket)
        
        @tensor E_new[r_bra, r_w, r_ket] := T2[k_lbra, r_ket, k_pout, r_w] * conj(A[k_lbra, k_pout, r_bra])
        
        E = E_new
    end
    
    return E[1, 1, 1]
end

# --- MPO-MPO Multiplication (Optional) ---

"""
    contract_mpo_mpo(a::MPO, b::MPO) -> MPO

Contract two MPOs: A * B.
Phys_In of A connects to Phys_Out of B.
"""
function contract_mpo_mpo(a::MPO, b::MPO)
    @assert a.length == b.length
    
    L = a.length
    new_tensors = Vector{Array{ComplexF64, 4}}(undef, L)
    
    for i in 1:L
        A_ten = a.tensors[i] # (L_a, P_out_a, P_in_a, R_a)
        B_ten = b.tensors[i] # (L_b, P_out_b, P_in_b, R_b)
        
        # Contraction: P_in_a == P_out_b
        # We keep P_out_a (Top) and P_in_b (Bottom)
        # Merge Bonds (L_a, L_b) and (R_a, R_b)
        
        @tensor C[l_a, l_b, p_out, p_in, r_a, r_b] := A_ten[l_a, p_out, k, r_a] * B_ten[l_b, k, p_in, r_b]
        
        l_a, l_b, p_out, p_in, r_a, r_b = size(C)
        new_left = l_a * l_b
        new_right = r_a * r_b
        
        # Reshape to (New_Left, P_out, P_in, New_Right)
        # We need to group (l_a, l_b) -> dim 1
        # And (r_a, r_b) -> dim 4
        # Reshape assumes column major packing.
        # C indices are currently: l_a, l_b, p_out, p_in, r_a, r_b
        # Merging 1+2 and 5+6 requires permutation?
        # No, reshape merges adjacent dimensions.
        # We want (l_a, l_b) merged. They are adjacent (1,2). Good.
        # We want (r_a, r_b) merged. They are adjacent (5,6). Good.
        # But wait, @tensor output index order is alphabetical/sorted by default unless specified?
        # NO! @tensor output order is DETERMINED BY THE LHS of :=
        # We wrote: C[l_a, l_b, p_out, p_in, r_a, r_b]
        # So the memory layout is exactly that.
        
        new_tensor = reshape(C, new_left, p_out, p_in, new_right)
        new_tensors[i] = new_tensor
    end
    
    return MPO(L, new_tensors, a.phys_dims)
end

end # module
