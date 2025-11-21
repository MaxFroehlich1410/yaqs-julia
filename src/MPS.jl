module MPSModule

using TensorOperations
using LinearAlgebra
using StaticArrays
using Printf
using Base.Threads
using ..Decompositions

export MPS, scalar_product, local_expect, evaluate_all_local_expectations, measure_shots, normalize!, check_canonical_form, shift_orthogonality_center_right!, shift_orthogonality_center_left!, truncate!, pad_bond_dimension!

abstract type AbstractTensorNetwork end

"""
    MPS{T} <: AbstractTensorNetwork

Matrix Product State (MPS) using Column-Major memory layout.

# Layout
Tensors are stored with indices: `(Left_Bond, Physical, Right_Bond)`.
This contrasts with Python's Row-Major `(Physical, Left_Bond, Right_Bond)`.

# Fields
- `tensors::Vector{Array{T, 3}}`: The MPS tensors.
- `phys_dims::Vector{Int}`: Physical dimension of each site.
- `length::Int`: Number of sites.
"""
mutable struct MPS{T<:Number} <: AbstractTensorNetwork
    tensors::Vector{Array{T, 3}}
    phys_dims::Vector{Int}
    length::Int
    flipped::Bool

    function MPS(length::Int, tensors::Vector{Array{T, 3}}, phys_dims::Vector{Int}) where T
        new{T}(tensors, phys_dims, length, false)
    end
end

# --- Constructors ---

"""
    MPS(length::Int; 
        tensors=nothing, 
        physical_dimensions=nothing, 
        state="zeros", 
        basis_string=nothing)

Initialize an MPS. 
"""
function MPS(length::Int; 
             tensors::Union{Vector{Array{ComplexF64, 3}}, Nothing}=nothing,
             physical_dimensions::Union{Vector{Int}, Int, Nothing}=nothing,
             state::String="zeros",
             pad::Union{Int, Nothing}=nothing,
             basis_string::Union{String, Nothing}=nothing)

    # 1. Handle Physical Dimensions
    phys_dims = if isnothing(physical_dimensions)
        fill(2, length)
    elseif isa(physical_dimensions, Int)
        fill(physical_dimensions, length)
    else
        physical_dimensions
    end
    @assert length == Base.length(phys_dims)

    # 2. Handle Tensors
    if !isnothing(tensors)
        @assert Base.length(tensors) == length
        # Assuming input tensors might need layout conversion if coming from external source?
        # For now, assume provided tensors match the strict (Left, Phys, Right) layout if passed directly.
        return MPS(length, tensors, phys_dims)
    end

    # 3. Initialize State
    T = ComplexF64
    generated_tensors = Vector{Array{T, 3}}(undef, length)

    if state == "basis"
        @assert !isnothing(basis_string) "basis_string must be provided for 'basis' state initialization."
        return init_mps_from_basis(basis_string, phys_dims)
    end

    for i in 1:length
        d = phys_dims[i]
        vector = zeros(T, d)
        
        if state == "zeros"
            vector[1] = 1.0
        elseif state == "ones"
            if d >= 2
                vector[2] = 1.0
            else
                vector[1] = 1.0 # Fallback
            end
        elseif state == "x+"
            vector[1] = 1.0 / sqrt(2)
            vector[2] = 1.0 / sqrt(2)
        elseif state == "x-"
            vector[1] = 1.0 / sqrt(2)
            vector[2] = -1.0 / sqrt(2)
        elseif state == "y+"
            vector[1] = 1.0 / sqrt(2)
            vector[2] = 1.0im / sqrt(2)
        elseif state == "y-"
            vector[1] = 1.0 / sqrt(2)
            vector[2] = -1.0im / sqrt(2)
        elseif state == "Neel"
            idx = (i % 2 != 0) ? 1 : 2 # 1-based indexing: Odd i -> bit 0, Even i -> bit 1
            vector[idx] = 1.0
        elseif state == "wall"
            idx = (i <= length ÷ 2) ? 1 : 2
            vector[idx] = 1.0
        elseif state == "random"
            vector = rand(T, d)
            LinearAlgebra.normalize!(vector)
        else
            error("Invalid state string: $state")
        end

        # Layout: (Left=1, Phys=d, Right=1)
        # Python: (d, 1, 1) -> Transpose(2, 0, 1) -> (1, d, 1)
        # We create (1, d, 1) directly.
        tensor = reshape(vector, (1, d, 1))
        generated_tensors[i] = tensor
    end

    mps = MPS(length, generated_tensors, phys_dims)

    if state == "random"
        normalize!(mps)
    end

    if !isnothing(pad)
        pad_bond_dimension!(mps, pad)
    end

    return mps
end

function init_mps_from_basis(basis_string::String, phys_dims::Vector{Int})
    length = Base.length(basis_string)
    @assert length == Base.length(phys_dims)
    
    tensors = Vector{Array{ComplexF64, 3}}(undef, length)
    
    for (i, char) in enumerate(basis_string)
        idx = parse(Int, char) + 1 # 1-based indexing
        d = phys_dims[i]
        
        # Layout: (Left=1, Phys=d, Right=1)
        tensor = zeros(ComplexF64, 1, d, 1)
        tensor[1, idx, 1] = 1.0
        tensors[i] = tensor
    end
    
    return MPS(length, tensors, phys_dims)
end

# --- Helper Methods ---

function Base.show(io::IO, mps::MPS)
    print(io, "MPS(length=$(mps.length), max_bond=$(write_max_bond_dim(mps)))")
end

function write_max_bond_dim(mps::MPS)
    global_max = 0
    for tensor in mps.tensors
        # (Left, Phys, Right). Bond dims are 1 and 3.
        local_max = max(size(tensor, 1), size(tensor, 3))
        global_max = max(global_max, local_max)
    end
    return global_max
end

"""
    flip_network!(mps::MPS)

Logically flip the network (transpose tensors) to allow Right-to-Left operations.
Layout change: (L, P, R) -> (R, P, L).
"""
function flip_network!(mps::MPS)
    new_tensors = Vector{Array{ComplexF64, 3}}(undef, mps.length)
    for i in 1:mps.length
        # Permute (Left, Phys, Right) -> (Right, Phys, Left)
        new_tensors[i] = permutedims(mps.tensors[i], (3, 2, 1))
    end
    reverse!(new_tensors)
    mps.tensors = new_tensors
    mps.flipped = !mps.flipped
end

# --- Canonicalization & Linear Algebra ---

"""
    shift_orthogonality_center_right!(mps::MPS, current_center::Int)

Perform QR/SVD to shift orthogonality center from `current_center` to `current_center + 1`.
Optimized to avoid full deepcopies.
"""
function shift_orthogonality_center_right!(mps::MPS, current_center::Int; decomposition::String="QR", cutoff::Float64=1e-15)
    A = mps.tensors[current_center]
    l, p, r = size(A)
    
    # Reshape for decomposition: Combine (Left, Phys) into Row
    # Matrix A_mat: (l*p) x r
    A_mat = reshape(A, l * p, r)
    
    if decomposition == "QR" || current_center == mps.length # QR is standard for shifting
        # QR Decomposition
        Q_fact = qr(A_mat)
        Q = Matrix(Q_fact.Q)
        R = Matrix(Q_fact.R)
        
        # Q has shape (l*p, new_bond). R has shape (new_bond, r).
        new_bond = size(Q, 2)
        
        # Reshape Q back to Tensor (Left, Phys, Right=new_bond)
        mps.tensors[current_center] = reshape(Q, l, p, new_bond)
        
        # Contract R into next site if exists
        if current_center < mps.length
            B = mps.tensors[current_center + 1]
            # B layout: (Left=r, Phys, Right)
            # R layout: (new_bond, r)
            # We need C[new_bond, p, next_r] := R[new_bond, k] * B[k, p, next_r]
            
            # Using @tensor
            @tensor B_new[a, b, c] := R[a, k] * B[k, b, c]
            mps.tensors[current_center + 1] = B_new
        end
        
    elseif decomposition == "SVD"
        # Only used if we need truncation during shift, but usually shift preserves dim
        # For shift, QR is preferred. Implementing SVD for completeness.
        F = svd(A_mat)
        U, S, Vt = F.U, F.S, F.Vt
        
        # Truncate?
        # (Simple shift usually keeps all)
        
        # U: (l*p, bond)
        # S*Vt: (bond, r)
        
        new_bond = length(S)
        mps.tensors[current_center] = reshape(U, l, p, new_bond)
        
        if current_center < mps.length
            B = mps.tensors[current_center + 1]
            rem = Diagonal(S) * Vt
            @tensor B_new[a, b, c] := rem[a, k] * B[k, b, c]
            mps.tensors[current_center + 1] = B_new
        end
    end
end

"""
    shift_orthogonality_center_left!(mps::MPS, current_center::Int; decomposition::String="QR", cutoff::Float64=1e-15)

Perform LQ/SVD to shift orthogonality center from `current_center` to `current_center - 1`.
Native implementation avoiding network flipping for maximum efficiency.
"""
function shift_orthogonality_center_left!(mps::MPS, current_center::Int; decomposition::String="QR", cutoff::Float64=1e-15)
    # Shift from i to i-1
    # Current Tensor B at i: (L, P, R)
    # Goal: Decompose B -> L * Q, where Q is Right Canonical (rows orthonormal)
    # Matrix shape: (L) x (P*R)
    
    B = mps.tensors[current_center]
    l, p, r = size(B)
    B_mat = reshape(B, l, p * r)
    
    if decomposition == "QR" || current_center == 1
        # LQ Decomposition via QR of transpose
        # B = L * Q  => B' = Q' * L' = Q_tilde * R_tilde
        # We perform QR on B' (dims (p*r) x l)
        # Q_fact = qr(B_mat')
        
        # Note: Julia's LQ is available? 
        # LinearAlgebra.lq exists.
        F = lq(B_mat)
        L_factor = Matrix(F.L)
        Q_ortho = Matrix(F.Q)
        
        # Q_ortho shape: (l, p*r) - effectively new bond dim might change if rank deficient
        new_bond = size(Q_ortho, 1)
        
        # Update current site to Right Canonical Form
        mps.tensors[current_center] = reshape(Q_ortho, new_bond, p, r)
        
        # Absorb L_factor into Left Neighbor (i-1)
        if current_center > 1
            A_left = mps.tensors[current_center - 1] # (L_prev, P_prev, R_prev=l)
            # Contract: A_new = A_left * L_factor
            # A_left: (l_prev, p_prev, k)
            # L_factor: (k, new_bond) [Since L matches R_prev, and new_bond matches Left of B]
            # Wait, B was (l, pr). L_factor is (l, new_bond)? No, LQ -> (m,n) = (m,k)*(k,n). 
            # B (l, pr) = L (l, l) * Q (l, pr) usually.
            # So L_factor is (l, l).
            
            @tensor A_new[l_prev, p_prev, new_r] := A_left[l_prev, p_prev, k] * L_factor[k, new_r]
            mps.tensors[current_center - 1] = A_new
        end
        
    elseif decomposition == "SVD"
        # SVD: B = U S V'
        # Right Canonical: V' is the tensor at site i
        # Left factor: U S
        F = svd(B_mat)
        U, S, Vt = F.U, F.S, F.Vt
        
        # Truncation logic could go here
        new_bond = length(S)
        
        # Update site i
        mps.tensors[current_center] = reshape(Vt, new_bond, p, r)
        
        # Update site i-1
        if current_center > 1
            A_left = mps.tensors[current_center - 1]
            US = U * Diagonal(S)
            @tensor A_new[l_prev, p_prev, new_r] := A_left[l_prev, p_prev, k] * US[k, new_r]
            mps.tensors[current_center - 1] = A_new
        end
    end
end

"""
    truncate!(mps::MPS, threshold=1e-12)

In-place truncation using SVD.
"""
function truncate!(mps::MPS; threshold::Float64=1e-12, max_bond_dim::Union{Int, Nothing}=nothing)
    # Assume currently in canonical form or start from center
    # Simplified: Sweep Right then Left (or just use the current center)
    
    if mps.length == 1
        return
    end
    
    # We need to find the center. For simplicity, assume we start at 1 and sweep to end, then back.
    # Or just implement the "Two-Site SVD" sweep.
    
    # Forward Sweep
    for i in 1:(mps.length - 1)
        A = mps.tensors[i]
        B = mps.tensors[i+1]
        A_new, B_new = two_site_svd(A, B, threshold; max_bond_dim=max_bond_dim)
        mps.tensors[i] = A_new
        mps.tensors[i+1] = B_new
    end
    
    # Backward Sweep
    flip_network!(mps)
    for i in 1:(mps.length - 1)
        A = mps.tensors[i]
        B = mps.tensors[i+1]
        A_new, B_new = two_site_svd(A, B, threshold; max_bond_dim=max_bond_dim)
        mps.tensors[i] = A_new
        mps.tensors[i+1] = B_new
    end
    flip_network!(mps)
    
    # Re-normalize to ensure unit norm state
    normalize!(mps)
end



function normalize!(mps::MPS; form::String="B")
    # Normalize to Right Canonical (B-form)
    # Sweep Right-to-Left (L -> 1)
    # Shift orthogonality center from L down to 1
    
    # We start by assuming the center is at L (or we move it there virtually?)
    # Actually, to make it Right Canonical, we need all sites 2..L to be B-tensors.
    # This is achieved by shifting the orthogonality center all the way to site 1.
    
    # Wait, if we start with arbitrary state, we first sweep Left->Right to make it Left Canonical (A-form)?
    # Or we can just sweep Right->Left directly using LQ.
    
    # Algorithm for full B-form normalization:
    # 1. Start at L. Perform LQ. Absorb L into L-1.
    # 2. Repeat until site 2.
    # 3. At site 1, normalize the vector.
    
    for i in mps.length:-1:2
        shift_orthogonality_center_left!(mps, i)
    end
    
    # Center is now at site 1.
    # Explicitly normalize the state (divide by norm)
    center_tensor = mps.tensors[1]
    norm_val = norm(center_tensor)
    if norm_val > 1e-15
        mps.tensors[1] ./= norm_val
    end
end


# --- Contractions ---

"""
    scalar_product(a::MPS, b::MPS)

Compute <a|b>.
"""
function scalar_product(a::MPS, b::MPS)
    @assert a.length == b.length
    
    # Zipper contraction
    # Initialize Transfer Matrix E (1x1)
    # We assume boundary conditions are 1.
    E = ones(ComplexF64, 1, 1) 
    
    for i in 1:a.length
        A_ten = a.tensors[i] # (L, P, R)
        B_ten = b.tensors[i] # (L, P, R)
        
        # Contract: E[l_a, l_b] * A[l_a, p, r_a] * conj(B[l_b, p, r_b])
        # Output E_next[r_a, r_b]
        @tensor E_new[r_a, r_b] := E[l_a, l_b] * A_ten[l_a, p, r_a] * conj(B_ten[l_b, p, r_b])
        E = E_new
    end
    
    return E[1, 1]
end

"""
    local_expect(mps::MPS, operator::AbstractMatrix, site::Int)

Compute <ψ|O_i|ψ>. Uses temporary tensor contraction to avoid deepcopy.
"""
function local_expect(mps::MPS, operator::AbstractMatrix{T}, site::Int) where T
    # 1. Fetch tensors
    A = mps.tensors[site] # (L, P, R)
    
    # 2. Apply gate locally: O[p_new, p_old] * A[l, p_old, r]
    # Note: Matrix usually (row, col) -> (p_new, p_old)
    @tensor A_prime[l, p_new, r] := operator[p_new, p_old] * A[l, p_old, r]
    
    # 3. Calculate overlap <ψ|ψ'>
    # We can reuse the scalar product logic but with one modified tensor.
    # Optimization: If MPS is in canonical form around `site`, we only need to contract locally!
    # But general case: Contract full zipper.
    
    # Zipper from Left
    E_left = ones(ComplexF64, 1, 1)
    for i in 1:(site-1)
        T_tens = mps.tensors[i]
        @tensor E_next[r1, r2] := E_left[l1, l2] * T_tens[l1, p, r1] * conj(T_tens[l2, p, r2])
        E_left = E_next
    end
    
    # Zipper from Right
    E_right = ones(ComplexF64, 1, 1)
    for i in mps.length:-1:(site+1)
        T_tens = mps.tensors[i]
        @tensor E_next[l1, l2] := T_tens[l1, p, r1] * conj(T_tens[l2, p, r2]) * E_right[r1, r2]
        E_right = E_next
    end
    
    # Combine at site
    # <L| <A| O |A> |R>
    # E_left[l_a, l_a'] * A'[l_a, p, r_a] * conj(A[l_a', p, r_a']) * E_right[r_a, r_a']
    
    A_orig = mps.tensors[site]
    @tensor val[] := E_left[l1, l2] * A_prime[l1, p, r1] * conj(A_orig[l2, p, r2]) * E_right[r1, r2]
    
    return val[]
end

"""
    evaluate_all_local_expectations(mps::MPS, operators::Vector{<:AbstractMatrix})

Efficiently compute expectation values for a list of local operators (one per site).
Complexity: O(L) instead of O(L^2).
"""
function evaluate_all_local_expectations(mps::MPS, operators::Vector{<:AbstractMatrix})
    @assert length(operators) == mps.length
    
    T_val = eltype(mps.tensors[1])
    
    # 1. Pre-calculate Right Environments
    # E_right[i] is the environment block contracted from site i to L.
    # We need E_right[i+1] for site i.
    E_right_storage = Vector{Array{T_val, 2}}(undef, mps.length + 1)
    E_right_storage[end] = ones(T_val, 1, 1)
    
    for i in mps.length:-1:2
        T = mps.tensors[i]
        prev_E = E_right_storage[i+1]
        # Contract: T[l,p,r] * conj(T[l',p,r']) * prev_E[r,r']
        @tensor next_E[l1, l2] := T[l1, p, r1] * conj(T[l2, p, r2]) * prev_E[r1, r2]
        E_right_storage[i] = next_E
    end
    
    # 2. Sweep Left-to-Right
    results = Vector{ComplexF64}(undef, mps.length)
    E_left = ones(T_val, 1, 1)
    
    for i in 1:mps.length
        T = mps.tensors[i]
        Op = operators[i]
        R_env = E_right_storage[i+1]
        
        # Apply Operator: Op[p_new, p_old] * T[l, p_old, r]
        @tensor T_op[l, p, r] := Op[p, p_old] * T[l, p_old, r]
        
        # Contract Center: E_left * T_op * conj(T) * E_right
        @tensor val[] := E_left[l1, l2] * T_op[l1, p, r1] * conj(T[l2, p, r2]) * R_env[r1, r2]
        results[i] = val[]
        
        # Update E_left
        if i < mps.length
            @tensor E_next[r1, r2] := E_left[l1, l2] * T[l1, p, r1] * conj(T[l2, p, r2])
            E_left = E_next
        end
    end
    
    return results
end

"""
    expect(mps::MPS, observable)

Generic expectation value. Assumes observable has `.gate.matrix` and `.sites`.
"""
function expect(mps::MPS, observable)
    # Duck-typing for observable
    # Assuming observable has field `gate` with `matrix` and field `sites`.
    # In a real package, use proper types.
    
    gate_matrix = observable.gate.matrix
    sites = observable.sites
    
    if size(gate_matrix, 1) == 2 # Single site (assuming qubit)
        site_idx = (sites isa Vector) ? sites[1] : sites
        return real(local_expect(mps, gate_matrix, site_idx))
    elseif size(gate_matrix, 1) == 4 # Two site
        s = (sites isa Vector) ? sites : [sites]
        @assert length(s) == 2 "Two-site operator requires 2 sites"
        return real(local_expect_two_site(mps, gate_matrix, s[1], s[2]))
    else
        error("Unsupported gate dimension")
    end
end


# --- Measurements ---

"""
    measure_shots(mps::MPS, shots::Int)

Perform `shots` parallel measurements. Returns a Dict of outcome -> count.
"""
function measure_shots(mps::MPS, shots::Int)
    # Thread-safe collection
    results = Vector{Int}(undef, shots)
    
    Threads.@threads for i in 1:shots
        results[i] = single_shot_measure(mps)
    end
    
    # Aggregate
    counts = Dict{Int, Int}()
    for r in results
        counts[r] = get(counts, r, 0) + 1
    end
    return counts
end

"""
    single_shot_measure(mps::MPS)

Perform one projective measurement pass without modifying the original MPS.
Strategy: Carry the collapsed state forward as a temporary tensor.
"""
function single_shot_measure(mps::MPS)
    bitstring = 0
    
    # Start with the first tensor
    # We maintain a "current_state" which effectively represents the contraction 
    # of the projected part into the next site.
    
    # Actually, we iterate sites.
    # At site i, we have the rest of the chain to the right.
    # But we need to know the "effective" input from the left.
    # Wait, the standard algorithm:
    # 1. Compute local RDM rho_i. (Requires contracting the whole rest of the network? No, only if canonical).
    #    Assumption: MPS should be in canonical form (orthogonality center at current site) for efficient sampling.
    #    But we can't re-canonicalize for every shot in parallel!
    #
    #    Correct approach for arbitrary MPS (expensive): Contract full environment.
    #    Correct approach for Canonical MPS (cheap): Tensors satisfy left/right orthogonality.
    #
    #    If we assume the MPS is NOT re-canonicalized for every sample (impossible in parallel),
    #    we must treat it carefully.
    #    However, the Python code does `copy.deepcopy(self)` then iterates. 
    #    It calculates `reduced_density_matrix` using ONLY the current tensor:
    #    `oe.contract("abc, dbc->ad", tensor, np.conj(tensor))`
    #    This implies the Python code ASSUMES the MPS is in canonical form (Orthogonality Center at site 0, sweeping right)?
    #    OR it assumes Left-Canonical form everywhere?
    #    
    #    If `tensor` is A (Left Orthogonal), then A^dag A = I. Contraction yields identity.
    #    If `tensor` is C (Center), contraction yields Rho.
    #    
    #    The Python code propagates the projection:
    #    `temp_state.tensors[site + 1] = 1/norm * contract(projected, next_tensor)`
    #
    #    This propagation effectively "pushes" the center to the right!
    #    So, yes, this algorithm effectively moves the orthogonality center as it measures.
    #
    #    So my "Carry Tensor" strategy is exactly correct.
    
    # Initial "incoming" Left Block is trivial (1x1 Identity)
    # But we actually modify the tensors themselves in the Python code.
    # In Julia, we will carry the MODIFIED tensor `T_curr` into the next step.
    
    # We need to handle the FIRST tensor specifically, then loop.
    # But wait, the tensor at site `i` depends on the projection at `i-1`.
    
    # Let's use a variable `current_tensor` that holds the tensor at `site`.
    # For `site=1`, it is `mps.tensors[1]`.
    # For `site>1`, it is `contract(proj_prev, mps.tensors[site])`.
    
    current_tensor = mps.tensors[1] # Copy? No, array reference. Be careful not to mutate it.
    # Actually, we will be contracting it, so we get a NEW array every time. Safe.
    
    for i in 1:mps.length
        # 1. Compute RDM at current site
        # Shape: (Left, Phys, Right).
        # Contract Left and Right indices with Self Conjugate.
        # rho[p, p'] = sum_{l,r} T[l, p, r] * conj(T[l, p', r])
        @tensor rho[p, p_prime] := current_tensor[l, p, r] * conj(current_tensor[l, p_prime, r])
        
        # 2. Probabilities (Diagonal)
        probs = real.(diag(rho))
        
        # Normalize probs (numerical stability)
        prob_sum = sum(probs)
        if prob_sum < 1e-12
            # Error or very unlikely path
            probs = ones(length(probs)) ./ length(probs)
        else
            probs ./= prob_sum
        end
        
        # 3. Sample
        outcome = sample_index(probs) # 1-based index
        bitstring += (outcome - 1) * (1 << (i - 1)) # Python logic: sum(c << i ...)
        
        # 4. Project
        # We select the slice `outcome` from physical index.
        # projected_T = current_tensor[:, outcome, :] -> Shape (Left, Right)
        projected_slice = @view current_tensor[:, outcome, :]
        
        # 5. Propagate to next site
        if i < mps.length
            next_site_tensor = mps.tensors[i+1] # (L_next, P_next, R_next)
            # L_next matches R of current.
            # Contract: proj[l, r] * next[r, p, next_r]
            # Norm factor: 1 / sqrt(probs[outcome])
            
            factor = 1.0 / sqrt(probs[outcome])
            
            @tensor next_T[l, p, r] := projected_slice[l, k] * next_site_tensor[k, p, r]
            current_tensor = next_T .* factor
        end
    end
    
    return bitstring
end

function sample_index(probs::Vector{Float64})
    r = rand()
    cumsum = 0.0
    for (i, p) in enumerate(probs)
        cumsum += p
        if r < cumsum
            return i
        end
    end
    return length(probs)
end

"""
    project_onto_bitstring(mps::MPS, bitstring::String)

Return probability |<b|psi>|^2.
"""
function project_onto_bitstring(mps::MPS, bitstring::String)
    @assert Base.length(bitstring) == mps.length
    
    current_tensor = mps.tensors[1]
    total_norm_sq = 1.0
    
    for (i, char) in enumerate(bitstring)
        val = parse(Int, char) + 1 # 1-based
        
        # Project
        proj = current_tensor[:, val, :] # (Left, Right)
        
        norm_val = norm(proj)
        if norm_val == 0
            return ComplexF64(0.0)
        end
        total_norm_sq *= norm_val^2
        
        if i < mps.length
            # Propagate
            next_tensor = mps.tensors[i+1]
            @tensor next_T[l, p, r] := proj[l, k] * next_tensor[k, p, r]
            current_tensor = next_T .* (1.0 / norm_val)
        end
    end
    
    return ComplexF64(total_norm_sq)
end

# --- Utilities ---

function pad_bond_dimension!(mps::MPS, target_dim::Int)
    for i in 1:mps.length
        tensor = mps.tensors[i] # (Left, Phys, Right)
        l, p, r = size(tensor)

        # Calculate targets
        if i == 1
            left_target = 1
        else
            exp_left = min(i - 1, mps.length - (i - 1)) # bond index i-1
            left_target = min(target_dim, 2^exp_left)
        end

        if i == mps.length
            right_target = 1
        else
            exp_right = min(i, mps.length - i) # bond index i
            right_target = min(target_dim, 2^exp_right)
        end

        if l > left_target || r > right_target
            error("Target bond dim must be at least current bond dim.")
        end

        if l < left_target || r < right_target
            T = eltype(tensor)
            new_tensor = zeros(T, left_target, p, right_target)
            
            # Copy data: indices 1:l, 1:p, 1:r
            # In Julia 1-based indexing
            new_tensor[1:l, :, 1:r] = tensor
            
            mps.tensors[i] = new_tensor
        end
    end
    normalize!(mps)
end

function check_canonical_form(mps::MPS)
    a_truth = falses(mps.length)
    b_truth = falses(mps.length)

    # Check Left Canonical (A-form)
    # Contract Left and Phys: T^dag * T = I
    for i in 1:mps.length
        T = mps.tensors[i] # (L, P, R)
        l, p, r = size(T)
        
        # Contract over L and P
        @tensor mat[r1, r2] := conj(T[l, p, r1]) * T[l, p, r2]
        
        if isapprox(mat, I; atol=1e-10)
            a_truth[i] = true
        end
    end

    # Check Right Canonical (B-form)
    # Contract Phys and Right: T * T^dag = I
    for i in mps.length:-1:1
        T = mps.tensors[i]
        l, p, r = size(T)
        
        # Contract over P and R
        @tensor mat[l1, l2] := T[l1, p, r] * conj(T[l2, p, r])
        
        if isapprox(mat, I; atol=1e-10)
            b_truth[i] = true
        end
    end

    # Identify Center
    mixed_truth = falses(mps.length)
    for i in 1:mps.length
        # If everything to the left is A-form AND everything to the right is B-form
        # Note: Python logic `all(a_truth[:i])` excludes i. `all(b_truth[i+1:])`.
        # So site i is the center.
        left_ok = (i == 1) ? true : all(a_truth[1:i-1])
        right_ok = (i == mps.length) ? true : all(b_truth[i+1:end])
        
        if left_ok && right_ok
            mixed_truth[i] = true
        end
    end

    return findall(mixed_truth)
end

end # module

