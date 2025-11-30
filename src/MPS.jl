module MPSModule

using LinearAlgebra
using StaticArrays
using Printf
using TensorOperations
using Random

export MPS, check_if_valid_mps, check_canonical_form, pad_bond_dimension!
export write_max_bond_dim, to_vec
export shift_orthogonality_center!, normalize!, truncate!
export scalar_product, norm, local_expect, local_expect_two_site, single_shot_measure, measure_single_shot, measure_shots, project_onto_bitstring, evaluate_all_local_expectations

# --- Constants & Types ---

const AbstractTensor3{T} = AbstractArray{T, 3}

"""
    MPS{T} <: AbstractTensorNetwork

Matrix Product State (MPS) with explicitly tracked orthogonality center.
Memory Layout: Column-Major (Left, Physical, Right).

Fields:
- `tensors`: Vector of rank-3 tensors.
- `phys_dims`: Physical dimensions (usually 2 for qubits).
- `length`: Number of sites.
- `orth_center`: Index of the orthogonality center (1-based). 
                 Sites 1..oc-1 are Left-Canonical (A).
                 Sites oc+1..L are Right-Canonical (B).
                 Site oc is the center (C).
"""
mutable struct MPS{T}
    tensors::Vector{Array{T, 3}}
    phys_dims::Vector{Int}
    length::Int
    orth_center::Int

    # Inner constructor for specific T
    function MPS{T}(length::Int, tensors::Vector{Array{T, 3}}, phys_dims::Vector{Int}, orth_center::Int=1) where T
        new{T}(tensors, phys_dims, length, orth_center)
    end
end

# Outer constructor to infer T
function MPS(length::Int, tensors::Vector{Array{T, 3}}, phys_dims::Vector{Int}, orth_center::Int=1) where T
    return MPS{T}(length, tensors, phys_dims, orth_center)
end

# --- Constructors ---

function MPS(length::Int; 
             tensors::Union{Vector{Array{ComplexF64, 3}}, Nothing}=nothing,
             physical_dimensions::Union{Vector{Int}, Int, Nothing}=nothing,
             state::String="zeros",
             pad::Union{Int, Nothing}=nothing,
             basis_string::Union{String, Nothing}=nothing)

    # 1. Physical Dims
    p_dims = if isnothing(physical_dimensions)
        fill(2, length)
    elseif isa(physical_dimensions, Int)
        fill(physical_dimensions, length)
    else
        physical_dimensions
    end
    
    T = ComplexF64

    # 2. Tensors
    if !isnothing(tensors)
        @assert Base.length(tensors) == length
        # Assume provided tensors are normalized/canonical if orth_center not specified? 
        # We default to 1 and let validity checks happen later if needed.
        return MPS(length, tensors, p_dims, 1)
    end

    new_tensors = Vector{Array{T, 3}}(undef, length)

    # 3. Initialization
    if state == "basis"
        @assert !isnothing(basis_string) "Must provide basis_string for state='basis'"
        @assert Base.length(basis_string) == length
        
        for i in 1:length
            d = p_dims[i]
            idx = parse(Int, basis_string[i]) + 1 # 1-based
            v = zeros(T, 1, d, 1)
            v[1, idx, 1] = 1.0
            new_tensors[i] = v
        end
        
    elseif state == "random"
        # Random initialization
    for i in 1:length
            d = p_dims[i]
            chi_l = (i == 1) ? 1 : 4 # Arbitrary small bond dim start
            chi_r = (i == length) ? 1 : 4
            new_tensors[i] = randn(T, chi_l, d, chi_r)
        end
        mps = MPS(length, new_tensors, p_dims, 1)
        normalize!(mps) # Puts in canonical form
        return mps
        
    else
        # Product States
        for i in 1:length
            d = p_dims[i]
            v = zeros(T, d)
        if state == "zeros"
                v[1] = 1.0
        elseif state == "ones"
                if d > 1; v[2] = 1.0; else; v[1] = 1.0; end
        elseif state == "x+"
                v[1] = 1/sqrt(2); v[2] = 1/sqrt(2)
        elseif state == "x-"
                v[1] = 1/sqrt(2); v[2] = -1/sqrt(2)
        elseif state == "y+"
                v[1] = 1/sqrt(2); v[2] = im/sqrt(2)
        elseif state == "y-"
                v[1] = 1/sqrt(2); v[2] = -im/sqrt(2)
        elseif state == "Neel"
                idx = (i % 2 != 0) ? 1 : 2
                v[idx] = 1.0
        elseif state == "wall"
            idx = (i <= length ÷ 2) ? 1 : 2
                v[idx] = 1.0
            else
                error("Unknown state: $state")
        end

            # Reshape to (1, d, 1)
            new_tensors[i] = reshape(v, 1, d, 1)
        end
    end

    mps = MPS(length, new_tensors, p_dims, 1)

    if !isnothing(pad)
        pad_bond_dimension!(mps, pad)
    end

    return mps
end

# --- Core Functionality ---

"""
    check_if_valid_mps(mps)

Verify consistency of bond dimensions.
"""
function check_if_valid_mps(mps::MPS)
    for i in 1:(mps.length - 1)
        # T[i]: (L, d, R)
        # T[i+1]: (L', d', R')
        # R must equal L'
        r_bond = size(mps.tensors[i], 3)
        l_next = size(mps.tensors[i+1], 1)
        @assert r_bond == l_next "Bond dimension mismatch at site $i ($r_bond) -> $(i+1) ($l_next)"
    end
    return true
end

"""
    write_max_bond_dim(mps)
"""
function write_max_bond_dim(mps::MPS)
    max_chi = 0
    for T in mps.tensors
        max_chi = max(max_chi, size(T, 1), size(T, 3))
    end
    return max_chi
end

"""
    shift_orthogonality_center!(mps, new_center)

Shift the orthogonality center to `new_center` using QR (right) or LQ (left).
No flipping. In-place updates.
"""
function shift_orthogonality_center!(mps::MPS{T}, new_center::Int) where T
    oc = mps.orth_center
    if oc == new_center
        return
    end
    
    if new_center > oc
        # Move Right: oc -> oc+1 -> ... -> new_center
        for i in oc:(new_center - 1)
            # Site i is currently center. Make it Left Canonical (A).
            # T[i]: (L, d, R)
            A = mps.tensors[i]
            L, d, R = size(A)
            
            # Reshape for QR: (L*d, R)
            Mat = reshape(A, L*d, R)
            
            # QR
            F = qr(Mat) # Thin QR by default
            Q_thin = Matrix(F.Q)
            R_thin = Matrix(F.R)
            
            # Dimensions: Q (L*d, r_new), R (r_new, R)
            r_new = size(R_thin, 1)
            
            # Update current site to Q (reshaped)
            mps.tensors[i] = reshape(Q_thin, L, d, r_new)
            
            # Absorb R into next site (i+1)
            # Next: (R, d_next, R_next)
            Next = mps.tensors[i+1]
            d_next = size(Next, 2)
            r_next = size(Next, 3)
            
            # Optimization: avoid full reshape if not needed, but reshape is cheap
            Next_mat = reshape(Next, R, d_next * r_next)
            
            # Mul: (r_new, R) * (R, dim_rest) -> (r_new, dim_rest)
            # In-place? R_thin is small usually. Next_mat can be large.
            # New_Next = R_thin * Next_mat
            New_Next = similar(Next_mat, r_new, d_next * r_next)
            mul!(New_Next, R_thin, Next_mat)
            
            mps.tensors[i+1] = reshape(New_Next, r_new, d_next, r_next)
        end
    else
        # Move Left: oc -> oc-1 -> ... -> new_center
        for i in range(oc, stop=new_center+1, step=-1)
            # Site i is Center. Make it Right Canonical (B).
            # T[i]: (L, d, R)
            A = mps.tensors[i]
            L, d, R = size(A)
            
            # Reshape for LQ: (L, d*R)
            Mat = reshape(A, L, d*R)
            
            # LQ
            F = lq(Mat)
            L_thin = Matrix(F.L)
            Q_thin = Matrix(F.Q)
        
            # Dimensions: L_thin (L, r_new), Q_thin (r_new, d*R)
            r_new = size(L_thin, 2)
        
            # Update current site to Q (Right Canonical)
            mps.tensors[i] = reshape(Q_thin, r_new, d, R)
        
            # Absorb L into prev site (i-1)
            # Prev: (L_prev, d_prev, L)
            Prev = mps.tensors[i-1]
            l_prev = size(Prev, 1)
            d_prev = size(Prev, 2)
            
            Prev_mat = reshape(Prev, l_prev * d_prev, L)
            
            # Mul: (dim_prev, L) * (L, r_new) -> (dim_prev, r_new)
            New_Prev = similar(Prev_mat, l_prev * d_prev, r_new)
            mul!(New_Prev, Prev_mat, L_thin)
            
            mps.tensors[i-1] = reshape(New_Prev, l_prev, d_prev, r_new)
        end
    end
    
    mps.orth_center = new_center
end

"""
    normalize!(mps; form="B")

Brings the MPS to right-canonical form (orth_center = 1) and normalizes the state norm to 1.
`form` argument is ignored but kept for compatibility; strictly enforces right-canonical (B) form which is equivalent to orth_center=1.
"""
function normalize!(mps::MPS{T}; form::String="B") where T
    # Sweep L -> 2. Site 1 absorbs everything.
    # This ensures sites 2..L are Right Canonical (B).
    
    for i in mps.length:-1:2
        # Site i: (L, d, R). Make B.
        A = mps.tensors[i]
        L, d, R = size(A)
        Mat = reshape(A, L, d*R)
        
        F = lq(Mat)
        L_mat = Matrix(F.L)
        Q_mat = Matrix(F.Q)
        
        r_new = size(L_mat, 2)
        mps.tensors[i] = reshape(Q_mat, r_new, d, R)
        
        # Update i-1
        Prev = mps.tensors[i-1]
        l_prev, d_prev, _ = size(Prev)
        Prev_mat = reshape(Prev, l_prev * d_prev, L)
        
        New_Prev = similar(Prev_mat, l_prev * d_prev, r_new)
        mul!(New_Prev, Prev_mat, L_mat)
        
        mps.tensors[i-1] = reshape(New_Prev, l_prev, d_prev, r_new)
    end
    
    # Now orthogonality center is at 1.
    # Normalize site 1
    A1 = mps.tensors[1]
    nrm = norm(A1)
    if nrm > 1e-13
        mps.tensors[1] .*= (1.0 / nrm)
    end
    
    mps.orth_center = 1
end

"""
    truncate!(mps; threshold, max_bond_dim)

Single sweep truncation (1->L-1). Updates MPS in place. 
Returns truncation error.
Center moves 1 -> L.
"""
function truncate!(mps::MPS{T}; threshold::Float64=1e-12, max_bond_dim::Union{Int, Nothing}=nothing) where T
    # Ensure we start at 1
    shift_orthogonality_center!(mps, 1)
    
    total_trunc_err = 0.0
    
    for i in 1:(mps.length - 1)
        # Contract A[i] (Center) and A[i+1] (Right Canonical)
        A = mps.tensors[i]   # (L, d1, k)
        B = mps.tensors[i+1] # (k, d2, R)
        
        l, d1, k = size(A)
        _, d2, r = size(B)
        
        # Merge: (L, d1, d2, R)
        # Reshape A: (L*d1, k)
        # Reshape B: (k, d2*R)
        Amat = reshape(A, l*d1, k)
        Bmat = reshape(B, k, d2*r)
        
        # Theta = Amat * Bmat
        Theta = similar(Amat, l*d1, d2*r)
        mul!(Theta, Amat, Bmat)
        
        # SVD
        F = svd(Theta)
        U, S, Vt = F.U, F.S, F.Vt
        
        # Truncate
        # Match Python YAQS exactly: absolute threshold, not relative
        # Python logic (from two_site_svd):
        # discard = 0.0
        # keep = len(s_vec)
        # min_keep = 2
        # for idx, s in enumerate(reversed(s_vec)):
        #     discard += s**2
        #     if discard >= threshold:
        #         keep = max(len(s_vec) - idx, min_keep)
        #         break
        #
        # In Julia: k goes from length(S) (smallest) down to 1 (largest)
        # Python idx=0 corresponds to Julia k=length(S)
        # Python idx corresponds to Julia k = length(S) - idx
        # When Python sets keep = len(s_vec) - idx, Julia sets keep_rank = k
        
        discarded_sq = 0.0
        keep_rank = length(S)
        min_keep = 2  # Python uses 2 to prevent pathological dimension-1 truncation
        
        # Accumulate discarded weight from smallest singular values
        # Python: enumerate(reversed(s_vec)) gives idx=0 for smallest, idx=1 for second-smallest, etc.
        # Julia: k=length(S) for smallest, k=length(S)-1 for second-smallest, etc.
        # Mapping: Python idx corresponds to Julia k = length(S) - idx
        # When Python sets keep = len(s_vec) - idx, Julia sets keep_rank = k
        for k in length(S):-1:1
            discarded_sq += S[k]^2
            if discarded_sq >= threshold
                # Python: keep = max(len(s_vec) - idx, min_keep)
                # Julia: keep_rank = max(k, min_keep) where k = len(S) - idx
                keep_rank = max(k, min_keep)
                break
            end
        end
        
        # 2. Max Bond
        if !isnothing(max_bond_dim)
            keep_rank = min(keep_rank, max_bond_dim)
        end
        
        # Update Error
        trunc_S = S[(keep_rank+1):end]
        total_trunc_err += sum(trunc_S.^2)
        
        # Resize
        U_trunc = U[:, 1:keep_rank]
        S_trunc = S[1:keep_rank]
        Vt_trunc = Vt[1:keep_rank, :]
        
        # Update A[i]: Left Canonical U
        mps.tensors[i] = reshape(U_trunc, l, d1, keep_rank)
        
        # Update A[i+1]: Center S*V
        SV = Diagonal(S_trunc) * Vt_trunc
        mps.tensors[i+1] = reshape(SV, keep_rank, d2, r)
    end
    
    mps.orth_center = mps.length
    normalize!(mps)
    return total_trunc_err
end

"""
    pad_bond_dimension!(mps, target_dim; noise_scale=1e-8, rng=Random.default_rng())

Pad every internal bond up to `target_dim` and seed freshly added virtual
subspaces with small complex noise so that single-site TDVP can explore the
enlarged manifold. The perturbation keeps the physical state unchanged up to
`noise_scale` (default `1e-8`) but guarantees that all Schmidt values are
strictly non-zero, which is required for entanglement growth in the 1-site
algorithm.
"""
function pad_bond_dimension!(mps::MPS{T}, target_dim::Int;
                             noise_scale::Real=1e-8,
                             rng::AbstractRNG=Random.default_rng()) where T
    @assert target_dim > 0 "target_dim must be positive"
    normalize!(mps)
    
    noise_amp = float(noise_scale)
    
    for i in 1:(mps.length - 1)
        A = mps.tensors[i]     # (L, d, R)
        B = mps.tensors[i+1]   # (R, d, R_next)
        
        chi_current = size(A, 3)
        if chi_current >= target_dim
            continue
        end
        
        new_chi = target_dim
        
        # Expand site i along its right bond
        Ldim, dphys, Rdim = size(A)
        new_A = zeros(T, Ldim, dphys, new_chi)
        new_A[:, :, 1:Rdim] .= A
        if noise_amp > 0 && new_chi > Rdim
            view_cols = @view new_A[:, :, (Rdim+1):new_chi]
            fill_complex_noise!(rng, view_cols, noise_amp)
        end
        mps.tensors[i] = new_A
        
        # Expand site i+1 along its left bond
        Rb, db, Rnext = size(B)
        new_B = zeros(T, new_chi, db, Rnext)
        new_B[1:Rb, :, :] .= B
        if noise_amp > 0 && new_chi > Rb
            view_rows = @view new_B[(Rb+1):new_chi, :, :]
            fill_complex_noise!(rng, view_rows, noise_amp)
        end
        mps.tensors[i+1] = new_B
    end
    
    normalize!(mps)
end

@inline function fill_complex_noise!(rng::AbstractRNG, A::AbstractArray{T}, scale::Real) where {T}
    for idx in eachindex(A)
        real_part = randn(rng)
        imag_part = randn(rng)
        A[idx] = T(scale * (real_part + imag_part * im))
    end
end

# --- Measurements ---

"""
    check_canonical_form(mps)

Returns info about canonical form.
"""
function check_canonical_form(mps::MPS)
    return [mps.orth_center]
end

"""
    norm(mps)
"""
function LinearAlgebra.norm(mps::MPS)
    c = mps.tensors[mps.orth_center]
    return norm(c)
end

"""
    scalar_product(a, b)

Compute <a|b>.
"""
function scalar_product(psi::MPS{T}, phi::MPS{T}) where T
    @assert psi.length == phi.length
    
    # E[a_bond, b_bond]
    E = ones(T, 1, 1)
    
    for i in 1:psi.length
        A = psi.tensors[i] # (La, d, Ra)
        B = phi.tensors[i] # (Lb, d, Rb)
        
        La_prev, d, Ra = size(A)
        Lb_prev, _, Rb = size(B)
        
        # Reshape A: (La_prev, d*Ra)
        A_mat = reshape(A, La_prev, d*Ra)
        
        # T1 = E^T * A_mat -> (Lb_prev, d*Ra)
        # E is (La_prev, Lb_prev).
        # We want sum_la E[la, lb] * A[la, ...] = sum_la (E^T)[lb, la] * A[la, ...]
        
        T1 = transpose(E) * A_mat 
        
        # Reshape T1: (Lb_prev, d, Ra) to match B
        # T1 is (Lb_prev, d*Ra). 
        # We need to contract with B: (Lb_prev, d, Rb).
        # Reshape T1 to (Lb_prev*d, Ra).
        # Reshape B to (Lb_prev*d, Rb).
        
        # BUT WAIT! Column major:
        # T1 (Lb, d*Ra). Layout: lb changes, then (p + d*r).
        # We want to merge (lb, p).
        # Does T1 have (lb, p) contiguous?
        # A_mat (La, d*Ra).
        # A (La, d, Ra).
        # A_mat reshape merges d and Ra.
        # So A_mat layout: la changes, then p, then r.
        # T1 layout: lb changes, then p, then r.
        
        # B (Lb, d, Rb).
        # B_mat `reshape(B, Lb*d, Rb)`.
        # B layout: lb changes, then p, then r.
        # B_mat layout: (lb, p) changes, then r.
        
        # T1 (Lb, d*Ra).
        # Reshape T1 to (Lb*d, Ra).
        # T1 layout: lb changes, then p, then r.
        # T1_reshaped layout: (lb, p) changes, then r.
        # This matches!
        
        T1_reshaped = reshape(T1, Lb_prev * d, Ra)
        B_mat = reshape(B, Lb_prev * d, Rb)
        
        # E_next = T1_reshaped^T * conj(B_mat) -> (Ra, Rb)
        E = transpose(T1_reshaped) * conj(B_mat)
    end
    
    return E[1, 1]
end

"""
    expect(mps, observable)
"""
function expect(mps::MPS, obs)
    sites = obs.sites
    gate = obs.gate.matrix
    
    if isa(sites, Int) || length(sites) == 1
        s = (isa(sites, Int)) ? sites : sites[1]
        return local_expect(mps, gate, s)
    elseif length(sites) == 2
        s1, s2 = sort(sites)
        return local_expect_two_site(mps, gate, s1, s2)
    else
        error("Only 1 or 2 site observables supported")
    end
end

"""
    local_expect(mps, op, site)
"""
function local_expect(mps::MPS{T}, op::AbstractMatrix, site::Int) where T
    shift_orthogonality_center!(mps, site)
    
    A = mps.tensors[site] # (L, d, R)
    L, d, R = size(A)
    
    # Permute A to (d, L*R)
    # A is (L, d, R). 
    # Permutedims (2, 1, 3) -> (d, L, R). Reshape to (d, L*R).
    A_perm = reshape(permutedims(A, (2, 1, 3)), d, L*R)
    
    # OpA = Op * A_perm -> (d, L*R)
    OpA = op * A_perm
    
    # Overlap
    return dot(A_perm, OpA)
end

function local_expect_two_site(mps::MPS{T}, op::AbstractMatrix, s1::Int, s2::Int) where T
    @assert s2 == s1 + 1 "Only nearest neighbor supported efficiently"
    
    shift_orthogonality_center!(mps, s1)
    
    A = mps.tensors[s1] # (L, d, k)
    B = mps.tensors[s2] # (k, d, R)
    
    L, d, k = size(A)
    _, _, R = size(B)
    
    # Form Theta (L*d, d*R) -> (L, d, d, R)
    Amat = reshape(A, L*d, k)
    Bmat = reshape(B, k, d*R)
    Theta = Amat * Bmat
    Theta = reshape(Theta, L, d, d, R)
    
    # Op is 4x4 (d1, d2, d1', d2')
    # Permute Theta to (d1, d2, L, R) -> (d*d, L*R)
    Theta_perm = reshape(permutedims(Theta, (2, 3, 1, 4)), d*d, L*R)
    
    # Apply Op
    OpTheta = op * Theta_perm
    
    return dot(Theta_perm, OpTheta)
end

"""
    evaluate_all_local_expectations(mps, operators)

Efficiently compute expectation values of a list of local operators (one per site).
"""
function evaluate_all_local_expectations(mps::MPS{T}, operators::Vector{<:AbstractMatrix}) where T
    @assert length(operators) == mps.length
    results = zeros(ComplexF64, mps.length)
    
    # We can sweep Center 1->L and measure as we go.
    shift_orthogonality_center!(mps, 1)
    
    for i in 1:mps.length
        if i > 1
            # Shift center i-1 -> i
            shift_orthogonality_center!(mps, i)
        end
        
        # Measure at i
        results[i] = local_expect(mps, operators[i], i)
    end
    
    return results
end

"""
    measure_single_shot(mps)

Return bitstring integer.
"""
function single_shot_measure(mps::MPS{T}) where T
    psi = deepcopy(mps)
    shift_orthogonality_center!(psi, 1)
    
    bits = 0
    
    for i in 1:psi.length
        A = psi.tensors[i] # (L, d, R)
        L, d, R = size(A)
        
        probs = zeros(Float64, d)
        for p in 1:d
            # Norm of slice A[:, p, :]
            # A is (L, d, R). A[:, p, :] is (L, R).
            # If L=1, it's just vector norm.
            probs[p] = real(sum(abs2, @view A[:, p, :]))
        end
        
        # Sample
        r = rand()
        accum = 0.0
        chosen = d
        for k in 1:d
            accum += probs[k]
            if r <= accum
                chosen = k
                break
            end
        end
        
        bits += (chosen - 1) << (i - 1)
        
        if i < psi.length
            # Project A on chosen: A_proj (L, R)
            A_proj = A[:, chosen, :] 
            
            nrm = sqrt(probs[chosen])
            if nrm > 1e-12
                A_proj .*= (1.0 / nrm)
            end
            
            # Contract with next
            Next = psi.tensors[i+1] # (R, d_next, R_next)
            _, dn, rn = size(Next)
            Next_mat = reshape(Next, R, dn*rn)
            
            New_Next = A_proj * Next_mat
            psi.tensors[i+1] = reshape(New_Next, L, dn, rn)
            
            psi.orth_center = i+1
        end
    end
    
    return bits
end

# Alias for backward compatibility
const measure_single_shot = single_shot_measure

function measure_shots(mps::MPS, shots::Int)
    counts = Dict{Int, Int}()
    for _ in 1:shots
        b = single_shot_measure(mps)
        counts[b] = get(counts, b, 0) + 1
    end
    return counts
end

function project_onto_bitstring(mps::MPS{T}, bitstring::String) where T
    vec = ones(T, 1) # (bond_dim)
    for i in 1:mps.length
        idx = parse(Int, bitstring[i]) + 1
        A = mps.tensors[i] # (L, d, R)
        
        L, d, R = size(A)
        A_slice = @view A[:, idx, :] # (L, R)
        
        vec = transpose(vec) * A_slice # (1, R)
        vec = reshape(vec, R)
    end
    return abs2(vec[1])
end

function to_vec(mps::MPS{T}) where T
    v = mps.tensors[1]
    for i in 2:mps.length
        T_next = mps.tensors[i]
        @tensor v_new[l, p_old, p_new, r_new] := v[l, p_old, k] * T_next[k, p_new, r_new]
        l, p_old, p_new, r_new = size(v_new)
        v = reshape(v_new, l, p_old * p_new, r_new)
    end
    return vec(v)
end

end # module
