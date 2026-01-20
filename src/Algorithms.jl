module Algorithms

using LinearAlgebra
using Printf
using TensorOperations
using KrylovKit
using Random
using Base.Threads
using ..MPSModule
using ..MPOModule
using ..SimulationConfigs
using ..Decompositions

# Timing is available when this module is loaded as part of `Yaqs`.
# Some standalone scripts `include("src/Algorithms.jl")` directly under `Main`,
# in which case `..Timing` does not exist. For those cases, fall back to a no-op `@t`.
if isdefined(parentmodule(@__MODULE__), :Timing)
    @eval using ..Timing: @t
else
    macro t(_key, ex)
        return esc(ex)
    end
end

export single_site_tdvp!, two_site_tdvp!, set_krylov_ishermitian_mode!, reset_krylov_ishermitian_cache!,
       reset_krylov_ishermitian_stats!, print_krylov_ishermitian_stats

# --- Krylov Subspace Methods ---

const _KRYLOV_ISHERMITIAN_MODE = Ref{Symbol}(:auto)  # :auto | :true | :false
const _KRYLOV_ISHERMITIAN_CACHE = Ref{Vector{Union{Nothing, Bool}}}(Vector{Union{Nothing, Bool}}(undef, 0))

# Statistics (thread-safe counters)
const _KRYLOV_CALLS_LANCZOS = Atomic{Int}(0)  # expm_krylov calls with ishermitian=true
const _KRYLOV_CALLS_ARNOLDI = Atomic{Int}(0)  # expm_krylov calls with ishermitian=false
const _KRYLOV_AUTO_DECISIONS_LANCZOS = Atomic{Int}(0)  # per-thread cache decisions in :auto
const _KRYLOV_AUTO_DECISIONS_ARNOLDI = Atomic{Int}(0)

"""
Initialize per-thread caches used by Krylov mode selection.

This sets up the thread-local cache that stores the Hermitian check decision per thread. It is
invoked when the module is loaded to ensure the cache matches the current thread count.

Args:
    None

Returns:
    Nothing: The cache is updated in-place.
"""
function __init__()
    # Ensure cache is sized to current thread count (important if precompiled with fewer threads).
    v = Vector{Union{Nothing, Bool}}(undef, Base.Threads.maxthreadid())
    fill!(v, nothing)
    _KRYLOV_ISHERMITIAN_CACHE[] = v
    return nothing
end

"""
Configure how KrylovKit decides the Hermitian mode for expm_krylov.

This controls whether the Krylov exponentiation uses a Lanczos (Hermitian) or Arnoldi (general)
subspace. The `:auto` mode performs a lightweight self-adjointness check once per thread and caches
the decision to avoid repeated overhead in tight loops.

Args:
    mode (Symbol): Mode selector, one of `:auto`, `:lanczos`, or `:arnoldi`.

Returns:
    Nothing: The global mode and cache are updated in-place.

Raises:
    AssertionError: If `mode` is not one of `:auto`, `:lanczos`, or `:arnoldi`.
"""
function set_krylov_ishermitian_mode!(mode::Symbol)
    @assert mode === :auto || mode === :lanczos || mode === :arnoldi
    _KRYLOV_ISHERMITIAN_MODE[] = mode
    reset_krylov_ishermitian_cache!()
    return nothing
end

"""
Set Krylov Hermitian mode using a Boolean switch.

This is a convenience overload that maps `true` to `:lanczos` and `false` to `:arnoldi`. It resets
the per-thread cache so the new mode takes effect immediately.

Args:
    flag (Bool): If `true` use Lanczos mode; if `false` use Arnoldi mode.

Returns:
    Nothing: The global mode and cache are updated in-place.
"""
function set_krylov_ishermitian_mode!(flag::Bool)
    return set_krylov_ishermitian_mode!(flag ? :lanczos : :arnoldi)
end

"""
Clear the per-thread cache used by auto Hermitian checks.

This resets cached decisions so each thread will re-run the self-adjointness check when using
`:auto` mode. The cache is resized if the current thread count exceeds the stored size.

Args:
    None

Returns:
    Nothing: The cache storage is updated in-place.
"""
function reset_krylov_ishermitian_cache!()
    v = _KRYLOV_ISHERMITIAN_CACHE[]
    if isempty(v) || length(v) < Base.Threads.maxthreadid()
        v2 = Vector{Union{Nothing, Bool}}(undef, Base.Threads.maxthreadid())
        fill!(v2, nothing)
        _KRYLOV_ISHERMITIAN_CACHE[] = v2
    else
        fill!(v, nothing)
    end
    return nothing
end

"""
Reset statistics for Krylov Hermitian mode decisions.

This clears counters for how many times Lanczos and Arnoldi were used, and how often auto mode
selected each option. It is useful for profiling and diagnostics.

Args:
    None

Returns:
    Nothing: All counters are reset to zero.
"""
function reset_krylov_ishermitian_stats!()
    atomic_xchg!(_KRYLOV_CALLS_LANCZOS, 0)
    atomic_xchg!(_KRYLOV_CALLS_ARNOLDI, 0)
    atomic_xchg!(_KRYLOV_AUTO_DECISIONS_LANCZOS, 0)
    atomic_xchg!(_KRYLOV_AUTO_DECISIONS_ARNOLDI, 0)
    return nothing
end

"""
Print usage statistics for Krylov Hermitian mode selection.

This reports how many times `expm_krylov` executed with Lanczos versus Arnoldi, and the number of
auto-mode decisions per thread. The output is formatted for quick diagnostic inspection.

Args:
    header (AbstractString): Optional heading text printed above the statistics.

Returns:
    Nothing: Statistics are printed to stdout.
"""
function print_krylov_ishermitian_stats(; header::AbstractString="Krylov ishermitian stats")
    # `Threads.Atomic` supports `getindex` for atomic load.
    l = _KRYLOV_CALLS_LANCZOS[]
    a = _KRYLOV_CALLS_ARNOLDI[]
    dl = _KRYLOV_AUTO_DECISIONS_LANCZOS[]
    da = _KRYLOV_AUTO_DECISIONS_ARNOLDI[]
    total = l + a
    @printf "\n\t%s\n" header
    @printf "\t  expm_krylov calls: total=%d  lanczos=%d  arnoldi=%d\n" total l a
    @printf "\t  auto decisions (per thread): lanczos=%d  arnoldi=%d\n" dl da
    return nothing
end

"""
Check whether a linear map behaves Hermitian under the Euclidean inner product.

This draws random probe vectors to test approximate self-adjointness and a real-valued quadratic
form. It is intentionally loose to avoid false negatives while still filtering obvious non-Hermitian
cases in auto mode.

Args:
    A_func: Linear map callable that applies the effective operator.
    v (AbstractArray): Prototype vector for sizing the random probes.

Returns:
    Bool: `true` if the map appears Hermitian within a loose tolerance.
"""
@inline function _ishermitian_check(A_func, v::AbstractArray{T}) where {T}
    # Diagnostic: check <x, A(y)> ≈ <A(x), y> and imag(<x,A(x)>) ≈ 0 in Euclidean inner product.
    x = randn!(similar(v, T))
    y = randn!(similar(v, T))
    Ax = A_func(x)
    Ay = A_func(y)

    lhs = dot(vec(x), vec(Ay))
    rhs = dot(vec(Ax), vec(y))
    denom = max(abs(lhs), abs(rhs), one(real(eltype(lhs))))
    rel = abs(lhs - rhs) / denom

    q = dot(vec(x), vec(Ax))
    imag_rel = abs(imag(q)) / max(abs(q), one(real(eltype(q))))

    # Loose tolerance: we only want to avoid obviously non-Hermitian cases.
    return (rel ≤ 1e-8) && (imag_rel ≤ 1e-8)
end

"""
Apply a Krylov subspace exponential to a vector-like tensor.

This computes `exp(-im * dt * A) * v` using KrylovKit with either Lanczos or Arnoldi depending on
the configured Hermitian mode. The function handles zero-norm inputs and updates diagnostic counters.

Args:
    A_func: Callable that applies the effective operator to a vector-like tensor.
    v (AbstractArray): Input vector or tensor to evolve.
    dt (Number): Time step used in the exponential.
    k (Int): Krylov subspace dimension.

Returns:
    AbstractArray: The evolved vector or tensor with the same shape as `v`.
"""
function expm_krylov(A_func, v::AbstractArray{T}, dt::Number, k::Int) where T
    norm_v = norm(v)
    if norm_v == 0
        return v
    end
        
    t_val = -1im * dt
    mode = _KRYLOV_ISHERMITIAN_MODE[]
    local isherm::Bool
    if mode === :lanczos
        isherm = true
    elseif mode === :arnoldi
        isherm = false
    else
        # :auto mode: cache once per thread to avoid paying the check cost in hot loops.
        tid = Base.Threads.threadid()
        cache = _KRYLOV_ISHERMITIAN_CACHE[]
        if tid > length(cache)
            reset_krylov_ishermitian_cache!()
            cache = _KRYLOV_ISHERMITIAN_CACHE[]
        end
        cached = cache[tid]
        if cached === nothing
            isherm = _ishermitian_check(A_func, v)
            cache[tid] = isherm
            if isherm
                atomic_add!(_KRYLOV_AUTO_DECISIONS_LANCZOS, 1)
            else
                atomic_add!(_KRYLOV_AUTO_DECISIONS_ARNOLDI, 1)
            end
        else
            isherm = cached::Bool
        end
    end

    if isherm
        atomic_add!(_KRYLOV_CALLS_LANCZOS, 1)
    else
        atomic_add!(_KRYLOV_CALLS_ARNOLDI, 1)
    end

    val, info = @t :tdvp_krylov_exponentiate exponentiate(A_func, t_val, v; tol=1e-10, krylovdim=k, maxiter=1, ishermitian=isherm)
    return val
end

# --- Environment Helpers ---

"""
Construct an identity environment tensor for MPO contractions.

This builds a 3-tensor with identity structure on the physical legs and a single active MPO index.
It is used to initialize left/right environments at the chain boundaries.

Args:
    dim (Int): Bond dimension for the bra/ket legs.
    mpo_dim (Int): MPO bond dimension for the operator leg.

Returns:
    Array{ComplexF64,3}: Identity environment tensor of shape `(dim, mpo_dim, dim)`.
"""
function make_identity_env(dim::Int, mpo_dim::Int)
    E = zeros(ComplexF64, dim, mpo_dim, dim)
    for i in 1:dim
        E[i, 1, i] = 1.0
    end
    return E
end

"""
Update the left environment by absorbing one MPS site and MPO tensor.

This contracts the current left environment with the site tensor `A` and the MPO tensor `W` to
produce the next left environment. The contraction ordering mirrors the TDVP environment build.

Args:
    A: MPS site tensor with shape `(Dl, d, Dr)`.
    W: MPO site tensor with shape `(Dl_mpo, d_out, d_in, Dr_mpo)`.
    E_left: Current left environment tensor.

Returns:
    Array{ComplexF64,3}: Updated left environment tensor.
"""
function update_left_environment(A, W, E_left)
    @t :tdvp_env_L_T1 @tensor T1[bra_l, mpo_l, p_in, ket_r] := E_left[bra_l, mpo_l, k] * A[k, p_in, ket_r]
    @t :tdvp_env_L_T2 @tensor T2[bra_l, ket_r, p_out, mpo_r] := T1[bra_l, k_ml, k_pin, ket_r] * W[k_ml, p_out, k_pin, mpo_r]
    @t :tdvp_env_L_E_next @tensor E_next[bra_r, mpo_r, ket_r] := T2[k_bl, ket_r, k_pout, mpo_r] * conj(A[k_bl, k_pout, bra_r])
    return E_next
end

"""
Update the right environment by absorbing one MPS site and MPO tensor.

This contracts the current right environment with the site tensor `A` and the MPO tensor `W` to
produce the next right environment. The contraction ordering matches the TDVP right-sweep update.

Args:
    A: MPS site tensor with shape `(Dl, d, Dr)`.
    W: MPO site tensor with shape `(Dl_mpo, d_out, d_in, Dr_mpo)`.
    E_right: Current right environment tensor.

Returns:
    Array{ComplexF64,3}: Updated right environment tensor.
"""
function update_right_environment(A, W, E_right)
    @t :tdvp_env_R_T1 @tensor T1[ket_l, p_in, bra_r, mpo_r] := A[ket_l, p_in, k] * E_right[bra_r, mpo_r, k]
    @t :tdvp_env_R_T2 @tensor T2[mpo_l, p_out, ket_l, bra_r] := W[mpo_l, p_out, k_pin, k_mr] * T1[ket_l, k_pin, bra_r, k_mr]
    @t :tdvp_env_R_E_next @tensor E_next[bra_l, mpo_l, ket_l] := T2[mpo_l, k_pout, ket_l, k_br] * conj(A[bra_l, k_pout, k_br])
    return E_next
end

# --- Projectors ---

"""
Apply the effective single-site projector to an MPS tensor.

This combines the left and right environments with the MPO tensor to project a site tensor into
the effective local action used by TDVP. It allocates intermediate tensors during the contraction.

Args:
    A: MPS site tensor to be projected.
    L: Left environment tensor.
    R: Right environment tensor.
    W: MPO tensor for the site.

Returns:
    Array{ComplexF64,3}: Projected site tensor with the same physical dimension.
"""
function project_site(A, L, R, W)
    @t :tdvp_proj_site_T1 @tensor T1[bra_l, mpo_l, p_in, ket_r] := L[bra_l, mpo_l, k] * A[k, p_in, ket_r]
    @t :tdvp_proj_site_T2 @tensor T2[bra_l, ket_r, p_out, mpo_r] := T1[bra_l, k_ml, k_pin, ket_r] * W[k_ml, p_out, k_pin, mpo_r]
    @t :tdvp_proj_site_out @tensor A_new[bra_l, p_out, bra_r] := T2[bra_l, k_kr, p_out, k_mr] * R[bra_r, k_mr, k_kr]
    return A_new
end

"""
Apply the effective bond projector to a bond matrix.

This combines left and right environments to project the center bond matrix for TDVP. It allocates
intermediate tensors during the contraction.

Args:
    C: Bond matrix to be projected.
    L: Left environment tensor.
    R: Right environment tensor.

Returns:
    Array{ComplexF64,2}: Projected bond matrix.
"""
function project_bond(C, L, R)
    @t :tdvp_proj_bond_T1 @tensor T1[bra_l, mpo, ket_r] := L[bra_l, mpo, k] * C[k, ket_r]
    @t :tdvp_proj_bond_out @tensor C_new[bra_l, bra_r] := T1[bra_l, k_mpo, k_kr] * R[bra_r, k_mpo, k_kr]
    return C_new
end

# --- Allocation-free projector workspaces for Krylov matvec ---
#
# KrylovKit will still need a fresh output object for every matvec (because it stores
# Krylov basis vectors). The goal here is to avoid *additional* temporaries (T1/T2)
# inside the matvec, by reusing per-thread buffers.

"""
Workspace for allocation-free single-site projector contractions.

This stores intermediate tensors needed by the TDVP site projector so repeated matvecs reuse
buffers instead of allocating. The workspace is thread-local and sized on demand.

Args:
    T1 (Array{T,4}): First contraction buffer.
    T2 (Array{T,4}): Second contraction buffer.

Returns:
    _ProjectSiteWS: Workspace object holding reusable buffers.
"""
mutable struct _ProjectSiteWS{T}
    T1::Array{T,4}
    T2::Array{T,4}
end

"""
Workspace for allocation-free bond projector contractions.

This stores intermediate tensors needed by the TDVP bond projector so repeated matvecs reuse
buffers instead of allocating. The workspace is thread-local and sized on demand.

Args:
    T1 (Array{T,3}): Contraction buffer for the bond projector.

Returns:
    _ProjectBondWS: Workspace object holding a reusable buffer.
"""
mutable struct _ProjectBondWS{T}
    T1::Array{T,3}
end

const _PROJECT_SITE_WS_C64 = Ref{Vector{Union{Nothing, _ProjectSiteWS{ComplexF64}}}}(Vector{Union{Nothing, _ProjectSiteWS{ComplexF64}}}(undef, 0))
const _PROJECT_BOND_WS_C64 = Ref{Vector{Union{Nothing, _ProjectBondWS{ComplexF64}}}}(Vector{Union{Nothing, _ProjectBondWS{ComplexF64}}}(undef, 0))

# Toggle for A/B benchmarking: projector workspaces in TDVP matvecs.
const _TDVP_USE_PROJECTOR_WORKSPACES = Ref{Bool}(true)

"""
Enable or disable TDVP projector workspaces for Krylov matvecs.

When enabled, per-thread workspaces reuse intermediate tensors inside Krylov matvecs to reduce
allocations. Disabling falls back to the allocating projector implementations for simplicity.

Args:
    flag (Bool): If `true`, use workspaces; if `false`, use allocating projectors.

Returns:
    Nothing: The global workspace toggle is updated in-place.
"""
function set_tdvp_projector_workspaces!(flag::Bool)
    _TDVP_USE_PROJECTOR_WORKSPACES[] = flag
    return nothing
end

"""
Report whether TDVP projector workspaces are enabled.

This exposes the current toggle that controls whether TDVP Krylov matvecs reuse thread-local
workspace buffers for projector contractions.

Args:
    None

Returns:
    Bool: `true` if workspaces are enabled, otherwise `false`.
"""
get_tdvp_projector_workspaces() = _TDVP_USE_PROJECTOR_WORKSPACES[]

"""
Convert the workspace toggle into a Val for dispatch.

This helper wraps the current workspace flag in a `Val` to enable compile-time specialization of
projector paths without branching in hot loops.

Args:
    None

Returns:
    Val{Bool}: A `Val` containing the current workspace setting.
"""
@inline _use_tdvp_ws_val() = Val(_TDVP_USE_PROJECTOR_WORKSPACES[] ? true : false)

"""
Build the single-site projector operator using workspace-backed matvecs.

This constructs a callable operator that reuses thread-local workspace buffers to reduce allocations
when applying the projector inside KrylovKit.

Args:
    L: Left environment tensor.
    R: Right environment tensor.
    W: MPO tensor for the site.

Returns:
    _ProjectSiteOpC64: Operator object wrapping the projector and workspace.
"""
@inline function _site_op(::Val{true}, L, R, W)
    return _ProjectSiteOpC64(L, R, W, _get_project_site_ws(ComplexF64))
end
"""
Build the single-site projector operator using allocating matvecs.

This constructs a callable closure that applies the projector without using workspaces, allocating
intermediate tensors on each call.

Args:
    L: Left environment tensor.
    R: Right environment tensor.
    W: MPO tensor for the site.

Returns:
    Function: Closure that applies the projector to a site tensor.
"""
@inline function _site_op(::Val{false}, L, R, W)
    return (x) -> project_site(x, L, R, W)
end

"""
Build the bond projector operator using workspace-backed matvecs.

This constructs a callable operator that reuses thread-local workspace buffers to reduce allocations
when applying the bond projector inside KrylovKit.

Args:
    L: Left environment tensor.
    R: Right environment tensor.

Returns:
    _ProjectBondOpC64: Operator object wrapping the bond projector and workspace.
"""
@inline function _bond_op(::Val{true}, L, R)
    return _ProjectBondOpC64(L, R, _get_project_bond_ws(ComplexF64))
end
"""
Build the bond projector operator using allocating matvecs.

This constructs a callable closure that applies the bond projector without using workspaces,
allocating intermediate tensors on each call.

Args:
    L: Left environment tensor.
    R: Right environment tensor.

Returns:
    Function: Closure that applies the bond projector to a bond matrix.
"""
@inline function _bond_op(::Val{false}, L, R)
    return (x) -> project_bond(x, L, R)
end

"""
Get the thread-local workspace for site projector contractions.

This ensures the workspace vector is sized for the current thread count and lazily initializes
the workspace for the calling thread if missing.

Args:
    ::Type{ComplexF64}: Element type tag for workspace selection.

Returns:
    _ProjectSiteWS{ComplexF64}: The workspace associated with the current thread.
"""
@inline function _get_project_site_ws(::Type{ComplexF64})
    v = _PROJECT_SITE_WS_C64[]
    nt = Base.Threads.maxthreadid()
    if isempty(v) || length(v) < nt
        v2 = Vector{Union{Nothing, _ProjectSiteWS{ComplexF64}}}(undef, nt)
        fill!(v2, nothing)
        if !isempty(v)
            copyto!(v2, 1, v, 1, length(v))
        end
        _PROJECT_SITE_WS_C64[] = v2
        v = v2
    end
    tid = Base.Threads.threadid()
    ws = v[tid]
    if ws === nothing
        ws = _ProjectSiteWS{ComplexF64}(Array{ComplexF64,4}(undef, 0, 0, 0, 0),
                                       Array{ComplexF64,4}(undef, 0, 0, 0, 0))
        v[tid] = ws
    end
    return ws:: _ProjectSiteWS{ComplexF64}
end

"""
Get the thread-local workspace for bond projector contractions.

This ensures the workspace vector is sized for the current thread count and lazily initializes
the workspace for the calling thread if missing.

Args:
    ::Type{ComplexF64}: Element type tag for workspace selection.

Returns:
    _ProjectBondWS{ComplexF64}: The workspace associated with the current thread.
"""
@inline function _get_project_bond_ws(::Type{ComplexF64})
    v = _PROJECT_BOND_WS_C64[]
    nt = Base.Threads.maxthreadid()
    if isempty(v) || length(v) < nt
        v2 = Vector{Union{Nothing, _ProjectBondWS{ComplexF64}}}(undef, nt)
        fill!(v2, nothing)
        if !isempty(v)
            copyto!(v2, 1, v, 1, length(v))
        end
        _PROJECT_BOND_WS_C64[] = v2
        v = v2
    end
    tid = Base.Threads.threadid()
    ws = v[tid]
    if ws === nothing
        ws = _ProjectBondWS{ComplexF64}(Array{ComplexF64,3}(undef, 0, 0, 0))
        v[tid] = ws
    end
    return ws:: _ProjectBondWS{ComplexF64}
end

"""
Ensure a buffer has the requested size, allocating if needed.

This helper returns the original array when its size matches `dims`, otherwise it allocates a new
array of the same element type and dimensionality.

Args:
    A (Array): Existing buffer to check.
    dims (NTuple{N,Int}): Desired dimensions for the buffer.

Returns:
    Array: Buffer with the requested size (may be newly allocated).
"""
@inline function _ensure_size!(A::Array{T,N}, dims::NTuple{N,Int}) where {T,N}
    if size(A) != dims
        return Array{T,N}(undef, dims...)
    end
    return A
end

"""
Apply the site projector using preallocated workspace buffers.

This performs the same contraction as `project_site` but reuses intermediate buffers stored in
`ws` to avoid allocations in Krylov matvec loops.

Args:
    A_new (Array{ComplexF64,3}): Output buffer for the projected tensor.
    ws (_ProjectSiteWS{ComplexF64}): Workspace with reusable intermediate buffers.
    A (AbstractArray{ComplexF64,3}): Input site tensor to project.
    L (AbstractArray{ComplexF64,3}): Left environment tensor.
    R (AbstractArray{ComplexF64,3}): Right environment tensor.
    W (AbstractArray{ComplexF64,4}): MPO tensor for the site.

Returns:
    Nothing: The result is written into `A_new`.
"""
@inline function _project_site_ws!(A_new::Array{ComplexF64,3},
                                  ws::_ProjectSiteWS{ComplexF64},
                                  A::AbstractArray{ComplexF64,3},
                                  L::AbstractArray{ComplexF64,3},
                                  R::AbstractArray{ComplexF64,3},
                                  W::AbstractArray{ComplexF64,4})
    # Sizes:
    #   L: (Dl_bra, mpo_l, Dl_ket)
    #   A: (Dl_ket, p_in, Dr_ket)
    #   W: (mpo_l, p_out, p_in, mpo_r)
    #   R: (Dl_bra_r, mpo_r, Dr_ket)
    Dl_bra  = size(L, 1)
    mpo_l   = size(L, 2)
    Dl_ket  = size(L, 3)
    p_in    = size(A, 2)
    Dr_ket  = size(A, 3)
    p_out   = size(W, 2)
    mpo_r   = size(W, 4)
    # NOTE: we assume dimensions are consistent (as they should be in TDVP).
    # Avoid runtime asserts here; this runs inside Krylov matvec hot loops.

    ws.T1 = _ensure_size!(ws.T1, (Dl_bra, mpo_l, p_in, Dr_ket))
    ws.T2 = _ensure_size!(ws.T2, (Dl_bra, Dr_ket, p_out, mpo_r))

    @t :tdvp_proj_site_T1 @tensor ws.T1[bra_l, mpo_l, p_in, ket_r] =
        L[bra_l, mpo_l, k] * A[k, p_in, ket_r]
    @t :tdvp_proj_site_T2 @tensor ws.T2[bra_l, ket_r, p_out, mpo_r] =
        ws.T1[bra_l, k_ml, k_pin, ket_r] * W[k_ml, p_out, k_pin, mpo_r]
    @t :tdvp_proj_site_out @tensor A_new[bra_l, p_out, bra_r] =
        ws.T2[bra_l, k_kr, p_out, k_mr] * R[bra_r, k_mr, k_kr]
    return nothing
end

"""
Apply the bond projector using a preallocated workspace buffer.

This performs the same contraction as `project_bond` but reuses the intermediate buffer stored in
`ws` to avoid allocations in Krylov matvec loops.

Args:
    C_new (Array{ComplexF64,2}): Output buffer for the projected bond matrix.
    ws (_ProjectBondWS{ComplexF64}): Workspace with reusable intermediate buffers.
    C (AbstractArray{ComplexF64,2}): Input bond matrix to project.
    L (AbstractArray{ComplexF64,3}): Left environment tensor.
    R (AbstractArray{ComplexF64,3}): Right environment tensor.

Returns:
    Nothing: The result is written into `C_new`.
"""
@inline function _project_bond_ws!(C_new::Array{ComplexF64,2},
                                  ws::_ProjectBondWS{ComplexF64},
                                  C::AbstractArray{ComplexF64,2},
                                  L::AbstractArray{ComplexF64,3},
                                  R::AbstractArray{ComplexF64,3})
    bra_l = size(L, 1)
    mpo   = size(L, 2)
    ket_l = size(L, 3)
    ket_r = size(C, 2)
    # NOTE: we assume dimensions are consistent (as they should be in TDVP).

    ws.T1 = _ensure_size!(ws.T1, (bra_l, mpo, ket_r))
    @t :tdvp_proj_bond_T1 @tensor ws.T1[bra_l, mpo, ket_r] =
        L[bra_l, mpo, k] * C[k, ket_r]
    @t :tdvp_proj_bond_out @tensor C_new[bra_l, bra_r] =
        ws.T1[bra_l, k_mpo, k_kr] * R[bra_r, k_mpo, k_kr]
    return nothing
end

"""
Callable operator that applies the site projector with ComplexF64 workspaces.

This wraps the left/right environments and MPO tensor alongside a workspace to provide a callable
object suitable for KrylovKit matvecs.

Args:
    L (AbstractArray{ComplexF64,3}): Left environment tensor.
    R (AbstractArray{ComplexF64,3}): Right environment tensor.
    W (AbstractArray{ComplexF64,4}): MPO tensor for the site.
    ws (_ProjectSiteWS{ComplexF64}): Workspace with reusable intermediate buffers.

Returns:
    _ProjectSiteOpC64: Callable projector operator.
"""
struct _ProjectSiteOpC64{TL<:AbstractArray{ComplexF64,3}, TR<:AbstractArray{ComplexF64,3}, TW<:AbstractArray{ComplexF64,4}}
    L::TL
    R::TR
    W::TW
    ws::_ProjectSiteWS{ComplexF64}
end

"""
Apply the site projector operator to an MPS site tensor.

This allocates the output tensor and uses workspace-backed contractions to compute the projected
tensor for Krylov matvecs.

Args:
    A (AbstractArray{ComplexF64,3}): Input site tensor to project.

Returns:
    Array{ComplexF64,3}: Projected site tensor.
"""
@inline function (op::_ProjectSiteOpC64)(A::AbstractArray{ComplexF64,3})
    A_new = Array{ComplexF64,3}(undef, size(op.L, 1), size(op.W, 2), size(op.R, 1))
    _project_site_ws!(A_new, op.ws, A, op.L, op.R, op.W)
    return A_new
end

"""
Callable operator that applies the bond projector with ComplexF64 workspaces.

This wraps the left/right environments alongside a workspace to provide a callable object suitable
for KrylovKit matvecs.

Args:
    L (AbstractArray{ComplexF64,3}): Left environment tensor.
    R (AbstractArray{ComplexF64,3}): Right environment tensor.
    ws (_ProjectBondWS{ComplexF64}): Workspace with reusable intermediate buffers.

Returns:
    _ProjectBondOpC64: Callable bond projector operator.
"""
struct _ProjectBondOpC64{TL<:AbstractArray{ComplexF64,3}, TR<:AbstractArray{ComplexF64,3}}
    L::TL
    R::TR
    ws::_ProjectBondWS{ComplexF64}
end

"""
Apply the bond projector operator to a bond matrix.

This allocates the output matrix and uses workspace-backed contractions to compute the projected
bond matrix for Krylov matvecs.

Args:
    C (AbstractArray{ComplexF64,2}): Input bond matrix to project.

Returns:
    Array{ComplexF64,2}: Projected bond matrix.
"""
@inline function (op::_ProjectBondOpC64)(C::AbstractArray{ComplexF64,2})
    C_new = Array{ComplexF64,2}(undef, size(op.L, 1), size(op.R, 1))
    _project_bond_ws!(C_new, op.ws, C, op.L, op.R)
    return C_new
end

# --- SVD Helper ---

"""
Split a two-site tensor into MPS factors via SVD with truncation.

This reshapes the two-site tensor into a matrix, runs an SVD, and truncates singular values using
the configured error threshold and maximum bond dimension. The truncated factors are returned for
updating adjacent MPS tensors.

Args:
    Theta: Two-site tensor with shape `(l_virt, p1, p2, r_virt)`.
    l_virt: Left virtual bond dimension.
    p1: Physical dimension for the left site.
    p2: Physical dimension for the right site.
    r_virt: Right virtual bond dimension.
    config: Time-evolution configuration containing truncation settings.

Returns:
    Tuple: `(U, S, Vt, keep)` where `U` and `Vt` are truncated factors, `S` is the singular value
        vector, and `keep` is the kept bond dimension.
"""
function split_mps_tensor_svd(Theta, l_virt, p1, p2, r_virt, config)
    # Reshape for SVD (L*p1, p2*R)
    Mat = @t :tdvp_svd_reshape reshape(Theta, l_virt*p1, p2*r_virt)
    
    # Check for NaNs/Infs which cause LAPACKException
    if @t :tdvp_svd_check_finite any(!isfinite, Mat)
        @warn "NaN or Inf detected in tensor before SVD. Replacing with zeros to avoid crash, but simulation may be compromised."
        replace!(Mat, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
    end

    # Use QRIteration for robustness against LAPACKException(1)
    F = @t :tdvp_svd svd(Mat)
    
    # Truncation
    discarded_sq = 0.0
    keep_rank = length(F.S)
    threshold = config.truncation_threshold
    min_keep = 2
    
    @t :tdvp_truncation_loop begin
        for k in length(F.S):-1:1
            discarded_sq += F.S[k]^2
            if discarded_sq >= threshold
                keep_rank = max(k, min_keep)
                break
            end
        end
    end
    
    keep = clamp(keep_rank, 1, config.max_bond_dim)
    
    U = F.U[:, 1:keep]
    S = F.S[1:keep]
    Vt = F.Vt[1:keep, :]
    
    return U, S, Vt, keep
end

# --- Main Dispatch Functions ---

"""
Evolve an MPS with one-site TDVP under a Hamiltonian MPO.

This performs a symmetric forward-and-backward sweep using half time steps to maintain second-order
accuracy. The MPS is updated in-place and remains in a consistent canonical form.

Args:
    state (MPS): State to evolve in-place.
    H (MPO): Hamiltonian MPO applied during evolution.
    config (TimeEvolutionConfig): Time step and truncation configuration.

Returns:
    Nothing: The `state` is updated in-place.
"""
function single_site_tdvp!(state::MPS, H::MPO, config::TimeEvolutionConfig)
    # Hamiltonian Simulation: Symmetric Sweep (Forward + Backward) with dt/2
    _tdvp_sweep_hamiltonian_1site!(state, H, config)
end

"""
Evolve an MPS with one-site TDVP under measurement-circuit dynamics.

This uses a single forward sweep with the measurement-config time-step conventions, mirroring the
Python circuit logic. The MPS is updated in-place without a backward sweep.

Args:
    state (MPS): State to evolve in-place.
    H (MPO): Effective circuit MPO applied during evolution.
    config (Union{MeasurementConfig, StrongMeasurementConfig}): Measurement configuration.

Returns:
    Nothing: The `state` is updated in-place.
"""
function single_site_tdvp!(state::MPS, H::MPO, config::Union{MeasurementConfig, StrongMeasurementConfig})
    # Circuit Simulation: Single Forward Sweep with dt=2 logic
    _tdvp_sweep_circuit_1site!(state, H, config)
end

"""
Evolve an MPS with two-site TDVP under a Hamiltonian MPO.

This performs a symmetric sweep with two-site updates, including a special edge step and a backward
correction sweep. The MPS is updated in-place and truncated according to the configuration.

Args:
    state (MPS): State to evolve in-place.
    H (MPO): Hamiltonian MPO applied during evolution.
    config (TimeEvolutionConfig): Time step and truncation configuration.

Returns:
    Nothing: The `state` is updated in-place.
"""
function two_site_tdvp!(state::MPS, H::MPO, config::TimeEvolutionConfig)
    # Hamiltonian Simulation: Symmetric Sweep
    _tdvp_sweep_hamiltonian_2site!(state, H, config)
end

"""
Evolve an MPS with two-site TDVP under measurement-circuit dynamics.

This performs a forward-only sweep with two-site updates following the measurement-circuit time-step
conventions. The MPS is updated in-place without a backward sweep.

Args:
    state (MPS): State to evolve in-place.
    H (MPO): Effective circuit MPO applied during evolution.
    config (Union{MeasurementConfig, StrongMeasurementConfig}): Measurement configuration.

Returns:
    Nothing: The `state` is updated in-place.
"""
function two_site_tdvp!(state::MPS, H::MPO, config::Union{MeasurementConfig, StrongMeasurementConfig})
    # Circuit Simulation: Single Forward Sweep
    _tdvp_sweep_circuit_2site!(state, H, config)
end

# --- Implementation of Sweeps ---

# 1. Hamiltonian 1-Site (Forward + Backward)
"""
Run a symmetric one-site TDVP sweep for Hamiltonian evolution.

This performs a forward sweep followed by a backward sweep with half time steps, updating site and
bond tensors while maintaining canonical forms. Environments are built and updated on the fly.

Args:
    state: MPS to evolve in-place.
    H: Hamiltonian MPO.
    config: Time evolution configuration with time step.

Returns:
    Nothing: The `state` is updated in-place.
"""
function _tdvp_sweep_hamiltonian_1site!(state, H, config)
    shift_orthogonality_center!(state, 1)
    L = state.length
    dt = config.dt
    use_ws = _use_tdvp_ws_val()
    
    # Init Envs
    E_left, E_right = _init_envs(state, H)
    
    # Forward (1 -> L)
    for i in 1:L
        W = H.tensors[i]
        func_site = _site_op(use_ws, E_left[i], E_right[i+1], W)
        
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
            func_bond = _bond_op(use_ws, E_left[i+1], E_right[i+1])
            C_new = expm_krylov(func_bond, R_mat, -dt/2, 25)
            
            @tensor Next[l, p, r] := C_new[l, k] * state.tensors[i+1][k, p, r]
            state.tensors[i+1] = Next
        end
    end
    
    # Backward (L -> 1)
    for i in L:-1:1
        W = H.tensors[i]
        func_site = _site_op(use_ws, E_left[i], E_right[i+1], W)
        
        state.tensors[i] = expm_krylov(func_site, state.tensors[i], dt/2, 25)
        
        if i > 1
            l, p, r = size(state.tensors[i])
            A_mat = reshape(state.tensors[i], l, p*r)
            F = lq(A_mat)
            Q_mat = Matrix(F.Q)
            state.tensors[i] = reshape(Q_mat, size(Q_mat, 1), p, r)
            L_mat = Matrix(F.L)
            
            E_right[i] = update_right_environment(state.tensors[i], W, E_right[i+1])
            
            func_bond = _bond_op(use_ws, E_left[i], E_right[i])
            C_new = expm_krylov(func_bond, L_mat, -dt/2, 25)
            
            @tensor Prev[l, p, r] := state.tensors[i-1][l, p, k] * C_new[k, r]
            state.tensors[i-1] = Prev
        end
    end
    println("max bond dim: ", write_max_bond_dim(state))
end

# 2. Circuit 1-Site (Forward Only, dt=2 logic)
"""
Run a forward-only one-site TDVP sweep for circuit evolution.

This uses the circuit-specific time-step logic, evolving each site forward and adjusting bonds
without a backward sweep. It mirrors the Python measurement-circuit schedule.

Args:
    state: MPS to evolve in-place.
    H: Circuit MPO.
    config: Measurement configuration.

Returns:
    Nothing: The `state` is updated in-place.
"""
function _tdvp_sweep_circuit_1site!(state, H, config)
    shift_orthogonality_center!(state, 1)
    L = state.length
    use_ws = _use_tdvp_ws_val()
    
    # Circuit Logic: dt starts at 2.0
    dt = 2.0
    
    E_left, E_right = _init_envs(state, H)
    
    # Forward Sweep (1 -> L-1)
    for i in 1:(L-1)
        W = H.tensors[i]
        func_site = _site_op(use_ws, E_left[i], E_right[i+1], W)
        
        # Evolve Site (0.5 * dt = 1.0)
        state.tensors[i] = expm_krylov(func_site, state.tensors[i], 0.5 * dt, 25)
        
        l, p, r = size(state.tensors[i])
        A_mat = reshape(state.tensors[i], l*p, r)
        F = qr(A_mat)
        state.tensors[i] = reshape(Matrix(F.Q), l, p, size(F.Q, 2))
        R_mat = Matrix(F.R)
        
        E_left[i+1] = update_left_environment(state.tensors[i], W, E_left[i])
        
        # Evolve Bond (-0.5 * dt = -1.0)
        func_bond = _bond_op(use_ws, E_left[i+1], E_right[i+1])
        C_new = expm_krylov(func_bond, R_mat, -0.5 * dt, 25)
        
        @tensor Next[l, p, r] := C_new[l, k] * state.tensors[i+1][k, p, r]
        state.tensors[i+1] = Next
    end
    
    # Final Site Update (dt becomes 1.0)
    dt = 1.0
    W = H.tensors[L]
    func_site_last = _site_op(use_ws, E_left[L], E_right[L+1], W)
    state.tensors[L] = expm_krylov(func_site_last, state.tensors[L], dt, 25)
    
    # No Backward Sweep
end

# 3. Hamiltonian 2-Site
"""
Run a symmetric two-site TDVP sweep for Hamiltonian evolution.

This performs forward two-site updates, a special edge update, and a backward correction sweep to
maintain second-order accuracy. Truncation and canonicalization are applied at each split.

Args:
    state: MPS to evolve in-place.
    H: Hamiltonian MPO.
    config: Time evolution configuration with time step and truncation settings.

Returns:
    Nothing: The `state` is updated in-place.
"""
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
"""
Run a forward-only two-site TDVP sweep for circuit evolution.

This performs two-site updates with circuit-specific time steps, including an edge update, and
skips the backward sweep entirely. The MPS is updated in-place.

Args:
    state: MPS to evolve in-place.
    H: Circuit MPO.
    config: Measurement configuration.

Returns:
    Nothing: The `state` is updated in-place.
"""
function _tdvp_sweep_circuit_2site!(state, H, config)
    @t :tdvp_shift_orth_center shift_orthogonality_center!(state, 1)
    L = state.length
    
    E_left, E_right = @t :tdvp_init_envs _init_envs(state, H)
    
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

"""
Perform a forward two-site TDVP update at bond `i`.

This merges two neighboring sites, evolves the combined tensor, splits via SVD, updates the left
environment, and optionally evolves the right site backward for symmetric steps.

Args:
    state: MPS to update in-place.
    H: MPO used in the evolution.
    E_left: Left environments array.
    E_right: Right environments array.
    i: Site index for the left site in the pair.
    dt_step: Time step for the two-site evolution.
    config: Time evolution configuration with truncation settings.
    evolve_back: Whether to apply a backward evolution to the right site.

Returns:
    Nothing: The `state` is updated in-place.
"""
function _two_site_update_forward!(state, H, E_left, E_right, i, dt_step, config, evolve_back)
    use_ws = _use_tdvp_ws_val()
    A1 = state.tensors[i]
    A2 = state.tensors[i+1]
    @t :tdvp_theta_contract @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
    
    W1 = H.tensors[i]
    W2 = H.tensors[i+1]
    @t :tdvp_mpo_merge @tensor W_merge[l, p1o, p1i, p2o, p2i, r] := W1[l, p1o, p1i, k] * W2[k, p2o, p2i, r]
    W_perm = @t :tdvp_mpo_permute permutedims(W_merge, (1, 2, 4, 3, 5, 6))
    l1, p1o, p1i, b1 = size(W1)
    l2, p2o, p2i, b2 = size(W2)
    W_group = @t :tdvp_mpo_reshape reshape(W_perm, l1, p1o*p2o, p1i*p2i, b2)
    
    l_theta, p1, p2, r_theta = size(Theta)
    Theta_group = @t :tdvp_theta_reshape reshape(Theta, l_theta, p1*p2, r_theta)
    
    # Evolve Theta
    func_two = _site_op(use_ws, E_left[i], E_right[i+2], W_group)
    Theta_new = @t :tdvp_expm_theta expm_krylov(func_two, Theta_group, dt_step, 25)
    
    # Split (Move Center Right: Keep S with V)
    Theta_split = @t :tdvp_theta_split_reshape reshape(Theta_new, l_theta, p1, p2, r_theta)
    U, S, Vt, keep = @t :tdvp_split_svd split_mps_tensor_svd(Theta_split, l_theta, p1, p2, r_theta, config)
    
    # Assign Left (U) -> Left Canonical
    state.tensors[i] = reshape(U, l_theta, p1, keep)
    
    # Update Left Env for next site
    E_left[i+1] = @t :tdvp_update_left_env update_left_environment(state.tensors[i], W1, E_left[i])
    
    # Form Right (S*V) -> Center
    A2_temp = @t :tdvp_form_A2temp reshape(Diagonal(S) * Vt, keep, p2, r_theta)
    
    if evolve_back
        func_back = _site_op(use_ws, E_left[i+1], E_right[i+2], W2)
        A2_new = @t :tdvp_expm_back expm_krylov(func_back, A2_temp, -dt_step, 25)
        state.tensors[i+1] = A2_new
    else
        state.tensors[i+1] = A2_temp
    end
end

"""
Perform the Hamiltonian edge two-site update at bond `i`.

This evolves the final bond by a full time step, splits the tensor keeping the center on the left,
and updates the right environment. It is the edge step of the symmetric two-site sweep.

Args:
    state: MPS to update in-place.
    H: Hamiltonian MPO used in the evolution.
    E_left: Left environments array.
    E_right: Right environments array.
    i: Site index for the left site in the pair.
    dt_step: Time step for the two-site evolution.
    config: Time evolution configuration with truncation settings.

Returns:
    Nothing: The `state` is updated in-place.
"""
function _two_site_update_edge_hamiltonian!(state, H, E_left, E_right, i, dt_step, config)
    # Edge Step: Evolve by dt_step. Split Left (Center moves to L-1).
    use_ws = _use_tdvp_ws_val()
    
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
    func_two = _site_op(use_ws, E_left[i], E_right[i+2], W_group)
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

"""
Perform the backward pre-correction two-site update at bond `i`.

This evolves the right site backward, merges the pair, evolves the combined tensor forward, then
splits keeping the center on the left to complete the symmetric backward sweep.

Args:
    state: MPS to update in-place.
    H: Hamiltonian MPO used in the evolution.
    E_left: Left environments array.
    E_right: Right environments array.
    i: Site index for the left site in the pair.
    dt_step: Time step for the two-site evolution.
    config: Time evolution configuration with truncation settings.

Returns:
    Nothing: The `state` is updated in-place.
"""
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
    use_ws = _use_tdvp_ws_val()
    
    func_back = _site_op(use_ws, E_left[i+1], E_right[i+2], W2)
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
    func_two = _site_op(use_ws, E_left[i], E_right[i+2], W_group)
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

"""
Perform the circuit edge two-site update at bond `i`.

This evolves the final bond by the circuit time step, splits keeping the center on the right, and
updates the left environment. It mirrors the circuit-specific edge logic in the original code.

Args:
    state: MPS to update in-place.
    H: Circuit MPO used in the evolution.
    E_left: Left environments array.
    E_right: Right environments array.
    i: Site index for the left site in the pair.
    dt_step: Time step for the two-site evolution.
    config: Measurement configuration with truncation settings.

Returns:
    Nothing: The `state` is updated in-place.
"""
function _two_site_update_edge_circuit!(state, H, E_left, E_right, i, dt_step, config)
    # Circuit Edge: Evolve by dt_step. Split Right. No Back.
    # Note: Python splits "right" here!
    # "state.tensors[i], state.tensors[i+1] = split_mps_tensor(..., "right", ...)"
    # So Center stays at Right (i+1).
    use_ws = _use_tdvp_ws_val()
    
    A1 = state.tensors[i]
    A2 = state.tensors[i+1]
    @t :tdvp_theta_contract @tensor Theta[l, p1, p2, r] := A1[l, p1, k] * A2[k, p2, r]
    
    W1 = H.tensors[i]
    W2 = H.tensors[i+1]
    @t :tdvp_mpo_merge @tensor W_merge[l, p1o, p1i, p2o, p2i, r] := W1[l, p1o, p1i, k] * W2[k, p2o, p2i, r]
    W_perm = @t :tdvp_mpo_permute permutedims(W_merge, (1, 2, 4, 3, 5, 6))
    l1, p1o, p1i, b1 = size(W1)
    l2, p2o, p2i, b2 = size(W2)
    W_group = @t :tdvp_mpo_reshape reshape(W_perm, l1, p1o*p2o, p1i*p2i, b2)
    
    l_theta, p1, p2, r_theta = size(Theta)
    Theta_group = @t :tdvp_theta_reshape reshape(Theta, l_theta, p1*p2, r_theta)
    
    # Evolve Theta
    func_two = _site_op(use_ws, E_left[i], E_right[i+2], W_group)
    Theta_new = @t :tdvp_expm_theta expm_krylov(func_two, Theta_group, dt_step, 25)
    
    # Split Right (Center moves to i+1)
    Theta_split = @t :tdvp_theta_split_reshape reshape(Theta_new, l_theta, p1, p2, r_theta)
    U, S, Vt, keep = @t :tdvp_split_svd split_mps_tensor_svd(Theta_split, l_theta, p1, p2, r_theta, config)
    
    # Assign Left (U) -> Left Canonical
    state.tensors[i] = reshape(U, l_theta, p1, keep)
    
    # Update Left Env
    E_left[i+1] = @t :tdvp_update_left_env update_left_environment(state.tensors[i], W1, E_left[i])
    
    # Assign Right (S*V) -> Center
    state.tensors[i+1] = @t :tdvp_form_A2temp reshape(Diagonal(S) * Vt, keep, p2, r_theta)
end

"""
Initialize left and right environment tensors for TDVP sweeps.

This builds the boundary identity environments and precomputes right environments by sweeping from
the end of the chain. The left environments are initialized with the left boundary identity.

Args:
    state: MPS whose tensors define bond dimensions.
    H: MPO whose tensors define operator bond dimensions.

Returns:
    Tuple: `(E_left, E_right)` environment arrays for all sites.
"""
function _init_envs(state, H)
    L = state.length
    E_right = Vector{Array{ComplexF64, 3}}(undef, L+1)
    r_bond = size(state.tensors[L], 3)
    r_mpo = size(H.tensors[L], 4)
    E_right[L+1] = @t :tdvp_env_make_id make_identity_env(r_bond, r_mpo)
    
    for i in L:-1:2
        E_right[i] = @t :tdvp_env_update_right update_right_environment(state.tensors[i], H.tensors[i], E_right[i+1])
    end
    
    E_left = Vector{Array{ComplexF64, 3}}(undef, L+1)
    l_bond = size(state.tensors[1], 1)
    l_mpo = size(H.tensors[1], 1)
    E_left[1] = @t :tdvp_env_make_id make_identity_env(l_bond, l_mpo)
    
    return E_left, E_right
end

end # module
