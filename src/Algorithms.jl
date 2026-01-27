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
       reset_krylov_ishermitian_stats!, print_krylov_ishermitian_stats,
       Cutoff, FixedDimension, no_truncation, random_contraction

# --- Krylov Subspace Methods ---

const _KRYLOV_ISHERMITIAN_MODE = Ref{Symbol}(:auto)  # :auto | :true | :false
const _KRYLOV_ISHERMITIAN_CACHE = Ref{Vector{Union{Nothing, Bool}}}(Vector{Union{Nothing, Bool}}(undef, 0))

# Statistics (thread-safe counters)
const _KRYLOV_CALLS_LANCZOS = Atomic{Int}(0)  # expm_krylov calls with ishermitian=true
const _KRYLOV_CALLS_ARNOLDI = Atomic{Int}(0)  # expm_krylov calls with ishermitian=false
const _KRYLOV_AUTO_DECISIONS_LANCZOS = Atomic{Int}(0)  # per-thread cache decisions in :auto
const _KRYLOV_AUTO_DECISIONS_ARNOLDI = Atomic{Int}(0)

function __init__()
    # Ensure cache is sized to current thread count (important if precompiled with fewer threads).
    v = Vector{Union{Nothing, Bool}}(undef, Base.Threads.maxthreadid())
    fill!(v, nothing)
    _KRYLOV_ISHERMITIAN_CACHE[] = v
    return nothing
end

"""
    set_krylov_ishermitian_mode!(mode::Symbol)

Controls whether KrylovKit uses Hermitian mode (Lanczos) or general mode (Arnoldi) in `expm_krylov`.

- `:auto`  : run a quick self-adjointness check once per thread and use Lanczos only if it passes.
- `:lanczos`  : force Lanczos (`ishermitian=true`) (unsafe if the effective map is not self-adjoint).
- `:arnoldi`  : force Arnoldi (`ishermitian=false`) (always safe).

You can also call `set_krylov_ishermitian_mode!(flag::Bool)`:
- `true`  => `:lanczos`
- `false` => `:arnoldi`
"""
function set_krylov_ishermitian_mode!(mode::Symbol)
    @assert mode === :auto || mode === :lanczos || mode === :arnoldi
    _KRYLOV_ISHERMITIAN_MODE[] = mode
    reset_krylov_ishermitian_cache!()
    return nothing
end

function set_krylov_ishermitian_mode!(flag::Bool)
    return set_krylov_ishermitian_mode!(flag ? :lanczos : :arnoldi)
end

"""
    reset_krylov_ishermitian_cache!()

Clear the per-thread cached decision used by `:auto` mode.
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
    reset_krylov_ishermitian_stats!()

Reset counters tracking how often `expm_krylov` ran in Lanczos vs Arnoldi mode.
"""
function reset_krylov_ishermitian_stats!()
    atomic_xchg!(_KRYLOV_CALLS_LANCZOS, 0)
    atomic_xchg!(_KRYLOV_CALLS_ARNOLDI, 0)
    atomic_xchg!(_KRYLOV_AUTO_DECISIONS_LANCZOS, 0)
    atomic_xchg!(_KRYLOV_AUTO_DECISIONS_ARNOLDI, 0)
    return nothing
end

"""
    print_krylov_ishermitian_stats(; header="Krylov ishermitian stats")

Print how often `expm_krylov` used Lanczos (`ishermitian=true`) vs Arnoldi (`false`).
Also prints how many times `:auto` mode decided for each (one decision per thread).
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
    expm_krylov(A_func, v, dt, k)

Compute exp(-im * dt * A) * v using Krylov subspace.
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

function make_identity_env(dim::Int, mpo_dim::Int)
    E = zeros(ComplexF64, dim, mpo_dim, dim)
    for i in 1:dim
        E[i, 1, i] = 1.0
    end
    return E
end

function update_left_environment(A, W, E_left)
    @t :tdvp_env_L_T1 @tensor T1[bra_l, mpo_l, p_in, ket_r] := E_left[bra_l, mpo_l, k] * A[k, p_in, ket_r]
    @t :tdvp_env_L_T2 @tensor T2[bra_l, ket_r, p_out, mpo_r] := T1[bra_l, k_ml, k_pin, ket_r] * W[k_ml, p_out, k_pin, mpo_r]
    @t :tdvp_env_L_E_next @tensor E_next[bra_r, mpo_r, ket_r] := T2[k_bl, ket_r, k_pout, mpo_r] * conj(A[k_bl, k_pout, bra_r])
    return E_next
end

function update_right_environment(A, W, E_right)
    @t :tdvp_env_R_T1 @tensor T1[ket_l, p_in, bra_r, mpo_r] := A[ket_l, p_in, k] * E_right[bra_r, mpo_r, k]
    @t :tdvp_env_R_T2 @tensor T2[mpo_l, p_out, ket_l, bra_r] := W[mpo_l, p_out, k_pin, k_mr] * T1[ket_l, k_pin, bra_r, k_mr]
    @t :tdvp_env_R_E_next @tensor E_next[bra_l, mpo_l, ket_l] := T2[mpo_l, k_pout, ket_l, k_br] * conj(A[bra_l, k_pout, k_br])
    return E_next
end

# --- Projectors ---

function project_site(A, L, R, W)
    @t :tdvp_proj_site_T1 @tensor T1[bra_l, mpo_l, p_in, ket_r] := L[bra_l, mpo_l, k] * A[k, p_in, ket_r]
    @t :tdvp_proj_site_T2 @tensor T2[bra_l, ket_r, p_out, mpo_r] := T1[bra_l, k_ml, k_pin, ket_r] * W[k_ml, p_out, k_pin, mpo_r]
    @t :tdvp_proj_site_out @tensor A_new[bra_l, p_out, bra_r] := T2[bra_l, k_kr, p_out, k_mr] * R[bra_r, k_mr, k_kr]
    return A_new
end

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

mutable struct _ProjectSiteWS{T}
    T1::Array{T,4}
    T2::Array{T,4}
end

mutable struct _ProjectBondWS{T}
    T1::Array{T,3}
end

const _PROJECT_SITE_WS_C64 = Ref{Vector{Union{Nothing, _ProjectSiteWS{ComplexF64}}}}(Vector{Union{Nothing, _ProjectSiteWS{ComplexF64}}}(undef, 0))
const _PROJECT_BOND_WS_C64 = Ref{Vector{Union{Nothing, _ProjectBondWS{ComplexF64}}}}(Vector{Union{Nothing, _ProjectBondWS{ComplexF64}}}(undef, 0))

# Toggle for A/B benchmarking: projector workspaces in TDVP matvecs.
const _TDVP_USE_PROJECTOR_WORKSPACES = Ref{Bool}(true)

"""
    set_tdvp_projector_workspaces!(flag::Bool)

Enable/disable per-thread scratch workspaces for the TDVP projector matvecs used by KrylovKit.

When `true` (default), intermediate tensors are reused (fewer allocations, faster).
When `false`, fall back to the simpler allocating `project_site`/`project_bond` implementations.
"""
function set_tdvp_projector_workspaces!(flag::Bool)
    _TDVP_USE_PROJECTOR_WORKSPACES[] = flag
    return nothing
end

get_tdvp_projector_workspaces() = _TDVP_USE_PROJECTOR_WORKSPACES[]

@inline _use_tdvp_ws_val() = Val(_TDVP_USE_PROJECTOR_WORKSPACES[] ? true : false)

@inline function _site_op(::Val{true}, L, R, W)
    return _ProjectSiteOpC64(L, R, W, _get_project_site_ws(ComplexF64))
end
@inline function _site_op(::Val{false}, L, R, W)
    return (x) -> project_site(x, L, R, W)
end

@inline function _bond_op(::Val{true}, L, R)
    return _ProjectBondOpC64(L, R, _get_project_bond_ws(ComplexF64))
end
@inline function _bond_op(::Val{false}, L, R)
    return (x) -> project_bond(x, L, R)
end

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

@inline function _ensure_size!(A::Array{T,N}, dims::NTuple{N,Int}) where {T,N}
    if size(A) != dims
        return Array{T,N}(undef, dims...)
    end
    return A
end

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

struct _ProjectSiteOpC64{TL<:AbstractArray{ComplexF64,3}, TR<:AbstractArray{ComplexF64,3}, TW<:AbstractArray{ComplexF64,4}}
    L::TL
    R::TR
    W::TW
    ws::_ProjectSiteWS{ComplexF64}
end

@inline function (op::_ProjectSiteOpC64)(A::AbstractArray{ComplexF64,3})
    A_new = Array{ComplexF64,3}(undef, size(op.L, 1), size(op.W, 2), size(op.R, 1))
    _project_site_ws!(A_new, op.ws, A, op.L, op.R, op.W)
    return A_new
end

struct _ProjectBondOpC64{TL<:AbstractArray{ComplexF64,3}, TR<:AbstractArray{ComplexF64,3}}
    L::TL
    R::TR
    ws::_ProjectBondWS{ComplexF64}
end

@inline function (op::_ProjectBondOpC64)(C::AbstractArray{ComplexF64,2})
    C_new = Array{ComplexF64,2}(undef, size(op.L, 1), size(op.R, 1))
    _project_bond_ws!(C_new, op.ws, C, op.L, op.R)
    return C_new
end

# --- SVD Helper ---

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
    # Truncation mode:
    # - if `threshold >= 0`: interpret as **relative discarded weight**
    #   sum(discarded S^2) / sum(all S^2) >= threshold
    #   (matches TenPy's `trunc_cut` convention on normalized Schmidt values)
    # - if `threshold < 0`: interpret as **absolute discarded weight** (legacy)
    #   sum(discarded S^2) >= -threshold
    threshold = config.truncation_threshold
    total_sq = sum(abs2, F.S)
    min_keep = 2
    
    @t :tdvp_truncation_loop begin
        discarded_sq = 0.0
        keep_rank = length(F.S)
        for k in length(F.S):-1:1
            discarded_sq += F.S[k]^2
            if threshold < 0
                if discarded_sq >= -threshold
                    keep_rank = max(k, min_keep)
                    break
                end
            else
                frac = (total_sq == 0.0) ? 0.0 : (discarded_sq / total_sq)
                if frac >= threshold
                    keep_rank = max(k, min_keep)
                    break
                end
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

# ==============================================================================
# Successive Randomized Compression (SRC) MPO-MPS product
# Port of `random_contraction` from:
#   third_party/RandomMPOMPS/code/tensornetwork/contraction.py
#
# Notes:
# - Python code relies heavily on reshape order in NumPy (row-major). In Julia
#   (column-major), we implement the *same algebra* using explicit tensor
#   contractions to avoid subtle index-linearization bugs.
# - We assume square local operators: `d_out == d_in == psi.phys_dims[i]`.
# ==============================================================================

"""
    StoppingRule(outputdim, mindim, maxdim, cutoff)

Stopping rule for SRC contraction (mirrors Python `StoppingRule`).

- `outputdim::Union{Nothing,Int}`: fixed output bond dimension (if set).
- `mindim::Union{Nothing,Int}`: minimum sketch dimension (adaptive mode).
- `maxdim::Int`: maximum sketch dimension cap (adaptive mode).
- `cutoff::Union{Nothing,Float64}`: relative tolerance for adaptive termination.
"""
struct StoppingRule
    outputdim::Union{Nothing, Int}
    mindim::Union{Nothing, Int}
    maxdim::Int
    cutoff::Union{Nothing, Float64}
end

"""
    Cutoff(cutoff; mindim=1, maxdim=typemax(Int))

Adaptive SRC stopping rule with relative cutoff `cutoff`.
"""
Cutoff(cutoff::Real; mindim::Int=1, maxdim::Int=typemax(Int)) =
    StoppingRule(nothing, mindim, maxdim, Float64(cutoff))

"""
    FixedDimension(dim)

Fixed-dimension SRC stopping rule (no adaptive error estimation).
"""
FixedDimension(dim::Int) = StoppingRule(dim, 1, typemax(Int), nothing)

"""
    no_truncation()

SRC stopping rule that disables truncation/error-based early stopping.
"""
no_truncation() = StoppingRule(nothing, nothing, typemax(Int), nothing)

@inline _is_truncation(stop::StoppingRule) = !((stop.outputdim === nothing) && (stop.cutoff === nothing))

@inline function _maxlinkdim_mps(psi::MPS)
    maxχ = 1
    for A in psi.tensors
        maxχ = max(maxχ, size(A, 1), size(A, 3))
    end
    return maxχ
end

@inline function _maxlinkdim_mpo(H::MPO)
    maxχ = 1
    for W in H.tensors
        maxχ = max(maxχ, size(W, 1), size(W, 4))
    end
    return maxχ
end

@inline function _randn_real_as_T!(rng::AbstractRNG, x::Vector{T}) where {T}
    @inbounds for i in eachindex(x)
        x[i] = T(randn(rng))
    end
    return x
end

"""
    random_contraction(H::MPO, psi::MPS;
        stop=Cutoff(1e-6), sketchdim=1, sketchincrement=1,
        finalround=nothing, accuracychecks=false, rng=Random.default_rng())

Compute a compressed approximation to the MPO–MPS product `H * psi` using
Successive Randomized Compression (SRC) (Camaño–Epperly–Tropp, 2025).

This is a direct Julia port of the Python reference implementation’s
`random_contraction` routine.

Returns an MPS in **left-canonical** form (orthogonality center at `L`).
"""
function random_contraction(H::MPO{T},
                            psi::MPS{T};
                            stop::StoppingRule=Cutoff(1e-6),
                            sketchdim::Int=1,
                            sketchincrement::Int=1,
                            finalround=nothing,
                            accuracychecks::Bool=false,
                            rng::AbstractRNG=Random.default_rng()) where {T<:Number}
    n = H.length
    @assert n == psi.length "lengths of MPO and MPS do not match"
    n == 1 && throw(ArgumentError("MPO-MPS product for n=1 is not implemented"))

    # Physical dimension (assume uniform and square local operators)
    d = H.phys_dims[1]
    @assert all(==(d), H.phys_dims) "SRC currently assumes uniform physical dimension in MPO"
    @assert all(==(d), psi.phys_dims) "SRC currently assumes uniform physical dimension in MPS"

    # Stopping parameters (mirror Python defaults)
    maxdim = stop.maxdim
    outdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outdim === nothing
        # If user left maxdim "effectively infinite", cap it like the Python code does when maxdim=None.
        if maxdim == typemax(Int)
            maxdim = _maxlinkdim_mpo(H) * _maxlinkdim_mps(psi)
        end
        mindim = max(something(mindim, 1), 1)
    else
        maxdim = outdim
        mindim = outdim
        sketchdim = outdim
    end

    # --- Boundary/bulk views in "Python algebra" index order ---
    # MPO layout used by the Python algorithm:
    #   first: (d_out, D_right, d_in)
    #   bulk : (D_left, d_out, D_right, d_in)
    #   last : (D_left, d_out, d_in)
    #
    # Our MPO layout:
    #   (D_left, d_out, d_in, D_right)
    #
    # MPS in Python:
    #   first: (d, χR)
    #   bulk : (χL, d, χR)
    #   last : (χL, d)
    #
    # Our MPS layout is always (χL, d, χR) with χL/χR possibly 1.
    @views psi_first = psi.tensors[1][1, :, :]          # (d, χ2)
    @views psi_last  = psi.tensors[n][:, :, 1]          # (χ_{n}, d)
    psi_bulk = (n > 2) ? psi.tensors[2:(n-1)] : Array{T,3}[]

    @views Hfirst_raw = H.tensors[1][1, :, :, :]        # (d_out, d_in, D2)
    H_first = permutedims(Hfirst_raw, (1, 3, 2))        # (d_out, D2, d_in)
    @views Hlast_raw  = H.tensors[n][:, :, :, 1]        # (D_n, d_out, d_in)
    H_last = Array{T,3}(Hlast_raw)                      # ensure dense
    H_bulk = Vector{Array{T,4}}(undef, max(n - 2, 0))    # sites 2..n-1
    for site in 2:(n-1)
        H_bulk[site - 1] = permutedims(H.tensors[site], (1, 2, 4, 3)) # (Dl, d_out, Dr, d_in)
    end

    # Output tensors (Julia MPS layout (χL, d, χR))
    psi_out = Vector{Array{T,3}}(undef, n)

    # Cached left environments for each sketch column.
    # envs[idx][k] is a matrix corresponding to the contraction up to site (k+1) (1-based sites):
    # k=1 corresponds to site 1, k=2 to site 2, ... k=j-1 to site j-1.
    envs = Vector{Vector{Matrix{T}}}()

    # "Cap" tensor carrying the right-side compression information.
    # Shape: (cap_dim, Dl_next, χL_next) where Dl_next is the left MPO bond at the next site
    # and χL_next is the left MPS bond at the next site (i.e. previous site’s χR).
    cap = Array{T,3}(undef, 1, 1, 1)
    cap[1, 1, 1] = one(T)
    cap_dim = 1

    visible_dim = d
    x = Vector{T}(undef, visible_dim)

    # Main right-to-left sweep: sites n, n-1, ..., 2
    for j in n:-1:2
        # Dimension heuristics (match Python)
        local prod_bond_dims::Int
        if j == n
            prod_bond_dims = size(H_last, 1) * size(psi_last, 1)
        else
            Hj = H_bulk[j - 1]
            psij = psi_bulk[j - 1]
            prod_bond_dims = max(size(Hj, 1) * size(psij, 1),
                                 size(Hj, 3) * size(psij, 3))
        end

        current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)
        current_mindim = min(mindim, current_maxdim)
        current_sketchdim = max(min(sketchdim, current_maxdim), current_mindim)

        sketches_complete = 0
        sketch = (j == n) ? zeros(T, visible_dim, current_sketchdim) : zeros(T, visible_dim, cap_dim, current_sketchdim)

        while true
            # --- 1) Build any missing environments up to current_sketchdim ---
            for idx in (length(envs) + 1):current_sketchdim
                # Environments only need sites 1..(j-1), but since the sweep starts at j=n,
                # the first time we build envs we end up constructing the full chain (1..n-1),
                # which is sufficient for later (smaller j) steps.
                env_len = j - 1
                env = Vector{Matrix{T}}(undef, env_len)

                # Site 1
                _randn_real_as_T!(rng, x)
                Dr1 = size(H_first, 2)
                din = size(H_first, 3)
                temp1 = Array{T,2}(undef, Dr1, din)
                @tensor temp1[Dr, pin] := H_first[pout, Dr, pin] * x[pout]
                env1 = Array{T,2}(undef, Dr1, size(psi_first, 2))
                mul!(env1, temp1, psi_first)
                env[1] = env1

                # Sites 2..(j-1)
                for site in 2:(j - 1)
                    _randn_real_as_T!(rng, x)
                    Hs = H_bulk[site - 1]     # (Dl, do, Dr, di)
                    psis = psi_bulk[site - 1] # (χL, di, χR)
                    Dl = size(Hs, 1)
                    Dr = size(Hs, 3)
                    di = size(Hs, 4)

                    # tempH[Dl,Dr,di] = Σ_do Hs[Dl,do,Dr,di] * x[do]
                    tempH = Array{T,3}(undef, Dl, Dr, di)
                    @tensor tempH[Dl_, Dr_, pin_] := Hs[Dl_, pout, Dr_, pin_] * x[pout]

                    # res[Dr,di,χ] = Σ_Dl tempH[Dl,Dr,di] * env_prev[Dl,χ]
                    env_prev = env[site - 1]
                    χprev = size(env_prev, 2)
                    res = Array{T,3}(undef, Dr, di, χprev)
                    @tensor res[Dr_, di_, χ_] := tempH[Dl_, Dr_, di_] * env_prev[Dl_, χ_]

                    # env_next[Dr,χR] = Σ_{di,χ} res[Dr,di,χ] * psis[χ,di,χR]
                    χR = size(psis, 3)
                    env_next = Array{T,2}(undef, Dr, χR)
                    @tensor env_next[Dr_, χR_] := res[Dr_, di_, χ_] * psis[χ_, di_, χR_]
                    env[site] = env_next
                end

                push!(envs, env)
            end

            # --- 2) Form any missing sketch columns ---
            for idx in (sketches_complete + 1):current_sketchdim
                if j == n
                    # sketch[:, idx] = contraction of env(site n-1), psi_last, H_last
                    env_prev = envs[idx][j - 1] # (Dl_last, χL_last)
                    mat = env_prev * psi_last   # (Dl_last, di)
                    y = Vector{T}(undef, visible_dim)
                    @tensor y[pout] := H_last[Dl, pout, pin] * mat[Dl, pin]
                    sketch[:, idx] .= y
                else
                    Hj = H_bulk[j - 1]       # (Dl, do, Dr, di)
                    psij = psi_bulk[j - 1]   # (χL, di, χR)
                    env_prev = envs[idx][j - 1] # (Dl, χL)

                    # t[Dl,di,χR] = Σ_χL env_prev[Dl,χL] * psij[χL,di,χR]
                    Dl = size(Hj, 1)
                    Dr = size(Hj, 3)
                    di = size(Hj, 4)
                    χR = size(psij, 3)
                    t = Array{T,3}(undef, Dl, di, χR)
                    @tensor t[Dl_, di_, χR_] := env_prev[Dl_, χL] * psij[χL, di_, χR_]

                    # u[do,Dr,χR] = Σ_{Dl,di} Hj[Dl,do,Dr,di] * t[Dl,di,χR]
                    u = Array{T,3}(undef, visible_dim, Dr, χR)
                    @tensor u[pout, Dr_, χR_] := Hj[Dl_, pout, Dr_, pin_] * t[Dl_, pin_, χR_]

                    # v[do,β] = Σ_{Dr,χR} u[do,Dr,χR] * cap[β,Dr,χR]
                    v = Array{T,2}(undef, visible_dim, cap_dim)
                    @tensor v[pout, β] := u[pout, Dr_, χR_] * cap[β, Dr_, χR_]
                    sketch[:, :, idx] .= v
                end
            end
            sketches_complete = current_sketchdim

            # --- 3) QR and (optional) error estimate ---
            local Qmat::Matrix{T}
            local Rmat::Matrix{T}
            local Qten::Union{Nothing, Array{T,3}} = nothing

            if j == n
                F = qr(sketch)
                Qmat = Matrix(F.Q)
                Rmat = Matrix(F.R)
            else
                M = reshape(sketch, visible_dim * cap_dim, current_sketchdim)
                F = qr(M)
                Qmat = Matrix(F.Q)
                Rmat = Matrix(F.R)
                Qten = reshape(Qmat, visible_dim, cap_dim, current_sketchdim)
            end

            done = false
            if outdim !== nothing
                done = true
            elseif cutoff === nothing
                done = (current_sketchdim == current_maxdim)
            else
                if current_sketchdim == current_maxdim
                    done = true
                else
                    # Python: norm_est = ||sketch|| / sqrt(r)
                    #         G = inv(R.T)
                    #         err_est = sqrt(sum(norm(G, axis=0)^(-2)) / r)
                    r = current_sketchdim
                    norm_est = norm((j == n) ? sketch : reshape(sketch, visible_dim * cap_dim, r)) / sqrt(r)
                    # `Rmat` can be (near-)singular if the current sketch columns are not
                    # linearly independent. In that case, the Python reference would also
                    # fail at `inv`. We treat that as "not done" and request more sketches.
                    local ok_inv = true
                    local err_est = Inf
                    try
                        G = inv(transpose(Rmat))
                        coln = vec(sum(abs2, G; dims=1)).^(0.5)
                        err_est = sqrt(sum((coln .^ (-2))) / r)
                    catch e
                        ok_inv = false
                    end
                    done = ok_inv && (err_est <= cutoff * norm_est) && (r >= current_mindim)
                end
            end

            if done
                # --- 4) Store output tensor at site j and update cap ---
                if j == n
                    # psi_out[n] : (χL, d, 1)
                    psi_out[j] = reshape(transpose(Qmat), current_sketchdim, visible_dim, 1)

                    # cap[α,Dl,χL] = Σ_{do,di} conj(Q[do,α]) * H_last[Dl,do,di] * psi_last[χL,di]
                    cap_new = Array{T,3}(undef, current_sketchdim, size(H_last, 1), size(psi_last, 1))
                    @tensor cap_new[α, Dl, χ] := conj(Qmat[pout, α]) * H_last[Dl, pout, pin] * psi_last[χ, pin]
                    cap = cap_new
                    cap_dim = current_sketchdim
                else
                    @assert Qten !== nothing
                    # Qten: (do, cap_dim, r) -> output tensor (r, do, cap_dim)
                    psi_out[j] = permutedims(Qten::Array{T,3}, (3, 1, 2))

                    Hj = H_bulk[j - 1]     # (Dl, do, Dr, di)
                    psij = psi_bulk[j - 1] # (χL, di, χR)

                    # cap_new[α,Dl,χL] = Σ conj(Q[do,β,α]) * cap[β,Dr,χR] * Hj[Dl,do,Dr,di] * psij[χL,di,χR]
                    cap_new = Array{T,3}(undef, current_sketchdim, size(Hj, 1), size(psij, 1))
                    @tensor cap_new[α, Dl, χL] :=
                        conj((Qten::Array{T,3})[pout, β, α]) *
                        cap[β, Dr, χR] *
                        Hj[Dl, pout, Dr, pin] *
                        psij[χL, pin, χR]
                    cap = cap_new
                    cap_dim = current_sketchdim
                end

                # Optional accuracy checks (not yet ported)
                if accuracychecks
                    @warn "accuracychecks=true is not implemented in Julia SRC port (skipping)."
                end

                break
            end

            # --- 5) Increase sketch dimension and continue ---
            current_sketchdim = min(current_maxdim, current_sketchdim + sketchincrement)
            if j == n
                old = sketch
                sketch = zeros(T, visible_dim, current_sketchdim)
                sketch[:, 1:size(old, 2)] .= old
            else
                old = sketch
                sketch = zeros(T, visible_dim, cap_dim, current_sketchdim)
                sketch[:, :, 1:size(old, 3)] .= old
            end
        end
    end

    # Final left boundary (site 1):
    # temp[α,Dr,di] = Σ_χ cap[α,Dr,χ] * psi_first[di,χ]
    Dr1 = size(H_first, 2)
    di = size(H_first, 3)
    tmp = Array{T,3}(undef, cap_dim, Dr1, di)
    @tensor tmp[α, Dr, di_] := cap[α, Dr, χ] * psi_first[di_, χ]

    A1mat = Array{T,2}(undef, visible_dim, cap_dim)
    @tensor A1mat[pout, α] := H_first[pout, Dr, pin_] * tmp[α, Dr, pin_]
    psi_out[1] = reshape(A1mat, 1, visible_dim, cap_dim)

    out = MPS(n, psi_out, psi.phys_dims, n)

    # Optional final rounding/compression: interpret `finalround` as (threshold, max_bond_dim)
    if finalround !== nothing
        if finalround isa NamedTuple
            thr = get(finalround, :threshold, 1e-12)
            mbd = get(finalround, :max_bond_dim, nothing)
            MPSModule.truncate!(out; threshold=float(thr), max_bond_dim=mbd)
        else
            @warn "finalround provided but unsupported type ($(typeof(finalround))); skipping."
        end
    end

    return out
end

end # module
