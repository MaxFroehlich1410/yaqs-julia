"""
    BenchmarkReference

Memory-efficient exact-reference backend for spin-1/2 models (TFIM, XXZ/general,
Haldane–Shastry) using **matrix-free Krylov exponential propagation**.

This module:
  * never materialises the dense Hamiltonian,
  * never calls `eigen` on a 2^N × 2^N matrix,
  * propagates via `KrylovKit.exponentiate` with a callable linear map,
  * exploits magnetisation-sector reduction for Sz-conserving models.

Basis convention:
  site 1 = bit 0 (LSB);  basis index `x ∈ 0:(2^N-1)`, stored at Julia index `x+1`.

Feasibility:
  N ≤ 20  full-space Krylov     — practical (state vector ≤ 16 MB)
  N ≤ 24  sector Krylov (XXZ/HS) — feasible if sector dim is moderate
  Beyond that, even storing the state vector becomes prohibitive.
"""
module BenchmarkReference

using Printf
using LinearAlgebra
using Serialization
using KrylovKit

export compute_or_load_reference, ReferenceData, KrylovConfig, validate_against_dense

# ═══════════════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════════════

struct ReferenceData
    t_grid::Vector{Float64}
    psi_T::Vector{ComplexF64}          # full statevector at final time (length 2^N)
    z_expect::Matrix{Float64}          # (N, n_times)
    energy::Vector{Float64}            # (n_times,)  — constant for unitary evolution
    norm_vals::Vector{Float64}         # (n_times,)  — ‖ψ‖² at each snapshot
end

"""
    KrylovConfig(; krylovdim=30, maxiter=100, tol=1e-12, backend=:krylovkit)

Settings for the Krylov exponential propagator.
`backend` is reserved for future use (e.g. `:exponentialutilities`).
"""
struct KrylovConfig
    krylovdim::Int
    maxiter::Int
    tol::Float64
    backend::Symbol
end
function KrylovConfig(; krylovdim::Int=30, maxiter::Int=100,
                        tol::Float64=1e-12, backend::Symbol=:krylovkit)
    return KrylovConfig(krylovdim, maxiter, tol, backend)
end

# ═══════════════════════════════════════════════════════════════════════
#  Magnetisation sector (Sz-conserving models)
# ═══════════════════════════════════════════════════════════════════════

"""
    MagnetizationSector(N, n_down)

Enumerate all computational-basis states of `N` qubits with exactly `n_down`
bits set (= number of spin-↓ sites).  Provides O(1) reverse lookup from a
full-basis index to its position in the sector.

Fields:
  `states`          — sorted `Vector{Int}` of 0-indexed basis indices
  `full_to_sector`  — `Vector{Int32}` of length `2^N`; entry `s+1` gives the
                      1-based sector index of state `s`, or `0` if `s` is not
                      in the sector
  `dim`             — sector dimension = `binomial(N, n_down)`
"""
struct MagnetizationSector
    N::Int
    n_down::Int
    states::Vector{Int}
    full_to_sector::Vector{Int32}
    dim::Int
end

function MagnetizationSector(N::Int, n_down::Int)
    @assert 0 <= n_down <= N
    full_dim = 1 << N
    states = Int[]
    sizehint!(states, binomial(N, n_down))
    for s in 0:(full_dim - 1)
        count_ones(s) == n_down && push!(states, s)
    end
    dim = length(states)
    full_to_sector = zeros(Int32, full_dim)
    for (k, s) in enumerate(states)
        full_to_sector[s + 1] = Int32(k)
    end
    return MagnetizationSector(N, n_down, states, full_to_sector, dim)
end

"""Return the Hamming weight (number of ↓ spins) for a fixed-weight initial state,
or `-1` if the state spans multiple sectors (e.g. `"x+"`)."""
function _initial_state_hamming_weight(N::Int, state::AbstractString)::Int
    if state == "zeros"
        return 0
    elseif state in ("Neel", "neel")
        return count(iseven, 1:N)
    elseif state == "wall"
        return count(i -> i > N ÷ 2, 1:N)
    else
        return -1
    end
end

"""Return the 0-indexed full-basis index of a computational-basis initial state,
or `-1` for superposition states like `"x+"`."""
function _initial_state_basis_index(N::Int, state::AbstractString)::Int
    if state == "zeros"
        return 0
    elseif state in ("Neel", "neel")
        idx = 0
        for i in 1:N
            iseven(i) && (idx += 1 << (i - 1))
        end
        return idx
    elseif state == "wall"
        idx = 0
        for i in 1:N
            i > N ÷ 2 && (idx += 1 << (i - 1))
        end
        return idx
    else
        return -1
    end
end

# ═══════════════════════════════════════════════════════════════════════
#  Full-space operators
# ═══════════════════════════════════════════════════════════════════════

# ── TFIM ──────────────────────────────────────────────────────────────
# H = -J Σ_{i=1}^{N-1} σz_i σz_{i+1}  -  g Σ_{i=1}^N σx_i

struct TFIMOperator
    N::Int
    dim::Int
    J::Float64
    g::Float64
end

Base.eltype(::TFIMOperator) = ComplexF64
Base.size(op::TFIMOperator) = (op.dim, op.dim)
Base.size(op::TFIMOperator, ::Integer) = op.dim
LinearAlgebra.ishermitian(::TFIMOperator) = true

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            op::TFIMOperator,
                            x::AbstractVector{ComplexF64})
    N   = op.N
    dim = op.dim
    J   = op.J
    g   = op.g
    fill!(y, zero(ComplexF64))
    @inbounds for s in 0:(dim - 1)
        amp = x[s + 1]
        iszero(amp) && continue
        # ── diagonal: -J Σ σz_i σz_{i+1} ──
        diag = 0.0
        for bond in 1:(N - 1)
            bi = (s >> (bond - 1)) & 1
            bj = (s >> bond) & 1
            diag += (bi == bj) ? -J : J
        end
        y[s + 1] += diag * amp
        # ── off-diagonal: -g Σ σx_i  (flip bit i) ──
        for site in 1:N
            s2 = s ⊻ (1 << (site - 1))
            y[s2 + 1] += (-g) * amp
        end
    end
    return y
end

function (op::TFIMOperator)(v::AbstractVector{ComplexF64})
    y = similar(v)
    mul!(y, op, v)
    return y
end

# ── General / XXZ ─────────────────────────────────────────────────────
# H = Σ_{i} [Jxx σx_i σx_{i+1} + Jyy σy_i σy_{i+1} + Jzz σz_i σz_{i+1}]
#   + Σ_{i} [hx σx_i + hy σy_i + hz σz_i]
#
# Matrix-free action per nearest-neighbour bond (i, i+1):
#   σx⊗σx:  always flips both bits, coefficient +1
#   σy⊗σy:  flips both bits, coefficient -1 if bits same, +1 if bits differ
#   Combined:  same bits → Jxx - Jyy  (changes Sz by ±2)
#              diff bits → Jxx + Jyy  (Sz-conserving swap)
#   σz⊗σz:  diagonal, +1 same, -1 different

struct GeneralOperator
    N::Int
    dim::Int
    Jxx::Float64
    Jyy::Float64
    Jzz::Float64
    hx::Float64
    hy::Float64
    hz::Float64
end

Base.eltype(::GeneralOperator) = ComplexF64
Base.size(op::GeneralOperator) = (op.dim, op.dim)
Base.size(op::GeneralOperator, ::Integer) = op.dim
LinearAlgebra.ishermitian(::GeneralOperator) = true

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            op::GeneralOperator,
                            x::AbstractVector{ComplexF64})
    N   = op.N
    dim = op.dim
    Jxx = op.Jxx;  Jyy = op.Jyy;  Jzz = op.Jzz
    hx  = op.hx;   hy  = op.hy;    hz  = op.hz
    fill!(y, zero(ComplexF64))
    @inbounds for s in 0:(dim - 1)
        amp = x[s + 1]
        iszero(amp) && continue
        diag_val = 0.0
        # ── two-body (nearest neighbour) ──
        for bond in 1:(N - 1)
            mask_i = 1 << (bond - 1)
            mask_j = 1 << bond
            bi = (s >> (bond - 1)) & 1
            bj = (s >> bond) & 1
            # ZZ diagonal
            diag_val += Jzz * ((bi == bj) ? 1.0 : -1.0)
            # XX + YY off-diagonal
            coeff = (bi == bj) ? (Jxx - Jyy) : (Jxx + Jyy)
            if !iszero(coeff)
                s2 = s ⊻ mask_i ⊻ mask_j
                y[s2 + 1] += coeff * amp
            end
        end
        # ── single-site terms ──
        for site in 1:N
            b = (s >> (site - 1)) & 1
            diag_val += hz * (b == 0 ? 1.0 : -1.0)
            if !iszero(hx)
                s2 = s ⊻ (1 << (site - 1))
                y[s2 + 1] += hx * amp
            end
            if !iszero(hy)
                s2 = s ⊻ (1 << (site - 1))
                phase = (b == 0) ? ComplexF64(0.0, hy) : ComplexF64(0.0, -hy)
                y[s2 + 1] += phase * amp
            end
        end
        y[s + 1] += diag_val * amp
    end
    return y
end

function (op::GeneralOperator)(v::AbstractVector{ComplexF64})
    y = similar(v)
    mul!(y, op, v)
    return y
end

# ── Haldane–Shastry ──────────────────────────────────────────────────
# H = Σ_{i<j} J_ij (S_i · S_j),   S^α = σ^α / 2
#
# For each pair (i,j):
#   diagonal  (Sz Sz):  +J_ij/4 if same bits, -J_ij/4 if different
#   exchange  (S+S- + S-S+) = (Sx Sx + Sy Sy):
#       if bits differ, swap them with coefficient J_ij/2

struct HaldaneShastryOperator
    N::Int
    dim::Int
    pairs::Vector{Tuple{Int,Int,Float64}}   # (i, j, J_ij), 1-indexed
end

function HaldaneShastryOperator(N::Int; J::Float64=1.0, pbc::Bool=true)
    dim = 1 << N
    pairs = Tuple{Int,Int,Float64}[]
    for i in 1:N, j in (i + 1):N
        Jij = pbc ? J * (π / N)^2 / sin(π * (j - i) / N)^2 :
                    J / (j - i)^2
        iszero(Jij) && continue
        push!(pairs, (i, j, Jij))
    end
    return HaldaneShastryOperator(N, dim, pairs)
end

Base.eltype(::HaldaneShastryOperator) = ComplexF64
Base.size(op::HaldaneShastryOperator) = (op.dim, op.dim)
Base.size(op::HaldaneShastryOperator, ::Integer) = op.dim
LinearAlgebra.ishermitian(::HaldaneShastryOperator) = true

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            op::HaldaneShastryOperator,
                            x::AbstractVector{ComplexF64})
    dim = op.dim
    fill!(y, zero(ComplexF64))
    @inbounds for s in 0:(dim - 1)
        amp = x[s + 1]
        iszero(amp) && continue
        diag_val = 0.0
        for (i, j, Jij) in op.pairs
            bi = (s >> (i - 1)) & 1
            bj = (s >> (j - 1)) & 1
            if bi == bj
                diag_val += Jij * 0.25
            else
                diag_val -= Jij * 0.25
                s2 = s ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                y[s2 + 1] += (Jij * 0.5) * amp
            end
        end
        y[s + 1] += diag_val * amp
    end
    return y
end

function (op::HaldaneShastryOperator)(v::AbstractVector{ComplexF64})
    y = similar(v)
    mul!(y, op, v)
    return y
end

# ═══════════════════════════════════════════════════════════════════════
#  Sector operators (Sz-conserving models only)
# ═══════════════════════════════════════════════════════════════════════

# ── Sector General / XXZ ──────────────────────────────────────────────
# Only valid when Jxx == Jyy and hx == hy == 0.
# In this regime the only off-diagonal term is the Sz-conserving swap
# with coefficient (Jxx + Jyy) for different-spin pairs.

struct SectorGeneralOperator
    N::Int
    sector::MagnetizationSector
    exchange::Float64       # Jxx + Jyy
    Jzz::Float64
    hz::Float64
end

Base.eltype(::SectorGeneralOperator) = ComplexF64
Base.size(op::SectorGeneralOperator) = (op.sector.dim, op.sector.dim)
Base.size(op::SectorGeneralOperator, ::Integer) = op.sector.dim
LinearAlgebra.ishermitian(::SectorGeneralOperator) = true

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            op::SectorGeneralOperator,
                            x::AbstractVector{ComplexF64})
    N        = op.N
    sec      = op.sector
    states   = sec.states
    f2s      = sec.full_to_sector
    exchange = op.exchange
    Jzz      = op.Jzz
    hz       = op.hz
    fill!(y, zero(ComplexF64))
    @inbounds for k in 1:sec.dim
        s   = states[k]
        amp = x[k]
        iszero(amp) && continue
        diag_val = 0.0
        for bond in 1:(N - 1)
            bi = (s >> (bond - 1)) & 1
            bj = (s >> bond) & 1
            diag_val += Jzz * ((bi == bj) ? 1.0 : -1.0)
            if bi != bj
                s2 = s ⊻ (1 << (bond - 1)) ⊻ (1 << bond)
                k2 = f2s[s2 + 1]
                y[k2] += exchange * amp
            end
        end
        for site in 1:N
            b = (s >> (site - 1)) & 1
            diag_val += hz * (b == 0 ? 1.0 : -1.0)
        end
        y[k] += diag_val * amp
    end
    return y
end

function (op::SectorGeneralOperator)(v::AbstractVector{ComplexF64})
    y = similar(v)
    mul!(y, op, v)
    return y
end

# ── Sector Haldane–Shastry ────────────────────────────────────────────

struct SectorHaldaneShastryOperator
    N::Int
    sector::MagnetizationSector
    pairs::Vector{Tuple{Int,Int,Float64}}
end

Base.eltype(::SectorHaldaneShastryOperator) = ComplexF64
Base.size(op::SectorHaldaneShastryOperator) = (op.sector.dim, op.sector.dim)
Base.size(op::SectorHaldaneShastryOperator, ::Integer) = op.sector.dim
LinearAlgebra.ishermitian(::SectorHaldaneShastryOperator) = true

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64},
                            op::SectorHaldaneShastryOperator,
                            x::AbstractVector{ComplexF64})
    sec    = op.sector
    states = sec.states
    f2s    = sec.full_to_sector
    fill!(y, zero(ComplexF64))
    @inbounds for k in 1:sec.dim
        s   = states[k]
        amp = x[k]
        iszero(amp) && continue
        diag_val = 0.0
        for (i, j, Jij) in op.pairs
            bi = (s >> (i - 1)) & 1
            bj = (s >> (j - 1)) & 1
            if bi == bj
                diag_val += Jij * 0.25
            else
                diag_val -= Jij * 0.25
                s2 = s ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                k2 = f2s[s2 + 1]
                y[k2] += (Jij * 0.5) * amp
            end
        end
        y[k] += diag_val * amp
    end
    return y
end

function (op::SectorHaldaneShastryOperator)(v::AbstractVector{ComplexF64})
    y = similar(v)
    mul!(y, op, v)
    return y
end

# ═══════════════════════════════════════════════════════════════════════
#  Initial states
# ═══════════════════════════════════════════════════════════════════════

"""
    _build_initial_state_full(N, state) -> Vector{ComplexF64}

Build the initial state as a dense 2^N vector.  Supported values:
`"x+"`, `"Neel"`, `"neel"`, `"wall"`, `"zeros"`.
"""
function _build_initial_state_full(N::Int, state::AbstractString)::Vector{ComplexF64}
    dim = 1 << N
    if state == "x+"
        return fill(ComplexF64(1.0 / sqrt(dim)), dim)
    elseif state in ("Neel", "neel")
        v = zeros(ComplexF64, dim)
        idx = 0
        for i in 1:N
            iseven(i) && (idx += 1 << (i - 1))
        end
        v[idx + 1] = 1.0
        return v
    elseif state == "zeros"
        v = zeros(ComplexF64, dim)
        v[1] = 1.0
        return v
    elseif state == "wall"
        v = zeros(ComplexF64, dim)
        idx = 0
        for i in 1:N
            i > N ÷ 2 && (idx += 1 << (i - 1))
        end
        v[idx + 1] = 1.0
        return v
    else
        error("Unsupported initial state: $state (supported: x+, Neel, neel, wall, zeros)")
    end
end

"""
    _build_initial_state_sector(sector, full_basis_index) -> Vector{ComplexF64}

Build a one-hot sector-basis vector for a computational-basis initial state.
"""
function _build_initial_state_sector(sector::MagnetizationSector,
                                     full_basis_index::Int)::Vector{ComplexF64}
    v = zeros(ComplexF64, sector.dim)
    k = sector.full_to_sector[full_basis_index + 1]
    @assert k > 0 "State index $full_basis_index not found in sector (n_down=$(sector.n_down))"
    v[k] = 1.0
    return v
end

# ═══════════════════════════════════════════════════════════════════════
#  Observable computation
# ═══════════════════════════════════════════════════════════════════════

"""
    _z_expect_full_into!(out, ψ, N)

Compute `out[i] = ⟨ψ|σz_i|ψ⟩` via bit manipulation (no Z matrices allocated).
Full-basis version.
"""
function _z_expect_full_into!(out::AbstractVector{Float64},
                              ψ::AbstractVector{ComplexF64}, N::Int)
    dim = 1 << N
    fill!(out, 0.0)
    @inbounds for x in 0:(dim - 1)
        p = abs2(ψ[x + 1])
        iszero(p) && continue
        for i in 1:N
            out[i] += ifelse(iszero((x >> (i - 1)) & 1), p, -p)
        end
    end
    return out
end

"""
    _z_expect_sector_into!(out, ψ, sector)

Sector-basis version of ⟨σz_i⟩ — iterates over sector states only.
"""
function _z_expect_sector_into!(out::AbstractVector{Float64},
                                ψ::AbstractVector{ComplexF64},
                                sector::MagnetizationSector)
    N = sector.N
    fill!(out, 0.0)
    @inbounds for k in 1:sector.dim
        s = sector.states[k]
        p = abs2(ψ[k])
        iszero(p) && continue
        for i in 1:N
            out[i] += ifelse(iszero((s >> (i - 1)) & 1), p, -p)
        end
    end
    return out
end

# ═══════════════════════════════════════════════════════════════════════
#  Krylov propagation
# ═══════════════════════════════════════════════════════════════════════

"""
    _propagate_and_observe!(z_expect, norm_vals, H_op, ψ0, t_grid, N, cfg; sector)

Incrementally propagate `ψ(t)` through `t_grid` via Krylov exponential,
recording ⟨σz_i⟩ and ‖ψ‖² at every snapshot.

Returns the final state vector (in the working basis — full or sector).
"""
function _propagate_and_observe!(
        z_expect::Matrix{Float64},
        norm_vals::Vector{Float64},
        H_op,
        ψ0::Vector{ComplexF64},
        t_grid::Vector{Float64},
        N::Int,
        cfg::KrylovConfig;
        sector::Union{Nothing,MagnetizationSector}=nothing)

    n_times   = length(t_grid)
    ψ         = copy(ψ0)
    t_current = 0.0

    for k in 1:n_times
        Δt = t_grid[k] - t_current
        if Δt > 1e-15
            ψ, info = exponentiate(H_op, -im * Δt, ψ;
                                   krylovdim=cfg.krylovdim,
                                   maxiter=cfg.maxiter,
                                   tol=cfg.tol,
                                   ishermitian=true)
            t_current = t_grid[k]
            if info.converged != 1
                @printf("[reference] WARNING: Krylov did not converge at t=%.6g (normres=%.3e, numiter=%d)\n",
                        t_grid[k], info.normres, info.numiter)
                flush(stdout)
            end
        end
        norm_vals[k] = sum(abs2, ψ)
        if sector === nothing
            _z_expect_full_into!(@view(z_expect[:, k]), ψ, N)
        else
            _z_expect_sector_into!(@view(z_expect[:, k]), ψ, sector)
        end
    end
    return ψ
end

# ═══════════════════════════════════════════════════════════════════════
#  Cache
# ═══════════════════════════════════════════════════════════════════════

function _cache_path(cache_dir::String, model_name::String, N::Int,
                     T::Float64, n_points::Int, init_state::String="",
                     H_params::NamedTuple=NamedTuple())
    suffix = isempty(init_state) ? "" : "_$(init_state)"
    T_slug = replace(string(T), "." => "p")
    h = isempty(pairs(H_params)) ? "" : "_h$(string(hash(H_params), base=16))"
    label  = "ref_$(model_name)_N$(N)_T$(T_slug)_pts$(n_points)_lsb$(suffix)$(h)_krylov.jls"
    return joinpath(cache_dir, label)
end

# ═══════════════════════════════════════════════════════════════════════
#  HS pair couplings (shared between full-space and sector constructors)
# ═══════════════════════════════════════════════════════════════════════

function _hs_pair_couplings(N::Int; J::Float64=1.0, pbc::Bool=true)
    pairs = Tuple{Int,Int,Float64}[]
    for i in 1:N, j in (i + 1):N
        Jij = pbc ? J * (π / N)^2 / sin(π * (j - i) / N)^2 :
                    J / (j - i)^2
        iszero(Jij) && continue
        push!(pairs, (i, j, Jij))
    end
    return pairs
end

# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

"""
    compute_or_load_reference(model_name, N, T, t_grid, H_params;
                              cache_dir, initial_state,
                              krylov_config, use_sector) -> ReferenceData

Compute exact reference via matrix-free Krylov propagation, or load from cache.

Arguments:
  `model_name`  — `"tfim"`, `"ising"`, `"general"`, `"haldane_shastry"`, `"hs"`
  `N`           — number of sites
  `T`           — final time (must equal `t_grid[end]` or exceed it)
  `t_grid`      — sorted vector of observation times
  `H_params`    — `NamedTuple` with model-specific couplings

Keyword arguments:
  `cache_dir`      — directory for serialised cache files
  `initial_state`  — override default (`"x+"` for TFIM, `"Neel"` for general/HS)
  `krylov_config`  — `KrylovConfig(...)` for propagation settings
  `use_sector`     — attempt Sz-sector reduction when applicable (default `true`)
"""
function compute_or_load_reference(
        model_name::AbstractString, N::Int, T::Float64,
        t_grid::AbstractVector{Float64}, H_params::NamedTuple;
        cache_dir::String="benchmarks/results/reference_cache",
        initial_state::String="",
        krylov_config::KrylovConfig=KrylovConfig(),
        use_sector::Bool=true)

    mkpath(cache_dir)
    cpath = _cache_path(cache_dir, model_name, N, T, length(t_grid), initial_state, H_params)

    if isfile(cpath)
        @printf("[reference] loading cached: %s\n", cpath)
        flush(stdout)
        return deserialize(cpath)::ReferenceData
    end

    @printf("[reference] computing exact reference (Krylov): model=%s  N=%d  T=%.4g  (%d time points)\n",
            model_name, N, T, length(t_grid))
    flush(stdout)

    t0_wall = time()
    m = lowercase(strip(model_name))

    # ── initial state string ──
    init_str = if !isempty(initial_state)
        initial_state
    elseif m in ("tfim", "ising")
        "x+"
    elseif m == "general"
        "Neel"
    elseif m in ("haldane_shastry", "hs")
        "Neel"
    else
        error("No default initial state for model $m; pass initial_state= explicitly")
    end

    tg = collect(Float64, t_grid)
    @assert issorted(tg) "t_grid must be sorted nondecreasing"

    # ── sector eligibility ──
    hw = _initial_state_hamming_weight(N, init_str)
    can_sector = (hw >= 0) && use_sector

    if m in ("tfim", "ising")
        can_sector = false
    elseif m == "general"
        Jxx_v = Float64(get(H_params, :Jxx, 0.0))
        Jyy_v = Float64(get(H_params, :Jyy, 0.0))
        hx_v  = Float64(get(H_params, :hx, 0.0))
        hy_v  = Float64(get(H_params, :hy, 0.0))
        if Jxx_v != Jyy_v || hx_v != 0.0 || hy_v != 0.0
            can_sector = false
        end
    end
    # HS always conserves Sz — no extra check needed

    # ── build operator + initial state ──
    local H_op
    local ψ0::Vector{ComplexF64}
    local sector::Union{Nothing,MagnetizationSector}

    if can_sector
        sector = MagnetizationSector(N, hw)
        @printf("[reference]   sector mode: n_down=%d  sector_dim=%d  (full_dim=%d)\n",
                hw, sector.dim, 1 << N)
        flush(stdout)

        basis_idx = _initial_state_basis_index(N, init_str)
        ψ0 = _build_initial_state_sector(sector, basis_idx)

        if m == "general"
            Jxx_v = Float64(get(H_params, :Jxx, 0.0))
            Jyy_v = Float64(get(H_params, :Jyy, 0.0))
            Jzz_v = Float64(get(H_params, :Jzz, 0.0))
            hz_v  = Float64(get(H_params, :hz, 0.0))
            H_op = SectorGeneralOperator(N, sector, Jxx_v + Jyy_v, Jzz_v, hz_v)
        elseif m in ("haldane_shastry", "hs")
            J_v   = Float64(get(H_params, :J, 1.0))
            pbc_v = Bool(get(H_params, :pbc, true))
            pairs = _hs_pair_couplings(N; J=J_v, pbc=pbc_v)
            H_op = SectorHaldaneShastryOperator(N, sector, pairs)
        else
            error("Sector mode not supported for model $m")
        end
    else
        sector = nothing
        ψ0 = _build_initial_state_full(N, init_str)

        if m in ("tfim", "ising")
            J_v = Float64(get(H_params, :J, 1.0))
            g_v = Float64(get(H_params, :g, 1.05))
            H_op = TFIMOperator(N, 1 << N, J_v, g_v)
        elseif m == "general"
            Jxx_v = Float64(get(H_params, :Jxx, 0.0))
            Jyy_v = Float64(get(H_params, :Jyy, 0.0))
            Jzz_v = Float64(get(H_params, :Jzz, 0.0))
            hx_v  = Float64(get(H_params, :hx, 0.0))
            hy_v  = Float64(get(H_params, :hy, 0.0))
            hz_v  = Float64(get(H_params, :hz, 0.0))
            H_op = GeneralOperator(N, 1 << N, Jxx_v, Jyy_v, Jzz_v, hx_v, hy_v, hz_v)
        elseif m in ("haldane_shastry", "hs")
            J_v   = Float64(get(H_params, :J, 1.0))
            pbc_v = Bool(get(H_params, :pbc, true))
            H_op = HaldaneShastryOperator(N; J=J_v, pbc=pbc_v)
        else
            error("Unsupported model_name=$model_name")
        end
    end

    # ── energy (constant for Schrödinger evolution) ──
    Hψ0 = H_op(ψ0)
    E_mean = real(dot(ψ0, Hψ0))

    # ── allocate outputs ──
    n_times   = length(tg)
    z_expect  = Matrix{Float64}(undef, N, n_times)
    energy    = fill(E_mean, n_times)
    norm_vals = Vector{Float64}(undef, n_times)

    # ── propagate ──
    working_dim = sector === nothing ? (1 << N) : sector.dim
    @printf("[reference]   propagating %d time points (working_dim=%d) ... ",
            n_times, working_dim)
    flush(stdout)

    ψ_final = _propagate_and_observe!(z_expect, norm_vals, H_op, ψ0, tg, N,
                                      krylov_config; sector=sector)

    # ── decompress to full basis if sector mode ──
    if sector !== nothing
        psi_T = zeros(ComplexF64, 1 << N)
        for k in 1:sector.dim
            psi_T[sector.states[k] + 1] = ψ_final[k]
        end
    else
        psi_T = copy(ψ_final)
    end

    wall = time() - t0_wall
    @printf("done (%.2f s)\n", wall)
    flush(stdout)

    max_norm_err = maximum(abs.(norm_vals .- 1.0))
    if max_norm_err > 1e-8
        @printf("[reference] WARNING: max ‖ψ‖² deviation from 1: %.3e\n", max_norm_err)
        flush(stdout)
    end

    ref = ReferenceData(tg, psi_T, z_expect, energy, norm_vals)

    serialize(cpath, ref)
    @printf("[reference] cached to: %s\n", cpath)
    flush(stdout)

    return ref
end

# ═══════════════════════════════════════════════════════════════════════
#  Validation against dense eigendecomposition
# ═══════════════════════════════════════════════════════════════════════

# Inline dense builders (for validation only — small N).

const _σx = ComplexF64[0 1; 1 0]
const _σy = ComplexF64[0 -im; im 0]
const _σz = ComplexF64[1 0; 0 -1]
const _I2 = ComplexF64[1 0; 0 1]

function _val_kron_lsb(ops::NTuple{K,Matrix{ComplexF64}}) where K
    acc = ones(ComplexF64, 1, 1)
    for s in K:-1:1
        acc = kron(acc, ops[s])
    end
    return acc
end

function _val_full_op(N::Int, op::Matrix{ComplexF64}, site::Int)
    ops = ntuple(s -> (s == site ? op : _I2), N)
    return _val_kron_lsb(ops)
end

function _val_full_two_body(N::Int, opA::Matrix{ComplexF64},
                            opB::Matrix{ComplexF64}, i::Int, j::Int)
    ops = ntuple(s -> (s == i ? opA : s == j ? opB : _I2), N)
    return _val_kron_lsb(ops)
end

function _val_dense_tfim(N::Int; J::Float64=1.0, g::Float64=1.05)
    dim = 1 << N
    H = zeros(ComplexF64, dim, dim)
    for i in 1:(N - 1)
        H .+= (-J) .* _val_full_two_body(N, _σz, _σz, i, i + 1)
    end
    for i in 1:N
        H .+= (-g) .* _val_full_op(N, _σx, i)
    end
    return H
end

function _val_dense_general(N::Int; Jxx=0.0, Jyy=0.0, Jzz=0.0, hx=0.0, hy=0.0, hz=0.0)
    dim = 1 << N
    H = zeros(ComplexF64, dim, dim)
    for i in 1:(N - 1)
        Jxx != 0 && (H .+= Jxx .* _val_full_two_body(N, _σx, _σx, i, i + 1))
        Jyy != 0 && (H .+= Jyy .* _val_full_two_body(N, _σy, _σy, i, i + 1))
        Jzz != 0 && (H .+= Jzz .* _val_full_two_body(N, _σz, _σz, i, i + 1))
    end
    for i in 1:N
        hx != 0 && (H .+= hx .* _val_full_op(N, _σx, i))
        hy != 0 && (H .+= hy .* _val_full_op(N, _σy, i))
        hz != 0 && (H .+= hz .* _val_full_op(N, _σz, i))
    end
    return H
end

function _val_dense_hs(N::Int; J::Float64=1.0, pbc::Bool=true)
    dim = 1 << N
    H = zeros(ComplexF64, dim, dim)
    Sx = ComplexF64[0 0.5; 0.5 0]
    Sy = ComplexF64[0 -0.5im; 0.5im 0]
    Sz = ComplexF64[0.5 0; 0 -0.5]
    for i in 1:N, j in (i + 1):N
        Jij = pbc ? J * (π / N)^2 / sin(π * (j - i) / N)^2 :
                    J / (j - i)^2
        iszero(Jij) && continue
        for S in (Sx, Sy, Sz)
            H .+= Jij .* _val_full_two_body(N, S, S, i, j)
        end
    end
    return H
end

"""
Dense eigendecomposition reference at all time points (for validation only).
Returns `(z_expect_dense, psi_T_dense, energy)`.
"""
function _val_dense_reference(H_dense::Matrix{ComplexF64}, ψ0::Vector{ComplexF64},
                              t_grid::Vector{Float64}, N::Int)
    F = eigen(Hermitian(H_dense))
    eigenvals = real.(F.values)
    eigvecs   = F.vectors
    c = eigvecs' * ψ0
    c_sq = abs2.(c)
    E_mean = dot(eigenvals, c_sq)

    n_times  = length(t_grid)
    z_expect = Matrix{Float64}(undef, N, n_times)
    psi_T    = zeros(ComplexF64, 1 << N)

    for (ti, t) in enumerate(t_grid)
        phases = exp.((-im) .* eigenvals .* t)
        ψt = eigvecs * (c .* phases)
        _z_expect_full_into!(@view(z_expect[:, ti]), ψt, N)
        if ti == n_times
            psi_T .= ψt
        end
    end
    return z_expect, psi_T, E_mean
end

"""
    validate_against_dense(; N=8, tol=1e-10, verbose=true) -> Bool

Compare the Krylov reference against dense eigendecomposition for TFIM,
General/XXZ, and Haldane–Shastry at small system size.

Returns `true` if all tests pass within `tol`.
"""
function validate_against_dense(; N::Int=8, tol::Float64=1e-10, verbose::Bool=true)
    all_pass = true
    T  = 2.0
    tg = collect(0.0:0.2:T)
    cfg = KrylovConfig(krylovdim=30, maxiter=100, tol=1e-13)
    tmpdir = mktempdir()

    function _check(label, model, H_params, H_dense, init_str; use_sector=true)
        ψ0_full = _build_initial_state_full(N, init_str)
        z_d, psi_d, E_d = _val_dense_reference(H_dense, ψ0_full, tg, N)

        ref = compute_or_load_reference(model, N, T, tg, H_params;
                cache_dir=tmpdir, initial_state=init_str,
                krylov_config=cfg, use_sector=use_sector)

        overlap   = abs(dot(psi_d, ref.psi_T))^2
        infid     = 1.0 - overlap
        max_z_err = maximum(abs.(z_d .- ref.z_expect))
        e_err     = abs(E_d - ref.energy[1])
        max_norm  = maximum(abs.(ref.norm_vals .- 1.0))

        ok = (infid < tol) && (max_z_err < tol) && (e_err < tol) && (max_norm < tol)
        if verbose
            status = ok ? "PASS" : "FAIL"
            @printf("  [%s] %-24s  infid=%.2e  max|ΔZ|=%.2e  |ΔE|=%.2e  max|Δ‖ψ‖²|=%.2e\n",
                    status, label, infid, max_z_err, e_err, max_norm)
            flush(stdout)
        end
        return ok
    end

    if verbose
        @printf("[validate] N=%d  T=%.1f  tol=%.1e\n", N, T, tol)
        flush(stdout)
    end

    # ── TFIM ──
    J, g = 1.0, 1.05
    hp = (J=J, g=g)
    Hd = _val_dense_tfim(N; J=J, g=g)
    all_pass &= _check("TFIM (x+)", "tfim", hp, Hd, "x+"; use_sector=false)
    all_pass &= _check("TFIM (zeros)", "tfim", hp, Hd, "zeros"; use_sector=false)

    # ── General / XXZ (with sector) ──
    hp_g = (Jxx=1.0, Jyy=1.0, Jzz=0.5, hx=0.0, hy=0.0, hz=0.0)
    Hd_g = _val_dense_general(N; Jxx=1.0, Jyy=1.0, Jzz=0.5)
    all_pass &= _check("XXZ Neel (sector)", "general", hp_g, Hd_g, "Neel"; use_sector=true)
    all_pass &= _check("XXZ Neel (full)", "general", hp_g, Hd_g, "Neel"; use_sector=false)
    all_pass &= _check("XXZ wall (sector)", "general", hp_g, Hd_g, "wall"; use_sector=true)

    # ── General with broken Sz (no sector) ──
    hp_broken = (Jxx=1.0, Jyy=0.8, Jzz=0.5, hx=0.2, hy=0.0, hz=0.1)
    Hd_broken = _val_dense_general(N; Jxx=1.0, Jyy=0.8, Jzz=0.5, hx=0.2, hz=0.1)
    all_pass &= _check("General broken-Sz (x+)", "general", hp_broken, Hd_broken, "x+";
                        use_sector=false)

    # ── Haldane–Shastry (PBC) ──
    hp_hs = (J=1.0, pbc=true)
    Hd_hs = _val_dense_hs(N; J=1.0, pbc=true)
    all_pass &= _check("HS PBC Neel (sector)", "hs", hp_hs, Hd_hs, "Neel"; use_sector=true)
    all_pass &= _check("HS PBC Neel (full)", "hs", hp_hs, Hd_hs, "Neel"; use_sector=false)
    all_pass &= _check("HS PBC wall (sector)", "hs", hp_hs, Hd_hs, "wall"; use_sector=true)

    # ── Haldane–Shastry (OBC) ──
    hp_hs_obc = (J=1.0, pbc=false)
    Hd_hs_obc = _val_dense_hs(N; J=1.0, pbc=false)
    all_pass &= _check("HS OBC Neel (sector)", "hs", hp_hs_obc, Hd_hs_obc, "Neel";
                        use_sector=true)

    if verbose
        @printf("[validate] %s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED")
        flush(stdout)
    end
    return all_pass
end

end # module BenchmarkReference
