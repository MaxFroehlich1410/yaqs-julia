"""
    BenchmarkReference

Compute and cache exact reference solutions using pure-Julia dense matrix
exponentiation via eigendecomposition.

Feasibility by system size (ComplexF64 matrices):
  N=12 → dim=4096,   H≈256 MB  — trivial
  N=14 → dim=16384,  H≈4.3 GB  — feasible (diagonalization ~5–30 min)
  N=16 → dim=65536,  H≈68 GB   — requires >64 GB RAM, not recommended
  N≥20 → physically impossible (16 TB+)

Memory-efficient design: Z expectations are computed via O(N·2^N) bit
manipulation (no full Z_i matrices stored), and energy is derived from the
eigenvalue decomposition (no H_mat needed after diagonalization).

Cached data (per model × N × T × t_grid):
- `psi_T`      : full statevector at final time T  (Vector{ComplexF64}, length 2^N)
- `z_expect`   : ⟨Z_i⟩(t) for all sites            (Matrix{Float64}, N × n_times)
- `energy`     : ⟨H⟩(t)                             (Vector{Float64})
- `norm_vals`  : ‖ψ‖²(t)                            (Vector{Float64})
"""
module BenchmarkReference

using Printf
using LinearAlgebra
using Serialization

export compute_or_load_reference, ReferenceData

struct ReferenceData
    t_grid::Vector{Float64}
    psi_T::Vector{ComplexF64}          # statevector at final time
    z_expect::Matrix{Float64}          # (N, n_times)
    energy::Vector{Float64}            # (n_times,)
    norm_vals::Vector{Float64}         # (n_times,)
end

# ── Dense operator builders ──────────────────────────────────────────

const _σx = ComplexF64[0 1; 1 0]
const _σy = ComplexF64[0 -im; im 0]
const _σz = ComplexF64[1 0; 0 -1]
const _I2 = ComplexF64[1 0; 0 1]

"""Kronecker product in LSB convention (site 1 = bit 0)."""
function _kron_lsb(ops::NTuple{K,Matrix{ComplexF64}}) where K
    acc = ones(ComplexF64, 1, 1)
    for s in K:-1:1
        acc = kron(acc, ops[s])
    end
    return acc
end

"""Build operator `op` acting on site `i` in an N-qubit Hilbert space."""
function _full_op(N::Int, op::Matrix{ComplexF64}, site::Int)
    @assert 1 <= site <= N
    ops = ntuple(s -> (s == site ? op : _I2), N)
    return _kron_lsb(ops)
end

"""Build two-body operator op_A ⊗ op_B on sites (i, i+1)."""
function _full_two_body(N::Int, opA::Matrix{ComplexF64}, opB::Matrix{ComplexF64}, i::Int)
    @assert 1 <= i < N
    ops = ntuple(s -> (s == i ? opA : s == i + 1 ? opB : _I2), N)
    return _kron_lsb(ops)
end

"""Build two-body operator opA on site i, opB on site j (arbitrary i < j)."""
function _full_two_body_arb(N::Int, opA::Matrix{ComplexF64}, opB::Matrix{ComplexF64},
                             i::Int, j::Int)
    @assert 1 <= i < j <= N
    ops = ntuple(s -> (s == i ? opA : s == j ? opB : _I2), N)
    return _kron_lsb(ops)
end

"""Build dense TFIM Hamiltonian: H = -J Σ Z_i Z_{i+1} - g Σ X_i"""
function _dense_tfim(N::Int; J::Float64=1.0, g::Float64=1.05)
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)
    for i in 1:(N-1)
        H .+= (-J) .* _full_two_body(N, _σz, _σz, i)
    end
    for i in 1:N
        H .+= (-g) .* _full_op(N, _σx, i)
    end
    return Hermitian(H)
end

"""Build dense general Hamiltonian: H = Σ (Jxx XX + Jyy YY + Jzz ZZ) + Σ (hx X + hy Y + hz Z)"""
function _dense_general(N::Int; Jxx::Float64=0.0, Jyy::Float64=0.0, Jzz::Float64=0.0,
                        hx::Float64=0.0, hy::Float64=0.0, hz::Float64=0.0)
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)
    for i in 1:(N-1)
        if Jxx != 0; H .+= Jxx .* _full_two_body(N, _σx, _σx, i); end
        if Jyy != 0; H .+= Jyy .* _full_two_body(N, _σy, _σy, i); end
        if Jzz != 0; H .+= Jzz .* _full_two_body(N, _σz, _σz, i); end
    end
    for i in 1:N
        if hx != 0; H .+= hx .* _full_op(N, _σx, i); end
        if hy != 0; H .+= hy .* _full_op(N, _σy, i); end
        if hz != 0; H .+= hz .* _full_op(N, _σz, i); end
    end
    return Hermitian(H)
end

"""
Build dense Haldane–Shastry Hamiltonian:
  H = J Σ_{i<j} J_{ij} (S_i·S_j),  S^α = σ^α/2
  J_{ij} = (π/N)² / sin²(π(j−i)/N)   (PBC)
  J_{ij} = 1/(j−i)²                   (OBC)
Uses the same basis convention as `MPSModule.to_vec` (site 1 = bit 0 / LSB).
"""
function _dense_hs(N::Int; J::Float64=1.0, pbc::Bool=true)
    dim = 2^N
    H   = zeros(ComplexF64, dim, dim)
    Sx  = ComplexF64[0 0.5; 0.5 0]
    Sy  = ComplexF64[0 -0.5im; 0.5im 0]
    Sz  = ComplexF64[0.5 0; 0 -0.5]
    for i in 1:N, j in (i + 1):N
        Jij = pbc ? J * (π / N)^2 / sin(π * (j - i) / N)^2 :
                    J / (j - i)^2
        iszero(Jij) && continue
        for S in (Sx, Sy, Sz)
            ops = ntuple(k -> (k == i || k == j) ? S : _I2, N)
            H .+= Jij .* _kron_lsb(ops)
        end
    end
    return Hermitian(H)
end

"""Build initial state as dense vector."""
function _dense_initial_state(N::Int, state::AbstractString)
    if state == "x+"
        v1 = ComplexF64[1/sqrt(2), 1/sqrt(2)]
    elseif state == "Neel"
        # |0101...> :  site 1=|0>, site 2=|1>, ...
        v = zeros(ComplexF64, 2^N)
        idx = 0
        for i in 1:N
            if iseven(i)  # site i = |1>
                idx += 2^(i-1)
            end
        end
        v[idx + 1] = 1.0   # 1-indexed
        return v
    elseif state == "zeros"
        v = zeros(ComplexF64, 2^N)
        v[1] = 1.0
        return v
    elseif state == "neel"
        # |↑↓↑↓…⟩  (odd sites ↑, even sites ↓)
        # LSB: site i = bit (i-1).  Even sites get bit (i-1) set.
        v   = zeros(ComplexF64, 2^N)
        idx = sum(iseven(i) ? 2^(i - 1) : 0 for i in 1:N)
        v[idx + 1] = 1.0
        return v
    elseif state == "wall"
        # |↑…↑↓…↓⟩  domain wall at centre
        # LSB: last ⌊N/2⌋ sites (i > N÷2) are ↓ → set bit (i-1).
        v   = zeros(ComplexF64, 2^N)
        idx = sum(i > N ÷ 2 ? 2^(i - 1) : 0 for i in 1:N)
        v[idx + 1] = 1.0
        return v
    else
        error("Unsupported initial state for reference: $state")
    end
    # For product states like x+: build by kron
    if state == "x+"
        v = ComplexF64[1/sqrt(2), 1/sqrt(2)]
        for _ in 2:N
            v = kron(v, ComplexF64[1/sqrt(2), 1/sqrt(2)])
        end
        return v
    end
end

# ── Efficient Z-expectation (no full-matrix allocation) ─────────────

"""
    _z_expect_into!(out, ψt, N)

Fill `out[i] = ⟨ψt|Z_i|ψt⟩` for each site `i ∈ 1:N` by iterating over
the 2^N basis states and accumulating ±|ψ_x|² according to the value of
bit (i-1) of the basis-state index `x`.

Memory: O(1) extra beyond `ψt` — no Z_i matrices are allocated.
"""
function _z_expect_into!(out::AbstractVector{Float64}, ψt::Vector{ComplexF64}, N::Int)
    dim = 2^N
    fill!(out, 0.0)
    @inbounds for x in 0:(dim - 1)
        p = abs2(ψt[x + 1])
        iszero(p) && continue
        for i in 1:N
            # bit (i-1) of x: 0 → Z eigenvalue +1, 1 → Z eigenvalue -1
            out[i] += ifelse(iszero((x >> (i - 1)) & 1), p, -p)
        end
    end
    return out
end

# ── Cache ────────────────────────────────────────────────────────────

function _cache_path(cache_dir::String, model_name::String, N::Int, T::Float64,
                     n_points::Int, init_state::String="")
    suffix = isempty(init_state) ? "" : "_$(init_state)"
    label  = "ref_$(model_name)_N$(N)_T$(replace(string(T), "." => "p"))_pts$(n_points)_lsb$(suffix).jls"
    return joinpath(cache_dir, label)
end

"""
    compute_or_load_reference(model_name, N, T, t_grid, H_params;
                              cache_dir, initial_state) -> ReferenceData

Compute exact reference via dense matrix exponentiation, or load from cache.

`initial_state` overrides the per-model default ("x+" for TFIM, "Neel" for general/HS).
Supported values: `"x+"`, `"Neel"`, `"neel"`, `"wall"`, `"zeros"`.
"""
function compute_or_load_reference(model_name::AbstractString, N::Int, T::Float64,
                                   t_grid::AbstractVector{Float64}, H_params::NamedTuple;
                                   cache_dir::String="benchmarks/results/reference_cache",
                                   initial_state::String="")
    mkpath(cache_dir)
    cpath = _cache_path(cache_dir, model_name, N, T, length(t_grid), initial_state)

    if isfile(cpath)
        @printf("[reference] loading cached: %s\n", cpath)
        flush(stdout)
        return deserialize(cpath)::ReferenceData
    end

    @printf("[reference] computing exact reference: model=%s N=%d T=%.4g  (%d time points) ...\n",
            model_name, N, T, length(t_grid))
    flush(stdout)

    t0_wall = time()

    # Build dense Hamiltonian
    m = lowercase(strip(model_name))
    H_dense = if m in ("tfim", "ising")
        _dense_tfim(N; J=get(H_params, :J, 1.0), g=get(H_params, :g, 1.05))
    elseif m == "general"
        _dense_general(N;
            Jxx=get(H_params, :Jxx, 0.0), Jyy=get(H_params, :Jyy, 0.0),
            Jzz=get(H_params, :Jzz, 0.0),
            hx=get(H_params, :hx, 0.0), hy=get(H_params, :hy, 0.0),
            hz=get(H_params, :hz, 0.0))
    elseif m in ("haldane_shastry", "hs")
        _dense_hs(N; J=get(H_params, :J, 1.0), pbc=get(H_params, :pbc, true))
    else
        error("Unsupported model_name=$model_name for reference")
    end

    # Diagonalize once.
    # For N=14 this is a 16384×16384 Hermitian matrix (~4.3 GB); it's the
    # unavoidable cost. We free everything else immediately after.
    @printf("[reference]   diagonalizing %dx%d Hamiltonian ... ", 2^N, 2^N)
    flush(stdout)
    F = eigen(H_dense)
    eigenvals = real.(F.values)   # real for Hermitian
    eigvecs   = F.vectors          # columns are eigenvectors
    H_dense   = nothing            # release dense Hamiltonian memory
    GC.gc()
    @printf("done\n")
    flush(stdout)

    # Initial state — use caller-supplied value, otherwise model default
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
    ψ0 = _dense_initial_state(N, init_str)

    # Coefficients in eigenbasis: c_k = ⟨k|ψ0⟩
    c = eigvecs' * ψ0

    # Energy is time-independent: ⟨H⟩ = Σ_k E_k |c_k|²
    # (|exp(-iE_k t)|² = 1, so eigenvalue mix doesn't change in time)
    c_sq = abs2.(c)
    E_mean = dot(eigenvals, c_sq)

    n_times = length(t_grid)
    z_expect = Matrix{Float64}(undef, N, n_times)
    energy   = Vector{Float64}(undef, n_times)
    norm_vals = Vector{Float64}(undef, n_times)
    psi_T    = zeros(ComplexF64, 2^N)

    @printf("[reference]   evolving %d time points (N=%d, dim=%d) ... ", n_times, N, 2^N)
    flush(stdout)

    for (ti, t) in enumerate(t_grid)
        # ψ(t) = Σ_k c_k exp(-i E_k t) |k>
        phases = exp.((-im) .* eigenvals .* t)
        ψt = eigvecs * (c .* phases)

        norm_vals[ti] = sum(abs2, ψt)
        energy[ti]    = E_mean

        # Compute ⟨Z_i⟩ via O(N·dim) bit-manipulation — no Z_i matrices needed.
        # Site i (1-indexed) occupies bit position (i-1) in the 0-indexed basis index.
        # Z_i eigenvalue: +1 if that bit is 0, -1 if it is 1.
        _z_expect_into!(@view(z_expect[:, ti]), ψt, N)

        if ti == n_times
            psi_T .= ψt
        end
    end

    wall = time() - t0_wall
    @printf("done (total %.2f s)\n", wall)
    flush(stdout)

    ref = ReferenceData(collect(t_grid), psi_T, z_expect, energy, norm_vals)

    serialize(cpath, ref)
    @printf("[reference] cached to: %s\n", cpath)
    flush(stdout)

    return ref
end

end # module BenchmarkReference