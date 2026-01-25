module BUGModule

using LinearAlgebra
using TensorOperations

using ..MPSModule: MPS, shift_orthogonality_center!, truncate!, check_if_valid_mps
using ..MPOModule: MPO
using ..SimulationConfigs: AbstractSimConfig, TimeEvolutionConfig, MeasurementConfig, StrongMeasurementConfig
using ..Decompositions: right_qr, left_qr
using ..Algorithms

export bug!, fixed_bug!, bug_second_order!, fixed_bug_second_order!, hybrid_bug_second_order!, FirstOrderBUGStrategy

"""
    FirstOrderBUGStrategy

Selects which first-order Basis-Update and Galerkin (BUG) sweep to use.

- `STANDARD`: performs basis enlargement by stacking the pre-update tensor with the local update.
- `FIXED`: does not enlarge bonds during the sweep (fixed-bond variant).
"""
@enum FirstOrderBUGStrategy::UInt8 begin
    STANDARD = 1
    FIXED = 2
end

@inline function _bug_dt(config::AbstractSimConfig)
    if config isa TimeEvolutionConfig
        return (config::TimeEvolutionConfig).dt
    else
        # Python sets dt=1 for circuit/measurement configs.
        return 1.0
    end
end

@inline function _bug_trunc_threshold(config::AbstractSimConfig)
    return getfield(config, :truncation_threshold)
end

@inline function _bug_max_bond_dim(config::AbstractSimConfig)
    return getfield(config, :max_bond_dim)
end

@inline function _identity_matrix(::Type{T}, n::Int) where {T}
    M = Matrix{T}(undef, n, n)
    fill!(M, zero(T))
    @inbounds for i in 1:n
        M[i, i] = one(T)
    end
    return M
end

@inline function _stack_left(A::Array{T,3}, B::Array{T,3}) where {T}
    lA, dA, rA = size(A)
    lB, dB, rB = size(B)
    @assert dA == dB
    @assert rA == rB
    C = Array{T,3}(undef, lA + lB, dA, rA)
    @inbounds begin
        C[1:lA, :, :] .= A
        C[(lA+1):(lA+lB), :, :] .= B
    end
    return C
end

"""
    prepare_canonical_site_tensors(state::MPS{ComplexF64}, mpo::MPO{ComplexF64})

Assuming `state` is (globally) left-canonical with orthogonality center at the last site,
construct the list of center tensors obtained by shifting the center from left to right,
and the corresponding left environments.

Returns `(canon_center_tensors, left_envs)` where:
- `canon_center_tensors[i]` is the tensor at site `i` when the orthogonality center is at `i`
  (layout `(Dl, d, Dr)`).
- `left_envs[i]` is the left environment *to the left of site `i`* (so `left_envs[1]` is the left boundary),
  and `left_envs` has length `L+1`.
"""
function prepare_canonical_site_tensors(state::MPS{ComplexF64}, mpo::MPO{ComplexF64})
    L = state.length
    @assert mpo.length == L

    # Ensure the state is left-canonicalized (center at last site).
    shift_orthogonality_center!(state, L)

    canon_center_tensors = copy(state.tensors) # shallow vector copy, tensors are replaced as needed

    left_envs = Vector{Array{ComplexF64,3}}(undef, L + 1)
    left_bond = size(state.tensors[1], 1)
    mpo_left_dim = size(mpo.tensors[1], 1)
    left_envs[1] = Algorithms.make_identity_env(left_bond, mpo_left_dim)

    # Shift center 1 -> 2 -> ... -> L by QR on each left tensor; store the center tensors.
    for i in 1:(L - 1)
        A = canon_center_tensors[i]              # (Dl, d, Dr_old)
        Q, R = right_qr(A)                       # Q: (Dl, d, Dr_new), R: (Dr_new, Dr_old)

        # Build the center tensor at i+1 by absorbing R into the next (old) tensor.
        Next = canon_center_tensors[i + 1]       # (Dr_old, d_next, Dr_next)
        Dr_old = size(Next, 1)
        d_next = size(Next, 2)
        Dr_next = size(Next, 3)
        @assert size(R, 2) == Dr_old

        Next_mat = reshape(Next, Dr_old, d_next * Dr_next)
        New_next_mat = Array{ComplexF64,2}(undef, size(R, 1), d_next * Dr_next)
        mul!(New_next_mat, R, Next_mat)
        canon_center_tensors[i + 1] = reshape(New_next_mat, size(R, 1), d_next, Dr_next)

        left_envs[i + 1] = Algorithms.update_left_environment(Q, mpo.tensors[i], left_envs[i])
    end

    # Right boundary env placeholder (not used by BUG directly, but keep shape consistent with TDVP).
    left_envs[L + 1] = Array{ComplexF64,3}(undef, 0, 0, 0)
    return canon_center_tensors, left_envs
end

@inline function choose_stack_tensor(site::Int, canon_center_tensors::Vector{Array{ComplexF64,3}}, state::MPS{ComplexF64})
    is_leaf = (site == state.length)
    return is_leaf ? state.tensors[site] : canon_center_tensors[site]
end

@inline function find_new_q(old_stack_tensor::Array{ComplexF64,3}, updated_tensor::Array{ComplexF64,3})
    stacked = _stack_left(old_stack_tensor, updated_tensor)
    _, new_q = left_qr(stacked) # LQ => right-canonical tensor (rows orthonormal)
    return new_q
end

@inline function find_new_q_fixed(updated_tensor::Array{ComplexF64,3})
    _, new_q = left_qr(updated_tensor)
    return new_q
end

"""
    build_basis_change_tensor(old_q, new_q, old_m) -> M

Build basis change matrix `M` mapping `old_left -> new_left` for the current site.

Layouts:
- `old_q`: (old_left, phys, old_right)
- `new_q`: (new_left, phys, new_right)
- `old_m`: (old_right, new_right)  (basis change of the site to the right)

Returns:
- `M`: (old_left, new_left)
"""
function build_basis_change_tensor(old_q::Array{ComplexF64,3}, new_q::Array{ComplexF64,3}, old_m::Array{ComplexF64,2})
    @tensor tmp[ol, p, nr] := old_q[ol, p, or] * old_m[or, nr]
    @tensor M[ol, nl] := tmp[ol, p, nr] * conj(new_q[nl, p, nr])
    return M
end

@inline function _update_site(Lenv::Array{ComplexF64,3},
                              Renv::Array{ComplexF64,3},
                              W::Array{ComplexF64,4},
                              A::Array{ComplexF64,3},
                              dt::Float64,
                              numiter_lanczos::Int)
    op = (x) -> Algorithms.project_site(x, Lenv, Renv, W)
    return Algorithms.expm_krylov(op, A, dt, numiter_lanczos)
end

function local_update!(state::MPS{ComplexF64},
                       mpo::MPO{ComplexF64},
                       left_envs::Vector{Array{ComplexF64,3}},
                       right_env::Array{ComplexF64,3},
                       canon_center_tensors::Vector{Array{ComplexF64,3}},
                       site::Int,
                       right_m_block::Array{ComplexF64,2},
                       dt::Float64,
                       numiter_lanczos::Int)
    old_tensor = canon_center_tensors[site]
    updated_tensor = _update_site(left_envs[site], right_env, mpo.tensors[site], old_tensor, dt, numiter_lanczos)

    old_stack_tensor = choose_stack_tensor(site, canon_center_tensors, state)
    new_q = find_new_q(old_stack_tensor, updated_tensor)

    old_q = state.tensors[site]
    basis_change_m = build_basis_change_tensor(old_q, new_q, right_m_block)

    state.tensors[site] = new_q

    # Propagate basis change into the (stored) center tensor of the left neighbor.
    prev = canon_center_tensors[site - 1]
    @tensor prev_new[lp, p, ln] := prev[lp, p, lo] * basis_change_m[lo, ln]
    canon_center_tensors[site - 1] = prev_new

    new_right_env = Algorithms.update_right_environment(new_q, mpo.tensors[site], right_env)
    return basis_change_m, new_right_env
end

function fixed_local_update!(state::MPS{ComplexF64},
                             mpo::MPO{ComplexF64},
                             left_envs::Vector{Array{ComplexF64,3}},
                             right_env::Array{ComplexF64,3},
                             canon_center_tensors::Vector{Array{ComplexF64,3}},
                             site::Int,
                             right_m_block::Array{ComplexF64,2},
                             dt::Float64,
                             numiter_lanczos::Int)
    old_tensor = canon_center_tensors[site]
    updated_tensor = _update_site(left_envs[site], right_env, mpo.tensors[site], old_tensor, dt, numiter_lanczos)

    new_q = find_new_q_fixed(updated_tensor)
    old_q = state.tensors[site]
    basis_change_m = build_basis_change_tensor(old_q, new_q, right_m_block)

    state.tensors[site] = new_q

    prev = canon_center_tensors[site - 1]
    @tensor prev_new[lp, p, ln] := prev[lp, p, lo] * basis_change_m[lo, ln]
    canon_center_tensors[site - 1] = prev_new

    new_right_env = Algorithms.update_right_environment(new_q, mpo.tensors[site], right_env)
    return new_right_env, basis_change_m
end

@inline function _trunc_keep_dim(S::AbstractVector{<:Real}, threshold::Real, max_bond_dim::Int)
    discarded_sq = 0.0
    keep_rank = length(S)
    min_keep = 2
    total_sq = sum(abs2, S)
    for k in length(S):-1:1
        discarded_sq += S[k]^2
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
    return clamp(keep_rank, 1, max_bond_dim)
end

"""
    _truncate_bond_keep_left(A_center, B_right, threshold, max_bond_dim)

Truncate the bond between `A_center` (Dl,d,D) and `B_right` (D,d,Dr), keeping the
singular values on the *left* so that the orthogonality center stays at the left site.

Returns `(A_new_center, B_new_right, P)` where `P` maps the old bond basis (size D)
to the new truncated basis (size χ), such that `B_new ≈ P * B_old` in matrix form.
"""
function _truncate_bond_keep_left(A_center::Array{ComplexF64,3},
                                  B_right::Array{ComplexF64,3},
                                  threshold::Float64,
                                  max_bond_dim::Int)
    Dl, d1, D = size(A_center)
    D2, d2, Dr = size(B_right)
    @assert D == D2

    @tensor Θ[dl, p1, p2, dr] := A_center[dl, p1, k] * B_right[k, p2, dr]
    Θmat = reshape(Θ, Dl * d1, d2 * Dr)

    F = svd(Θmat)
    keep = _trunc_keep_dim(F.S, threshold, max_bond_dim)

    U = F.U[:, 1:keep]
    S = F.S[1:keep]
    Vt = F.Vt[1:keep, :]

    # Keep S on the left: U * Diagonal(S)
    US = copy(U)
    @inbounds for j in 1:keep
        @views US[:, j] .*= S[j]
    end

    A_new = reshape(US, Dl, d1, keep)
    B_new = reshape(Vt, keep, d2, Dr)

    # Basis change on the bond: B_new_mat = P * B_old_mat
    Bold = reshape(B_right, D, d2 * Dr)
    P = Vt * adjoint(Bold)  # (keep x D)

    return A_new, B_new, P
end

"""
    bug!(state, mpo, config; numiter_lanczos=25)

First-order BUG sweep (right-to-left). Updates `state` in-place and truncates at the end.
"""
function bug!(state::MPS{ComplexF64},
              mpo::MPO{ComplexF64},
              config::AbstractSimConfig;
              numiter_lanczos::Int=25,
              do_truncate::Bool=true,
              truncation_granularity::Symbol=:after_sweep)
    L = state.length
    @assert mpo.length == L "State and Hamiltonian must have the same number of sites"
    check_if_valid_mps(state)

    dt = _bug_dt(config)

    canon_center_tensors, left_envs = prepare_canonical_site_tensors(state, mpo)

    # Right boundary env and basis-change block.
    r_bond = size(state.tensors[L], 3)
    mpo_r = size(mpo.tensors[L], 4)
    right_env = Algorithms.make_identity_env(r_bond, mpo_r)
    right_m_block = _identity_matrix(ComplexF64, r_bond)

    threshold = _bug_trunc_threshold(config)
    max_bond_dim = _bug_max_bond_dim(config)
    @assert truncation_granularity === :after_sweep || truncation_granularity === :after_site

    for site in L:-1:2
        right_env_in = right_env
        right_m_block, right_env = local_update!(
            state, mpo, left_envs, right_env, canon_center_tensors, site, right_m_block, dt, numiter_lanczos
        )

        if do_truncate && truncation_granularity === :after_site
            A_new, B_new, P = _truncate_bond_keep_left(canon_center_tensors[site - 1], state.tensors[site],
                                                       threshold, max_bond_dim)
            canon_center_tensors[site - 1] = A_new
            state.tensors[site] = B_new
            right_m_block = right_m_block * adjoint(P)  # (old -> D) * (D -> χ)
            right_env = Algorithms.update_right_environment(B_new, mpo.tensors[site], right_env_in)
        end
    end

    # Update first site.
    updated = _update_site(left_envs[1], right_env, mpo.tensors[1], canon_center_tensors[1], dt, numiter_lanczos)
    state.tensors[1] = updated

    # Truncation:
    # - :after_sweep => single sweep-level truncation at the end
    # - :after_site  => already truncated each bond during the sweep
    if do_truncate && truncation_granularity === :after_sweep
        truncate!(state; threshold=threshold, max_bond_dim=max_bond_dim)
    end
    return nothing
end

"""
    fixed_bug!(state, mpo, config; numiter_lanczos=25)

First-order BUG sweep without basis enlargement. Updates `state` in-place (no truncation at end).
"""
function fixed_bug!(state::MPS{ComplexF64},
                    mpo::MPO{ComplexF64},
                    config::AbstractSimConfig;
                    numiter_lanczos::Int=25,
                    truncation_granularity::Symbol=:after_sweep)
    L = state.length
    @assert mpo.length == L "State and Hamiltonian must have the same number of sites"
    check_if_valid_mps(state)

    dt = _bug_dt(config)

    canon_center_tensors, left_envs = prepare_canonical_site_tensors(state, mpo)

    r_bond = size(state.tensors[L], 3)
    mpo_r = size(mpo.tensors[L], 4)
    right_env = Algorithms.make_identity_env(r_bond, mpo_r)
    right_m_block = _identity_matrix(ComplexF64, r_bond)

    threshold = _bug_trunc_threshold(config)
    max_bond_dim = _bug_max_bond_dim(config)
    @assert truncation_granularity === :after_sweep || truncation_granularity === :after_site

    for site in L:-1:2
        right_env_in = right_env
        right_env, right_m_block = fixed_local_update!(
            state, mpo, left_envs, right_env, canon_center_tensors, site, right_m_block, dt, numiter_lanczos
        )
        if truncation_granularity === :after_site
            A_new, B_new, P = _truncate_bond_keep_left(canon_center_tensors[site - 1], state.tensors[site],
                                                       threshold, max_bond_dim)
            canon_center_tensors[site - 1] = A_new
            state.tensors[site] = B_new
            right_m_block = right_m_block * adjoint(P)
            right_env = Algorithms.update_right_environment(B_new, mpo.tensors[site], right_env_in)
        end
    end

    updated = _update_site(left_envs[1], right_env, mpo.tensors[1], canon_center_tensors[1], dt, numiter_lanczos)
    state.tensors[1] = updated

    return nothing
end

@inline function _do_truncate_after_second_order(truncation_timing::Symbol)
    return truncation_timing === :after_window
end

function _flip_mps!(state::MPS{ComplexF64})
    L = state.length
    reverse!(state.tensors)
    for i in 1:L
        # (Dl, d, Dr) -> (Dr, d, Dl)
        state.tensors[i] = permutedims(state.tensors[i], (3, 2, 1))
    end
    state.orth_center = (state.orth_center == 0) ? 0 : (L + 1 - state.orth_center)
    return nothing
end

function _flip_mpo!(mpo::MPO{ComplexF64})
    L = mpo.length
    reverse!(mpo.tensors)
    for i in 1:L
        # (Dl, po, pi, Dr) -> (Dr, po, pi, Dl)
        mpo.tensors[i] = permutedims(mpo.tensors[i], (4, 2, 3, 1))
    end
    mpo.orth_center = (mpo.orth_center == 0) ? 0 : (L + 1 - mpo.orth_center)
    return nothing
end

function _abstract_bug_second_order!(state::MPS{ComplexF64},
                                    mpo::MPO{ComplexF64},
                                    strategies::Tuple{FirstOrderBUGStrategy, FirstOrderBUGStrategy},
                                    config::AbstractSimConfig;
                                    numiter_lanczos::Int=25,
                                    truncation_timing::Symbol=:during,
                                    truncation_granularity::Symbol=:after_sweep)
    L = state.length
    @assert mpo.length == L "State and Hamiltonian must have the same number of sites"
    @assert truncation_timing === :during || truncation_timing === :after_window

    dt_orig = _bug_dt(config)
    dt_half = (config isa TimeEvolutionConfig) ? (0.5 * dt_orig) : 1.0
    do_trunc_half = !_do_truncate_after_second_order(truncation_timing)

    # First half-step (right-to-left).
    if config isa TimeEvolutionConfig
        (config::TimeEvolutionConfig).dt = dt_half
    end
    if strategies[1] == STANDARD
        bug!(state, mpo, config; numiter_lanczos=numiter_lanczos,
             do_truncate=do_trunc_half, truncation_granularity=truncation_granularity)
    else
        fixed_bug!(state, mpo, config; numiter_lanczos=numiter_lanczos,
                   truncation_granularity=truncation_granularity)
    end

    # Second half-step in opposite direction by flipping the network (matches Python implementation).
    _flip_mps!(state)
    _flip_mpo!(mpo)
    shift_orthogonality_center!(state, state.length)
    if strategies[2] == STANDARD
        bug!(state, mpo, config; numiter_lanczos=numiter_lanczos,
             do_truncate=do_trunc_half, truncation_granularity=truncation_granularity)
    else
        fixed_bug!(state, mpo, config; numiter_lanczos=numiter_lanczos,
                   truncation_granularity=truncation_granularity)
    end
    _flip_mps!(state)
    _flip_mpo!(mpo)

    # Restore dt.
    if config isa TimeEvolutionConfig
        (config::TimeEvolutionConfig).dt = dt_orig
    end

    # Optional: truncate only once after the full second-order step.
    if _do_truncate_after_second_order(truncation_timing) &&
       (strategies[1] == STANDARD || strategies[2] == STANDARD)
        threshold = _bug_trunc_threshold(config)
        max_bond_dim = _bug_max_bond_dim(config)
        truncate!(state; threshold=threshold, max_bond_dim=max_bond_dim)
    end

    return nothing
end

"""
    bug_second_order!(state, mpo, config; numiter_lanczos=25)

Second-order BUG via Strang splitting: two half-steps with opposite sweep directions.
"""
function bug_second_order!(state::MPS{ComplexF64},
                           mpo::MPO{ComplexF64},
                           config::AbstractSimConfig;
                           numiter_lanczos::Int=25,
                           truncation_timing::Symbol=:during,
                           truncation_granularity::Symbol=:after_sweep)
    return _abstract_bug_second_order!(state, mpo, (STANDARD, STANDARD), config;
                                      numiter_lanczos=numiter_lanczos,
                                      truncation_timing=truncation_timing,
                                      truncation_granularity=truncation_granularity)
end

function fixed_bug_second_order!(state::MPS{ComplexF64},
                                 mpo::MPO{ComplexF64},
                                 config::AbstractSimConfig;
                                 numiter_lanczos::Int=25,
                                 truncation_timing::Symbol=:during,
                                 truncation_granularity::Symbol=:after_sweep)
    return _abstract_bug_second_order!(state, mpo, (FIXED, FIXED), config;
                                      numiter_lanczos=numiter_lanczos,
                                      truncation_timing=truncation_timing,
                                      truncation_granularity=truncation_granularity)
end

function hybrid_bug_second_order!(state::MPS{ComplexF64},
                                  mpo::MPO{ComplexF64},
                                  config::AbstractSimConfig;
                                  numiter_lanczos::Int=25,
                                  truncation_timing::Symbol=:during,
                                  truncation_granularity::Symbol=:after_sweep)
    return _abstract_bug_second_order!(state, mpo, (STANDARD, FIXED), config;
                                      numiter_lanczos=numiter_lanczos,
                                      truncation_timing=truncation_timing,
                                      truncation_granularity=truncation_granularity)
end

end # module BUGModule

