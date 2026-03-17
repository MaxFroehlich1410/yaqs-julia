using LinearAlgebra

"""
    embed_one_site_lr(op, site, N)

Embed single-site operator into full Hilbert space with basis convention
matching `to_vec(mps)` (site 1 = least significant bit).
"""
function embed_one_site_lr(op::AbstractMatrix{<:Number}, site::Int, N::Int)
    @assert 1 <= site <= N
    ops = [id2_lr() for _ in 1:N]
    ops[site] = Matrix{ComplexF64}(op)
    full = ops[N]
    for s in (N - 1):-1:1
        full = kron(full, ops[s])
    end
    return full
end

"""
    embed_two_site_lr(op1, site1, op2, site2, N)

Embed a two-site product operator into full Hilbert space.
"""
function embed_two_site_lr(op1::AbstractMatrix{<:Number}, site1::Int,
                           op2::AbstractMatrix{<:Number}, site2::Int, N::Int)
    @assert 1 <= site1 <= N
    @assert 1 <= site2 <= N
    @assert site1 != site2
    ops = [id2_lr() for _ in 1:N]
    ops[site1] = Matrix{ComplexF64}(op1)
    ops[site2] = Matrix{ComplexF64}(op2)
    full = ops[N]
    for s in (N - 1):-1:1
        full = kron(full, ops[s])
    end
    return full
end

"""
    build_longrange_k_dense(N; J0=1.0, lambda=0.5, g=0.7, gamma=0.2)

Construct dense long-range non-Hermitian generator:

    K_lr = -i Σ_{i<j} J_ij Z_i Z_j - i g Σ_j X_j - γ Σ_j (I - Z_j)/2.
"""
function build_longrange_k_dense(N::Int;
                                 J0::Real=1.0,
                                 lambda::Real=0.5,
                                 g::Real=0.7,
                                 gamma::Real=0.2)
    @assert N >= 1
    @assert J0 > 0
    @assert 0 < lambda < 1

    dim = 1 << N
    K = zeros(ComplexF64, dim, dim)
    X = pauli_x_lr()
    Z = pauli_z_lr()
    I2 = id2_lr()

    for i in 1:(N - 1)
        for j in (i + 1):N
            Jij = coupling_lr(i, j; J0=J0, lambda=lambda)
            K .+= (-1im * Jij) .* embed_two_site_lr(Z, i, Z, j, N)
        end
    end
    for s in 1:N
        K .+= (-1im * g) .* embed_one_site_lr(X, s, N)
        K .+= (-gamma / 2) .* embed_one_site_lr(I2, s, N)
        K .+= (gamma / 2) .* embed_one_site_lr(Z, s, N)
    end
    return K
end

"""
    precompute_z_diagonals_lr(N)

`zdiag[j, b]` is Z-eigenvalue at site `j` for basis index `b`.
"""
function precompute_z_diagonals_lr(N::Int)
    dim = 1 << N
    zdiag = Array{Float64}(undef, N, dim)
    for basis_idx in 0:(dim - 1)
        for j in 1:N
            bit = (basis_idx >>> (j - 1)) & 0x1
            zdiag[j, basis_idx + 1] = bit == 0 ? 1.0 : -1.0
        end
    end
    return zdiag
end

"""
    z_expectations_from_state(psi, zdiag)

Return `(z_unnorm, z_normed)` where:
- `z_unnorm[j] = psi† Z_j psi`
- `z_normed[j] = z_unnorm[j] / (psi†psi)` (if norm is nonzero)
"""
function z_expectations_from_state(psi::AbstractVector{<:Complex}, zdiag::AbstractMatrix{<:Real})
    nsites = size(zdiag, 1)
    z_unnorm = Vector{ComplexF64}(undef, nsites)
    for j in 1:nsites
        z_unnorm[j] = dot(psi, zdiag[j, :] .* psi)
    end
    norm_sq = real(dot(psi, psi))
    if norm_sq > 0
        z_normed = z_unnorm ./ norm_sq
    else
        z_normed = fill(ComplexF64(NaN), nsites)
    end
    return z_unnorm, z_normed
end

"""
    run_longrange_exact_reference(K_dense, psi0; dt=0.05, tmax=1.0)

Exact dense reference evolution with one matrix exponential:
`U = exp(dt*K_dense)` and repeated `psi <- U*psi`.
"""
function run_longrange_exact_reference(K_dense::AbstractMatrix{<:Complex},
                                       psi0::AbstractVector{<:Complex};
                                       dt::Real=0.05,
                                       tmax::Real=1.0)
    @assert dt > 0
    @assert tmax >= 0
    nsteps = Int(round(tmax / dt))
    @assert isapprox(nsteps * dt, tmax; atol=1e-12) "tmax must be an integer multiple of dt."

    N = Int(round(log2(length(psi0))))
    zdiag = precompute_z_diagonals_lr(N)
    times = collect(0:nsteps) .* dt
    states = Vector{Vector{ComplexF64}}(undef, nsteps + 1)
    states[1] = copy(psi0)

    t0 = time()
    U = exp(ComplexF64(dt) .* Matrix{ComplexF64}(K_dense))
    for n in 1:nsteps
        states[n + 1] = U * states[n]
    end
    runtime_seconds = time() - t0
    return times, states, zdiag, runtime_seconds
end
