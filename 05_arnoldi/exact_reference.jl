using LinearAlgebra

"""
    embed_one_site(op, site, N)

Embed a single-site operator into the full Hilbert space.
Basis convention matches `to_vec(mps)`: site 1 is the least-significant bit.
"""
function embed_one_site(op::AbstractMatrix{<:Number}, site::Int, N::Int)
    @assert 1 <= site <= N
    ops = [id2() for _ in 1:N]
    ops[site] = Matrix{ComplexF64}(op)
    full = ops[N]
    for s in (N - 1):-1:1
        full = kron(full, ops[s])
    end
    return full
end

"""
    embed_two_site(op1, site1, op2, site2, N)

Embed a product operator acting on two (possibly adjacent) sites.
"""
function embed_two_site(op1::AbstractMatrix{<:Number}, site1::Int,
                        op2::AbstractMatrix{<:Number}, site2::Int, N::Int)
    @assert 1 <= site1 <= N
    @assert 1 <= site2 <= N
    @assert site1 != site2
    ops = [id2() for _ in 1:N]
    ops[site1] = Matrix{ComplexF64}(op1)
    ops[site2] = Matrix{ComplexF64}(op2)
    full = ops[N]
    for s in (N - 1):-1:1
        full = kron(full, ops[s])
    end
    return full
end

"""
    build_k_dense(N; J=1.0, g=0.7, gamma=0.2)

Construct the full dense non-Hermitian generator

    K = -iJ Σ Z_j Z_{j+1} - ig Σ X_j - γ Σ (I - Z_j)/2

in `ComplexF64^(2^N × 2^N)`.
"""
function build_k_dense(N::Int; J::Real=1.0, g::Real=0.7, gamma::Real=0.2)
    @assert N >= 1
    dim = 1 << N
    K = zeros(ComplexF64, dim, dim)
    X = pauli_x()
    Z = pauli_z()
    I2 = id2()

    for j in 1:(N - 1)
        K .+= (-1im * J) .* embed_two_site(Z, j, Z, j + 1, N)
    end
    for j in 1:N
        K .+= (-1im * g) .* embed_one_site(X, j, N)
        K .+= (-gamma / 2) .* embed_one_site(I2, j, N)
        K .+= (gamma / 2) .* embed_one_site(Z, j, N)
    end
    return K
end

"""
    precompute_z_diagonals(N)

Return matrix `zdiag` with shape `(N, 2^N)` where `zdiag[j, b]` is the eigenvalue
of `Z_j` for basis index `b` (1-based), with site 1 as least-significant bit.
"""
function precompute_z_diagonals(N::Int)
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
    exact_z_expectations(psi, zdiag)

Compute unnormalized `⟨Z_j⟩ = psi† Z_j psi` for all sites.
"""
function exact_z_expectations(psi::AbstractVector{<:Complex}, zdiag::AbstractMatrix{<:Real})
    N = size(zdiag, 1)
    vals = Vector{ComplexF64}(undef, N)
    for j in 1:N
        vals[j] = dot(psi, zdiag[j, :] .* psi)
    end
    return vals
end

"""
    run_exact_reference(K_dense, psi0; dt=0.05, tmax=1.0)

Evolve the dense system exactly on a uniform time grid using
`U = exp(dt * K_dense)` once, then repeated `psi <- U * psi`.
Returns `(times, states, zdiag, runtime_seconds)`.
"""
function run_exact_reference(K_dense::AbstractMatrix{<:Complex},
                             psi0::AbstractVector{<:Complex};
                             dt::Real=0.05, tmax::Real=1.0)
    @assert dt > 0
    @assert tmax >= 0
    nsteps = Int(round(tmax / dt))
    @assert isapprox(nsteps * dt, tmax; atol=1e-12) "tmax must be an integer multiple of dt."

    N = Int(round(log2(length(psi0))))
    zdiag = precompute_z_diagonals(N)
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
