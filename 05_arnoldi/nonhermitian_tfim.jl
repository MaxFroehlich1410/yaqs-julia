using LinearAlgebra

"""
    pauli_x()

Return the Pauli-X matrix as `Matrix{ComplexF64}`.
"""
function pauli_x()
    return ComplexF64[0 1; 1 0]
end

"""
    pauli_z()

Return the Pauli-Z matrix as `Matrix{ComplexF64}`.
"""
function pauli_z()
    return ComplexF64[1 0; 0 -1]
end

"""
    id2()

Return the 2x2 identity as `Matrix{ComplexF64}`.
"""
function id2()
    return Matrix{ComplexF64}(I, 2, 2)
end

"""
    k_local(J, g, gamma)

Local onsite contribution

    k_loc = -im * g * X + (gamma/2) * Z - (gamma/2) * I

used in the bond-dimension-3 MPO for the non-Hermitian TFIM generator `K`.
"""
function k_local(g::Real, gamma::Real)
    X = pauli_x()
    Z = pauli_z()
    I2 = id2()
    return (-1im * g) .* X .+ (gamma / 2) .* Z .- (gamma / 2) .* I2
end

@inline function _block_to_mpo_tensor(blocks::Matrix{Matrix{ComplexF64}})
    dl, dr = size(blocks)
    tensor = zeros(ComplexF64, dl, 2, 2, dr)
    for l in 1:dl, r in 1:dr
        tensor[l, :, :, r] .= blocks[l, r]
    end
    return tensor
end

"""
    build_nonhermitian_k_mpo(N; J=1.0, g=0.7, gamma=0.2)

Build the non-Hermitian nearest-neighbor TFIM generator MPO for

    dψ/dt = K ψ

with

    K = -i J Σ Z_j Z_{j+1} - i g Σ X_j - γ Σ (I - Z_j)/2.

MPO layout follows the repository convention `(LeftBond, PhysOut, PhysIn, RightBond)`.
"""
function build_nonhermitian_k_mpo(N::Int; J::Real=1.0, g::Real=0.7, gamma::Real=0.2)
    @assert N >= 1 "N must be >= 1."

    X = pauli_x()
    Z = pauli_z()
    I2 = id2()
    ZJ = (-1im * J) .* Z
    kloc = k_local(g, gamma)

    tensors = Vector{Array{ComplexF64, 4}}(undef, N)

    if N == 1
        tensors[1] = reshape(kloc, 1, 2, 2, 1)
        return Yaqs.MPOModule.MPO(N, tensors, fill(2, N), 0)
    end

    left = Matrix{Matrix{ComplexF64}}(undef, 1, 3)
    left[1, 1] = kloc
    left[1, 2] = Z
    left[1, 3] = I2
    tensors[1] = _block_to_mpo_tensor(left)

    bulk = Matrix{Matrix{ComplexF64}}(undef, 3, 3)
    zero2 = zeros(ComplexF64, 2, 2)
    bulk[1, 1] = I2
    bulk[1, 2] = zero2
    bulk[1, 3] = zero2
    bulk[2, 1] = ZJ
    bulk[2, 2] = zero2
    bulk[2, 3] = zero2
    bulk[3, 1] = kloc
    bulk[3, 2] = Z
    bulk[3, 3] = I2

    for site in 2:(N - 1)
        tensors[site] = _block_to_mpo_tensor(bulk)
    end

    right = Matrix{Matrix{ComplexF64}}(undef, 3, 1)
    right[1, 1] = I2
    right[2, 1] = ZJ
    right[3, 1] = kloc
    tensors[N] = _block_to_mpo_tensor(right)

    return Yaqs.MPOModule.MPO(N, tensors, fill(2, N), 0)
end

"""
    tdvp_generator_from_k(K_mpo)

The repository TDVP evolves with an internal `exp(-im * dt * A)` convention.
To realize the target ODE `dψ/dt = K ψ`, pass `A = iK` to TDVP.
"""
function tdvp_generator_from_k(K_mpo::Yaqs.MPOModule.MPO)
    return (1im) * K_mpo
end

"""
    initial_plus_mps(N)

Construct `|+>^{⊗N}` in MPS form using the repository constructor.
"""
function initial_plus_mps(N::Int)
    return Yaqs.MPSModule.MPS(N; state="x+")
end

"""
    initial_plus_dense(N)

Construct `|+>^{⊗N}` as a dense full-Hilbert-space vector in the same basis
ordering as `to_vec(mps)` (site 1 is least-significant bit).
"""
function initial_plus_dense(N::Int)
    dim = 1 << N
    amp = inv(sqrt(dim))
    return fill(ComplexF64(amp), dim)
end
