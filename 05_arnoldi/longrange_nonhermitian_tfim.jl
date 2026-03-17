using LinearAlgebra

"""
    pauli_x_lr(), pauli_z_lr(), id2_lr()

Small local operators used by the long-range non-Hermitian benchmark.
"""
pauli_x_lr() = ComplexF64[0 1; 1 0]
pauli_z_lr() = ComplexF64[1 0; 0 -1]
id2_lr() = Matrix{ComplexF64}(I, 2, 2)

"""
    coupling_lr(i, j; J0=1.0, lambda=0.5)

Long-range coupling

    J_ij = J0 * lambda^(|i-j|-1),  i != j.
"""
function coupling_lr(i::Int, j::Int; J0::Real=1.0, lambda::Real=0.5)
    @assert i != j
    dist = abs(i - j)
    return J0 * (lambda^(dist - 1))
end

function _local_operator_mpo(N::Int, site::Int, op::AbstractMatrix{<:Number})
    @assert 1 <= site <= N
    tensors = Vector{Array{ComplexF64, 4}}(undef, N)
    I2 = id2_lr()
    for s in 1:N
        local_op = s == site ? Matrix{ComplexF64}(op) : I2
        tensors[s] = reshape(local_op, 1, 2, 2, 1)
    end
    return Yaqs.MPOModule.MPO(N, tensors, fill(2, N), 0)
end

"""
    build_longrange_nonhermitian_k_mpo(N; J0=1.0, lambda=0.5, g=0.7, gamma=0.2)

Build an exact MPO for

    K_lr = -i Σ_{i<j} J_ij Z_i Z_j - i g Σ_j X_j - γ Σ_j (I - Z_j)/2,

using robust composition from existing MPO building blocks:
- each pair term is built via `mpo_from_two_qubit_gate_matrix`,
- each local term via bond-1 local-operator MPO,
- all terms summed with repository MPO addition.
"""
function build_longrange_nonhermitian_k_mpo(N::Int;
                                            J0::Real=1.0,
                                            lambda::Real=0.5,
                                            g::Real=0.7,
                                            gamma::Real=0.2)
    @assert N >= 1
    @assert J0 > 0
    @assert 0 < lambda < 1

    X = pauli_x_lr()
    Z = pauli_z_lr()
    I2 = id2_lr()
    ZZ = kron(Z, Z)

    K = Yaqs.MPOModule.MPO(N; identity=false, physical_dimensions=2)

    # Long-range pair terms
    for i in 1:(N - 1)
        for j in (i + 1):N
            Jij = coupling_lr(i, j; J0=J0, lambda=lambda)
            pair_mpo = Yaqs.MPOModule.mpo_from_two_qubit_gate_matrix(ZZ, i, j, N; d=2)
            K = K + ((-1im * Jij) * pair_mpo)
        end
    end

    # Local transverse-field and dissipative terms
    for s in 1:N
        K = K + ((-1im * g) * _local_operator_mpo(N, s, X))
        K = K + ((-gamma / 2) * _local_operator_mpo(N, s, I2))
        K = K + ((gamma / 2) * _local_operator_mpo(N, s, Z))
    end

    return K
end

"""
    tdvp_generator_from_k_lr(K_lr_mpo)

TDVP path uses local propagators `exp(-im*dt*A)`. For target ODE `dψ/dt = K_lr ψ`,
pass `A = i*K_lr`.
"""
tdvp_generator_from_k_lr(K_lr_mpo::Yaqs.MPOModule.MPO) = (1im) * K_lr_mpo

"""
    initial_plus_mps_lr(N), initial_plus_dense_lr(N)

Consistent `|+>^{⊗N}` initial states in MPS and dense-vector form.
"""
initial_plus_mps_lr(N::Int) = Yaqs.MPSModule.MPS(N; state="x+")

function initial_plus_dense_lr(N::Int)
    dim = 1 << N
    return fill(ComplexF64(inv(sqrt(dim))), dim)
end

"""
    verify_longrange_mpo_dims(K_mpo, N)

Sanity-check MPO tensor ranks and physical dimensions.
"""
function verify_longrange_mpo_dims(K_mpo::Yaqs.MPOModule.MPO, N::Int)
    @assert K_mpo.length == N
    for s in 1:N
        T = K_mpo.tensors[s]
        @assert ndims(T) == 4
        @assert size(T, 2) == 2
        @assert size(T, 3) == 2
    end
    return true
end
