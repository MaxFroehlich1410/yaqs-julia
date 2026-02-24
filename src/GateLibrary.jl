module GateLibrary

using StaticArrays
using LinearAlgebra

export AbstractOperator, AbstractGate, AbstractNoise
export matrix, generator, is_unitary, hamiltonian_coeff
export XGate, YGate, ZGate, HGate, SGate, TGate, SdgGate, TdgGate
export RxGate, RyGate, RzGate, PhaseGate, UGate
export CXGate, CYGate, CZGate, CHGate, CPhaseGate, SWAPGate, iSWAPGate
export RxxGate, RyyGate, RzzGate
export Barrier, IdGate, RaisingGate, LoweringGate

# --- Abstract Types ---

"""
Abstract supertype for all operators in the gate library.

This serves as the common ancestor for unitary gates and noise operators, enabling shared
dispatch for matrix and generator utilities.

Args:
    None

Returns:
    AbstractOperator: Abstract type for operator definitions.
"""
abstract type AbstractOperator end
"""
Abstract supertype for unitary gate operators.

This subtype is used for gates with well-defined unitary matrices and, optionally, Hamiltonian
generators for evolution.

Args:
    None

Returns:
    AbstractGate: Abstract type for unitary gates.
"""
abstract type AbstractGate <: AbstractOperator end
"""
Abstract supertype for non-unitary noise operators.

This subtype represents dissipative or non-unitary processes that still act as operators on
qubit states.

Args:
    None

Returns:
    AbstractNoise: Abstract type for noise operators.
"""
abstract type AbstractNoise <: AbstractOperator end

"""
Return the matrix representation of an operator.

This is the generic fallback and must be implemented for each concrete operator type.

Args:
    op (AbstractOperator): Operator instance.

Returns:
    AbstractMatrix: Matrix representation of the operator.

Raises:
    ErrorException: If the operator does not implement `matrix`.
"""
function matrix(op::AbstractOperator)
    error("matrix() not implemented for $(typeof(op))")
end

"""
Return the generator matrices for a gate.

This is the generic fallback and must be implemented for gates that support Hamiltonian-based
evolution.

Args:
    op (AbstractOperator): Operator instance.

Returns:
    Vector{AbstractMatrix}: Generator matrices for the operator.

Raises:
    ErrorException: If the operator does not implement `generator`.
"""
function generator(op::AbstractOperator)
    error("generator() not implemented for $(typeof(op))")
end

"""
Return the Hamiltonian coefficient for a gate generator.

This provides the scalar `c` such that the gate equals `exp(-i * c * Generator)`, with a default
value of `1.0` when no special scaling is required.

Args:
    op (AbstractOperator): Operator instance.

Returns:
    Float64: Scaling coefficient for the generator.
"""
function hamiltonian_coeff(op::AbstractOperator)
    return 1.0
end

"""
Report whether an operator is unitary.

This returns `true` by default and can be specialized for non-unitary operators.

Args:
    op (AbstractOperator): Operator instance.

Returns:
    Bool: `true` if the operator is unitary, otherwise `false`.
"""
function is_unitary(op::AbstractOperator)
    return true
end

# --- Basic Gates (1 Qubit) ---

"""
Identity gate on a single qubit.

This represents the 2x2 identity operator, useful as a no-op or placeholder in circuits.

Args:
    None

Returns:
    IdGate: Identity gate operator.
"""
struct IdGate <: AbstractGate end
matrix(::IdGate) = SMatrix{2,2,ComplexF64}(1, 0, 0, 1)
generator(::IdGate) = [matrix(IdGate())]

"""
Pauli-X gate on a single qubit.

This represents the bit-flip operator with matrix [[0,1],[1,0]].

Args:
    None

Returns:
    XGate: Pauli-X gate operator.
"""
struct XGate <: AbstractGate end
matrix(::XGate) = SMatrix{2,2,ComplexF64}(0, 1, 1, 0)
generator(::XGate) = [matrix(XGate())]

"""
Pauli-Y gate on a single qubit.

This represents the Y operator with matrix [[0,-i],[i,0]].

Args:
    None

Returns:
    YGate: Pauli-Y gate operator.
"""
struct YGate <: AbstractGate end
matrix(::YGate) = SMatrix{2,2,ComplexF64}(0, im, -im, 0)
generator(::YGate) = [matrix(YGate())]

"""
Pauli-Z gate on a single qubit.

This represents the phase-flip operator with matrix [[1,0],[0,-1]].

Args:
    None

Returns:
    ZGate: Pauli-Z gate operator.
"""
struct ZGate <: AbstractGate end
matrix(::ZGate) = SMatrix{2,2,ComplexF64}(1, 0, 0, -1)
generator(::ZGate) = [matrix(ZGate())]

"""
Hadamard gate on a single qubit.

This maps computational basis states to equal superpositions, using the standard Hadamard matrix.

Args:
    None

Returns:
    HGate: Hadamard gate operator.
"""
struct HGate <: AbstractGate end
matrix(::HGate) = SMatrix{2,2,ComplexF64}(1/sqrt(2), 1/sqrt(2), 1/sqrt(2), -1/sqrt(2))

"""
Phase S gate on a single qubit.

This applies a π/2 phase to the |1⟩ state, with matrix diag(1, i).

Args:
    None

Returns:
    SGate: S gate operator.
"""
struct SGate <: AbstractGate end
matrix(::SGate) = SMatrix{2,2,ComplexF64}(1, 0, 0, im)

"""
Phase T gate on a single qubit.

This applies a π/4 phase to the |1⟩ state, with matrix diag(1, exp(iπ/4)).

Args:
    None

Returns:
    TGate: T gate operator.
"""
struct TGate <: AbstractGate end
matrix(::TGate) = SMatrix{2,2,ComplexF64}(1, 0, 0, exp(im*π/4))

"""
Adjoint of the S gate.

This applies a -π/2 phase to the |1⟩ state, with matrix diag(1, -i).

Args:
    None

Returns:
    SdgGate: S-dagger gate operator.
"""
struct SdgGate <: AbstractGate end
matrix(::SdgGate) = SMatrix{2,2,ComplexF64}(1, 0, 0, -im)

"""
Adjoint of the T gate.

This applies a -π/4 phase to the |1⟩ state, with matrix diag(1, exp(-iπ/4)).

Args:
    None

Returns:
    TdgGate: T-dagger gate operator.
"""
struct TdgGate <: AbstractGate end
matrix(::TdgGate) = SMatrix{2,2,ComplexF64}(1, 0, 0, exp(-im*π/4))

# --- Non-Unitary / Noise Operators ---

"""
Raising (creation) operator on a single qubit.

This represents the |1⟩⟨0| operator used for non-unitary processes and noise modeling.

Args:
    None

Returns:
    RaisingGate: Raising operator.
"""
struct RaisingGate <: AbstractOperator end
matrix(::RaisingGate) = SMatrix{2,2,ComplexF64}(0, 1, 0, 0) # |1><0| = [0 0; 1 0] (Col-major: 0, 1, 0, 0)

"""
Lowering (annihilation) operator on a single qubit.

This represents the |0⟩⟨1| operator used for non-unitary processes and noise modeling.

Args:
    None

Returns:
    LoweringGate: Lowering operator.
"""
struct LoweringGate <: AbstractOperator end
matrix(::LoweringGate) = SMatrix{2,2,ComplexF64}(0, 0, 1, 0) # |0><1| = [0 1; 0 0] (Col-major: 0, 0, 1, 0)

is_unitary(::RaisingGate) = false
is_unitary(::LoweringGate) = false

# --- Parametric Gates (1 Qubit) ---

"""
Rotation around the X axis.

This applies a single-qubit rotation `exp(-i * theta/2 * X)` with parameter `theta`.

Args:
    theta (Float64): Rotation angle.

Returns:
    RxGate: Parameterized X-rotation gate.
"""
struct RxGate <: AbstractGate
    theta::Float64
end
function matrix(g::RxGate)
    c = cos(g.theta/2)
    s = -im*sin(g.theta/2)
    return SMatrix{2,2,ComplexF64}(c, s, s, c)
end
generator(g::RxGate) = [matrix(XGate())]
hamiltonian_coeff(g::RxGate) = g.theta / 2.0

"""
Rotation around the Y axis.

This applies a single-qubit rotation `exp(-i * theta/2 * Y)` with parameter `theta`.

Args:
    theta (Float64): Rotation angle.

Returns:
    RyGate: Parameterized Y-rotation gate.
"""
struct RyGate <: AbstractGate
    theta::Float64
end
function matrix(g::RyGate)
    c = cos(g.theta/2)
    s = sin(g.theta/2)
    return SMatrix{2,2,ComplexF64}(c, s, -s, c)
end
generator(g::RyGate) = [matrix(YGate())]
hamiltonian_coeff(g::RyGate) = g.theta / 2.0

"""
Rotation around the Z axis.

This applies a single-qubit rotation `exp(-i * theta/2 * Z)` with parameter `theta`.

Args:
    theta (Float64): Rotation angle.

Returns:
    RzGate: Parameterized Z-rotation gate.
"""
struct RzGate <: AbstractGate
    theta::Float64
end
function matrix(g::RzGate)
    e_m = exp(-im*g.theta/2)
    e_p = exp(im*g.theta/2)
    return SMatrix{2,2,ComplexF64}(e_m, 0, 0, e_p)
end
generator(g::RzGate) = [matrix(ZGate())]
hamiltonian_coeff(g::RzGate) = g.theta / 2.0

"""
Single-qubit phase gate with angle theta.

This applies a phase `exp(i * theta)` to the |1⟩ state while leaving |0⟩ unchanged.

Args:
    theta (Float64): Phase angle.

Returns:
    PhaseGate: Parameterized phase gate.
"""
struct PhaseGate <: AbstractGate
    theta::Float64
end
matrix(g::PhaseGate) = SMatrix{2,2,ComplexF64}(1, 0, 0, exp(im*g.theta))

"""
Generic single-qubit U gate with three Euler angles.

This applies the standard `U(θ, φ, λ)` gate as used in Qiskit, combining Rz and Ry rotations.

Args:
    theta (Float64): Polar rotation angle.
    phi (Float64): First azimuthal rotation angle.
    lam (Float64): Second azimuthal rotation angle.

Returns:
    UGate: Parameterized U3 gate.
"""
struct UGate <: AbstractGate
    theta::Float64
    phi::Float64
    lam::Float64
end
function matrix(g::UGate)
    cos_val = cos(g.theta / 2)
    sin_val = sin(g.theta / 2)
    return SMatrix{2,2,ComplexF64}(
        cos_val, 
        exp(im * g.phi) * sin_val,
        -exp(im * g.lam) * sin_val,
        exp(im * (g.phi + g.lam)) * cos_val
    )
end

# --- 2 Qubit Gates ---

"""
Controlled-X (CNOT) gate.

This applies an X gate on the target qubit when the control qubit is |1⟩.

Args:
    None

Returns:
    CXGate: Controlled-X gate operator.
"""
struct CXGate <: AbstractGate end
matrix(::CXGate) = SMatrix{4,4,ComplexF64}(
    1,0,0,0,
    0,1,0,0,
    0,0,0,1,
    0,0,1,0
)
# Generator: π/4 * (I-Z) ⊗ (I-X)
# gen[1] = (I - Z) = [[0, 0], [0, 2]] in column-major
# gen[2] = (I - X) = [[1, -1], [-1, 1]] in column-major
function generator(::CXGate)
    I_minus_Z = SMatrix{2,2,ComplexF64}(0, 0, 0, 2)  # (I - Z)
    I_minus_X = SMatrix{2,2,ComplexF64}(1, -1, -1, 1)  # (I - X)
    return [I_minus_Z, I_minus_X]
end
hamiltonian_coeff(::CXGate) = π / 4.0

"""
Controlled-Y gate.

This applies a Y gate on the target qubit when the control qubit is |1⟩.

Args:
    None

Returns:
    CYGate: Controlled-Y gate operator.
"""
struct CYGate <: AbstractGate end
matrix(::CYGate) = SMatrix{4,4,ComplexF64}(
    1,0,0,0,
    0,1,0,0,
    0,0,0,im,
    0,0,-im,0
)

"""
Controlled-Z gate.

This applies a Z phase flip on the target qubit when the control qubit is |1⟩.

Args:
    None

Returns:
    CZGate: Controlled-Z gate operator.
"""
struct CZGate <: AbstractGate end
matrix(::CZGate) = SMatrix{4,4,ComplexF64}(
    1,0,0,0,
    0,1,0,0,
    0,0,1,0,
    0,0,0,-1
)
# Generator: π/4 * (I-Z) ⊗ (I-Z)
# CZ = exp(-i π/4 (I-Z)⊗(I-Z))
# The term (I-Z)⊗(I-Z) acts as:
# |00>: (1-1)(1-1) = 0
# |01>: (1-1)(1-(-1)) = 0
# |10>: (1-(-1))(1-1) = 0
# |11>: (1-(-1))(1-(-1)) = 4
# So exp(-i π/4 * 4) = exp(-i π) = -1 on |11>, 1 elsewhere. Matches CZ.
function generator(::CZGate)
    I_minus_Z = SMatrix{2,2,ComplexF64}(0, 0, 0, 2)  # (I - Z) = [0 0; 0 2]
    return [I_minus_Z, I_minus_Z]
end
hamiltonian_coeff(::CZGate) = π / 4.0

"""
Controlled-Hadamard gate.

This applies a Hadamard on the target qubit when the control qubit is |1⟩.

Args:
    None

Returns:
    CHGate: Controlled-Hadamard gate operator.
"""
struct CHGate <: AbstractGate end
matrix(::CHGate) = SMatrix{4,4,ComplexF64}(
    1,0,0,0,
    0,1,0,0,
    0,0,1/sqrt(2),1/sqrt(2),
    0,0,1/sqrt(2),-1/sqrt(2)
)

"""
Controlled phase gate with angle theta.

This applies a phase `exp(i * theta)` to the |11⟩ state while leaving other basis states unchanged.

Args:
    theta (Float64): Phase angle.

Returns:
    CPhaseGate: Parameterized controlled-phase gate.
"""
struct CPhaseGate <: AbstractGate
    theta::Float64
end
matrix(g::CPhaseGate) = SMatrix{4,4,ComplexF64}(
    1,0,0,0,
    0,1,0,0,
    0,0,1,0,
    0,0,0,exp(im*g.theta)
)
# CPhase(theta) = diag(1, 1, 1, exp(i theta))
#               = exp(i theta/4) * exp(-i theta/4 (I-Z)⊗(I-Z))
# The trace of the generator must be traceless for SU(4), but here we match the phase.
# Wait, let's use the same logic as CZ.
# CZ = diag(1,1,1,-1) = exp(-i pi/4 (I-Z)(I-Z)) (up to global phase?)
# Let's check phase for CZ:
# exp(-i pi/4 * 0) = 1
# exp(-i pi/4 * 4) = -1. Correct.
#
# For CPhase(theta):
# We want diag(1,1,1, exp(i theta)).
# Using generator G = (I-Z)⊗(I-Z). Eigenvalues: 0, 0, 0, 4.
# exp(-i alpha * G) -> 1, 1, 1, exp(-4i alpha).
# Set -4 alpha = theta => alpha = -theta/4.
# So coeff = -theta/4.
# Wait, standard convention here is exp(-i * coeff * G).
# So coeff should be theta/4 and we need a sign change in G or coeff?
# If we use G=(I-Z)(I-Z), eigenvalues are {0,0,0,4}.
# exp(-i c * 4) = exp(i theta) => -4c = theta => c = -theta/4.
#
# But usually we define positive coeff. Let's use G = -(I-Z)(I-Z)?
# No, let's just return negative coeff.
function generator(::CPhaseGate)
    I_minus_Z = SMatrix{2,2,ComplexF64}(0, 0, 0, 2)
    return [I_minus_Z, I_minus_Z]
end
hamiltonian_coeff(g::CPhaseGate) = -g.theta / 4.0

"""
SWAP gate.

This swaps the states of two qubits.

Args:
    None

Returns:
    SWAPGate: SWAP gate operator.
"""
struct SWAPGate <: AbstractGate end
matrix(::SWAPGate) = SMatrix{4,4,ComplexF64}(
    1,0,0,0,
    0,0,1,0,
    0,1,0,0,
    0,0,0,1
)

"""
iSWAP gate.

This swaps two qubits and adds a phase of i to the |01⟩ and |10⟩ components.

Args:
    None

Returns:
    iSWAPGate: iSWAP gate operator.
"""
struct iSWAPGate <: AbstractGate end
matrix(::iSWAPGate) = SMatrix{4,4,ComplexF64}(
    1,0,0,0,
    0,0,im,0,
    0,im,0,0,
    0,0,0,1
)

# --- Hamiltonian Evolution Gates (Rxx, Ryy, Rzz) ---
# e^{-i theta/2 P \otimes P}

"""
Two-qubit XX rotation gate.

This applies `exp(-i * theta/2 * X ⊗ X)` with rotation angle `theta`.

Args:
    theta (Float64): Rotation angle.

Returns:
    RxxGate: Parameterized XX rotation gate.
"""
struct RxxGate <: AbstractGate
    theta::Float64
end
function matrix(g::RxxGate)
    c = cos(g.theta / 2)
    s = -im * sin(g.theta / 2)
    return SMatrix{4,4,ComplexF64}(
        c,0,0,s,
        0,c,s,0,
        0,s,c,0,
        s,0,0,c
    )
end
generator(::RxxGate) = [matrix(XGate()), matrix(XGate())]
hamiltonian_coeff(g::RxxGate) = g.theta / 2.0

"""
Two-qubit YY rotation gate.

This applies `exp(-i * theta/2 * Y ⊗ Y)` with rotation angle `theta`.

Args:
    theta (Float64): Rotation angle.

Returns:
    RyyGate: Parameterized YY rotation gate.
"""
struct RyyGate <: AbstractGate
    theta::Float64
end
function matrix(g::RyyGate)
    c = cos(g.theta / 2)
    s = -im * sin(g.theta / 2)
    return SMatrix{4,4,ComplexF64}(
        c,0,0,-s,
        0,c,s,0,
        0,s,c,0,
        -s,0,0,c
    )
end
generator(::RyyGate) = [matrix(YGate()), matrix(YGate())]
hamiltonian_coeff(g::RyyGate) = g.theta / 2.0

"""
Two-qubit ZZ rotation gate.

This applies `exp(-i * theta/2 * Z ⊗ Z)` with rotation angle `theta`.

Args:
    theta (Float64): Rotation angle.

Returns:
    RzzGate: Parameterized ZZ rotation gate.
"""
struct RzzGate <: AbstractGate
    theta::Float64
end
function matrix(g::RzzGate)
    # exp(-i theta/2 Z Z)
    # diagonals: exp(-i theta/2), exp(i theta/2), exp(i theta/2), exp(-i theta/2)
    e_m = exp(-im * g.theta / 2)
    e_p = exp(im * g.theta / 2)
    return SMatrix{4,4,ComplexF64}(
        e_m,0,0,0,
        0,e_p,0,0,
        0,0,e_p,0,
        0,0,0,e_m
    )
end
generator(::RzzGate) = [matrix(ZGate()), matrix(ZGate())]
hamiltonian_coeff(g::RzzGate) = g.theta / 2.0

# --- Barrier ---
"""
Barrier gate to separate layers or trigger sampling.

This is a no-op gate used to mark circuit boundaries or sampling points via its label.

Args:
    label (String): Barrier label used by processing logic.

Returns:
    Barrier: Barrier gate operator.
"""
struct Barrier <: AbstractGate
    label::String
end
matrix(::Barrier) = error("Barrier has no matrix representation")
generator(::Barrier) = nothing
is_unitary(::Barrier) = true # Effectively Identity
hamiltonian_coeff(::Barrier) = 0.0

end # module
