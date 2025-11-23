module GateLibrary

using StaticArrays
using LinearAlgebra

export AbstractOperator, AbstractGate, AbstractNoise
export matrix, generator, is_unitary, hamiltonian_coeff
export XGate, YGate, ZGate, HGate, SGate, TGate, SdgGate, TdgGate
export RxGate, RyGate, RzGate, PhaseGate, UGate
export CXGate, CYGate, CZGate, CHGate, CPhaseGate, SWAPGate, iSWAPGate
export RxxGate, RyyGate, RzzGate
export Barrier

# --- Abstract Types ---

abstract type AbstractOperator end
abstract type AbstractGate <: AbstractOperator end
abstract type AbstractNoise <: AbstractOperator end

function matrix(op::AbstractOperator)
    error("matrix() not implemented for $(typeof(op))")
end

function generator(op::AbstractOperator)
    error("generator() not implemented for $(typeof(op))")
end

"""
    hamiltonian_coeff(op)

Returns the scaling coefficient `c` such that the gate is `exp(-i * c * Generator)`.
Default is 1.0.
"""
function hamiltonian_coeff(op::AbstractOperator)
    return 1.0
end

function is_unitary(op::AbstractOperator)
    return true
end

# --- Basic Gates (1 Qubit) ---

struct XGate <: AbstractGate end
matrix(::XGate) = SMatrix{2,2,ComplexF64}([0 1; 1 0])
generator(::XGate) = [matrix(XGate())]

struct YGate <: AbstractGate end
matrix(::YGate) = SMatrix{2,2,ComplexF64}([0 -im; im 0])
generator(::YGate) = [matrix(YGate())]

struct ZGate <: AbstractGate end
matrix(::ZGate) = SMatrix{2,2,ComplexF64}([1 0; 0 -1])
generator(::ZGate) = [matrix(ZGate())]

struct HGate <: AbstractGate end
matrix(::HGate) = SMatrix{2,2,ComplexF64}([1 1; 1 -1] ./ sqrt(2))

struct SGate <: AbstractGate end
matrix(::SGate) = SMatrix{2,2,ComplexF64}([1 0; 0 im])

struct TGate <: AbstractGate end
matrix(::TGate) = SMatrix{2,2,ComplexF64}([1 0; 0 exp(im*π/4)])

struct SdgGate <: AbstractGate end
matrix(::SdgGate) = SMatrix{2,2,ComplexF64}([1 0; 0 -im])

struct TdgGate <: AbstractGate end
matrix(::TdgGate) = SMatrix{2,2,ComplexF64}([1 0; 0 exp(-im*π/4)])

# --- Parametric Gates (1 Qubit) ---

struct RxGate <: AbstractGate
    theta::Float64
end
matrix(g::RxGate) = SMatrix{2,2,ComplexF64}([cos(g.theta/2) -im*sin(g.theta/2); -im*sin(g.theta/2) cos(g.theta/2)])
generator(g::RxGate) = [matrix(XGate())]
hamiltonian_coeff(g::RxGate) = g.theta / 2.0

struct RyGate <: AbstractGate
    theta::Float64
end
matrix(g::RyGate) = SMatrix{2,2,ComplexF64}([cos(g.theta/2) -sin(g.theta/2); sin(g.theta/2) cos(g.theta/2)])
generator(g::RyGate) = [matrix(YGate())]
hamiltonian_coeff(g::RyGate) = g.theta / 2.0

struct RzGate <: AbstractGate
    theta::Float64
end
matrix(g::RzGate) = SMatrix{2,2,ComplexF64}([exp(-im*g.theta/2) 0; 0 exp(im*g.theta/2)])
generator(g::RzGate) = [matrix(ZGate())]
hamiltonian_coeff(g::RzGate) = g.theta / 2.0

struct PhaseGate <: AbstractGate
    theta::Float64
end
matrix(g::PhaseGate) = SMatrix{2,2,ComplexF64}([1 0; 0 exp(im*g.theta)])

struct UGate <: AbstractGate
    theta::Float64
    phi::Float64
    lam::Float64
end
function matrix(g::UGate)
    cos_val = cos(g.theta / 2)
    sin_val = sin(g.theta / 2)
    return SMatrix{2,2,ComplexF64}([
        cos_val               -exp(im * g.lam) * sin_val;
        exp(im * g.phi) * sin_val  exp(im * (g.phi + g.lam)) * cos_val
    ])
end

# --- 2 Qubit Gates ---

struct CXGate <: AbstractGate end
matrix(::CXGate) = SMatrix{4,4,ComplexF64}([
    1 0 0 0;
    0 1 0 0;
    0 0 0 1;
    0 0 1 0
])

struct CYGate <: AbstractGate end
matrix(::CYGate) = SMatrix{4,4,ComplexF64}([
    1 0 0 0;
    0 1 0 0;
    0 0 0 -im;
    0 0 im 0
])

struct CZGate <: AbstractGate end
matrix(::CZGate) = SMatrix{4,4,ComplexF64}([
    1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 -1
])

struct CHGate <: AbstractGate end
matrix(::CHGate) = SMatrix{4,4,ComplexF64}([
    1 0 0 0;
    0 1 0 0;
    0 0 1/sqrt(2) 1/sqrt(2);
    0 0 1/sqrt(2) -1/sqrt(2)
])

struct CPhaseGate <: AbstractGate
    theta::Float64
end
matrix(g::CPhaseGate) = SMatrix{4,4,ComplexF64}([
    1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 exp(im*g.theta)
])

struct SWAPGate <: AbstractGate end
matrix(::SWAPGate) = SMatrix{4,4,ComplexF64}([
    1 0 0 0;
    0 0 1 0;
    0 1 0 0;
    0 0 0 1
])

struct iSWAPGate <: AbstractGate end
matrix(::iSWAPGate) = SMatrix{4,4,ComplexF64}([
    1 0 0 0;
    0 0 im 0;
    0 im 0 0;
    0 0 0 1
])

# --- Hamiltonian Evolution Gates (Rxx, Ryy, Rzz) ---
# e^{-i theta/2 P \otimes P}

struct RxxGate <: AbstractGate
    theta::Float64
end
function matrix(g::RxxGate)
    c = cos(g.theta / 2)
    s = -im * sin(g.theta / 2)
    return SMatrix{4,4,ComplexF64}([
        c 0 0 s;
        0 c s 0;
        0 s c 0;
        s 0 0 c
    ])
end
generator(::RxxGate) = [matrix(XGate()), matrix(XGate())]
hamiltonian_coeff(g::RxxGate) = g.theta / 2.0

struct RyyGate <: AbstractGate
    theta::Float64
end
function matrix(g::RyyGate)
    c = cos(g.theta / 2)
    s = -im * sin(g.theta / 2)
    return SMatrix{4,4,ComplexF64}([
        c 0 0 -s;
        0 c s 0;
        0 s c 0;
        -s 0 0 c
    ])
end
generator(::RyyGate) = [matrix(YGate()), matrix(YGate())]
hamiltonian_coeff(g::RyyGate) = g.theta / 2.0

struct RzzGate <: AbstractGate
    theta::Float64
end
function matrix(g::RzzGate)
    # exp(-i theta/2 Z Z)
    # diagonals: exp(-i theta/2), exp(i theta/2), exp(i theta/2), exp(-i theta/2)
    e_m = exp(-im * g.theta / 2)
    e_p = exp(im * g.theta / 2)
    return SMatrix{4,4,ComplexF64}([
        e_m 0 0 0;
        0 e_p 0 0;
        0 0 e_p 0;
        0 0 0 e_m
    ])
end
generator(::RzzGate) = [matrix(ZGate()), matrix(ZGate())]
hamiltonian_coeff(g::RzzGate) = g.theta / 2.0

# --- Barrier ---
struct Barrier <: AbstractGate
    label::String
end
matrix(::Barrier) = error("Barrier has no matrix representation")
generator(::Barrier) = nothing
is_unitary(::Barrier) = true # Effectively Identity
hamiltonian_coeff(::Barrier) = 0.0

end # module
