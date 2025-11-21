module GateLibrary

using StaticArrays
using LinearAlgebra

export AbstractOperator
export XGate, YGate, ZGate, HGate, IdGate, SGate, TGate, SXGate, RaisingGate, LoweringGate
export RxGate, RyGate, RzGate, PhaseGate, UGate
export CXGate, CZGate, SwapGate
export matrix

"""
    AbstractOperator

Base type for all quantum operators.
"""
abstract type AbstractOperator end

# --- Constant Gates (Singletons) ---

struct XGate <: AbstractOperator end
struct YGate <: AbstractOperator end
struct ZGate <: AbstractOperator end
struct HGate <: AbstractOperator end
struct IdGate <: AbstractOperator end
struct SGate <: AbstractOperator end
struct TGate <: AbstractOperator end
struct SXGate <: AbstractOperator end
struct RaisingGate <: AbstractOperator end
struct LoweringGate <: AbstractOperator end

# --- Parameterized Gates ---

struct RxGate{T<:Real} <: AbstractOperator
    theta::T
end

struct RyGate{T<:Real} <: AbstractOperator
    theta::T
end

struct RzGate{T<:Real} <: AbstractOperator
    theta::T
end

struct PhaseGate{T<:Real} <: AbstractOperator
    theta::T
end

"""
    UGate(theta, phi, lambda)

Generic single-qubit rotation gate U3.
"""
struct UGate{T<:Real} <: AbstractOperator
    theta::T
    phi::T
    lam::T
end

# --- Two Qubit Gates ---

struct CXGate <: AbstractOperator end
struct CZGate <: AbstractOperator end
struct SwapGate <: AbstractOperator end

# --- Matrix Definitions (Dispatch) ---
# Using SMatrix for zero-allocation 2x2 and 4x4 matrices.

const C128 = ComplexF64

# Pauli Matrices
function matrix(::XGate)
    return SMatrix{2,2,C128}(0, 1, 1, 0)
end

function matrix(::YGate)
    return SMatrix{2,2,C128}(0, 1im, -1im, 0)
end

function matrix(::ZGate)
    return SMatrix{2,2,C128}(1, 0, 0, -1)
end

function matrix(::IdGate)
    return SMatrix{2,2,C128}(1, 0, 0, 1)
end

function matrix(::HGate)
    inv_sqrt2 = 1 / sqrt(2)
    return SMatrix{2,2,C128}(inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2)
end

function matrix(::SXGate)
    # Sqrt(X) = 0.5 * [[1+i, 1-i], [1-i, 1+i]]
    a = 0.5 * (1 + 1im)
    b = 0.5 * (1 - 1im)
    return SMatrix{2,2,C128}(a, b, b, a)
end

function matrix(::SGate)
    return SMatrix{2,2,C128}(1, 0, 0, 1im)
end

function matrix(::TGate)
    return SMatrix{2,2,C128}(1, 0, 0, exp(1im * π/4))
end

function matrix(::RaisingGate)
    # |1> -> |0> (Spin Raising / Energy Relaxing in some contexts, usually sigma_plus)
    # [0 1; 0 0]
    # Col-Major: (1,1)=0, (2,1)=0, (1,2)=1, (2,2)=0
    return SMatrix{2,2,C128}(0, 0, 1, 0)
end

function matrix(::LoweringGate)
    # |0> -> |1> (Spin Lowering / Energy Exciting, usually sigma_minus)
    # [0 0; 1 0]
    # Col-Major: (1,1)=0, (2,1)=1, (1,2)=0, (2,2)=0
    return SMatrix{2,2,C128}(0, 1, 0, 0)
end

# Rotations
# Note: Julia SMatrix constructor is Column-Major: (1,1), (2,1), (1,2), (2,2)

function matrix(g::RxGate)
    c = cos(g.theta / 2)
    s = -1im * sin(g.theta / 2)
    # [c s; s c]
    return SMatrix{2,2,C128}(c, s, s, c)
end

function matrix(g::RyGate)
    c = cos(g.theta / 2)
    s = sin(g.theta / 2)
    # [c -s; s c] -> Col-Major: c, s, -s, c
    return SMatrix{2,2,C128}(c, s, -s, c)
end

function matrix(g::RzGate)
    e_m = exp(-1im * g.theta / 2)
    e_p = exp(1im * g.theta / 2)
    # [e- 0; 0 e+]
    return SMatrix{2,2,C128}(e_m, 0, 0, e_p)
end

function matrix(g::PhaseGate)
    # [1 0; 0 e^iθ]
    return SMatrix{2,2,C128}(1, 0, 0, exp(1im * g.theta))
end

function matrix(g::UGate)
    # U3 definition from Qiskit/Standard
    # [cos(t/2)          -e^(il)sin(t/2)]
    # [e^(ip)sin(t/2)    e^(i(p+l))cos(t/2)]
    
    t_2 = g.theta / 2
    c = cos(t_2)
    s = sin(t_2)
    
    # Col-Major inputs: (1,1), (2,1), (1,2), (2,2)
    return SMatrix{2,2,C128}(
        c, 
        exp(1im * g.phi) * s, 
        -exp(1im * g.lam) * s, 
        exp(1im * (g.phi + g.lam)) * c
    )
end

# Two Qubit Gates (4x4)
# Layout: |00>, |01>, |10>, |11> -> Indices 1, 2, 3, 4

function matrix(::CXGate)
    # CNOT (Control 0, Target 1)
    # [1 0 0 0]
    # [0 1 0 0]
    # [0 0 0 1]
    # [0 0 1 0]
    # Sparse constructor or direct?
    # Direct SMatrix 4x4
    return SMatrix{4,4,C128}(
        1,0,0,0, 
        0,1,0,0,
        0,0,0,1,
        0,0,1,0
    )
end

function matrix(::CZGate)
    # Diag(1, 1, 1, -1)
    return SMatrix{4,4,C128}(
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,-1
    )
end

function matrix(::SwapGate)
    # [1 0 0 0]
    # [0 0 1 0]
    # [0 1 0 0]
    # [0 0 0 1]
    return SMatrix{4,4,C128}(
        1,0,0,0,
        0,0,1,0,
        0,1,0,0,
        0,0,0,1
    )
end

end # module

