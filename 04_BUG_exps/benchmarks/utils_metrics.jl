"""
    BenchmarkMetrics

Error metrics for comparing MPS results against exact references.
"""
module BenchmarkMetrics

using LinearAlgebra
using Yaqs
const MPSMod = Yaqs.MPSModule

export mps_to_statevector, infidelity, maxZ_error, energy_error

"""
    mps_to_statevector(psi::MPS) -> Vector{ComplexF64}

Convert an MPS to a full statevector by contracting all tensors.
Only feasible for small N (e.g. N=12 → 4096 entries).
Uses the existing `MPSModule.to_vec`.
"""
function mps_to_statevector(psi::MPSMod.MPS)
    return MPSMod.to_vec(psi)
end

"""
    infidelity(psi_mps_vec, psi_exact_vec) -> Float64

Compute 1 - |⟨ψ_exact|ψ_mps⟩|² from two full statevectors.
"""
function infidelity(psi_mps_vec::AbstractVector{<:Complex},
                    psi_exact_vec::AbstractVector{<:Complex})
    @assert length(psi_mps_vec) == length(psi_exact_vec)
    overlap = dot(psi_exact_vec, psi_mps_vec)
    return 1.0 - abs2(overlap)
end

"""
    maxZ_error(z_mps, z_exact) -> Float64

Maximum absolute local Z error over all sites at a single time:
    max_i |⟨Z_i⟩_mps - ⟨Z_i⟩_exact|

Both arguments are vectors of length N.
"""
function maxZ_error(z_mps::AbstractVector{<:Real}, z_exact::AbstractVector{<:Real})
    @assert length(z_mps) == length(z_exact)
    return maximum(abs.(z_mps .- z_exact))
end

"""
    energy_error(energy_mps_T, energy_exact_T) -> Float64

Absolute energy error at final time:
    |⟨H⟩_mps(T) - ⟨H⟩_exact(T)|
"""
function energy_error(energy_mps_T::Real, energy_exact_T::Real)
    return abs(energy_mps_T - energy_exact_T)
end

end # module BenchmarkMetrics