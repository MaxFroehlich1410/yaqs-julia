"""
    Local utilities for the Israel fidelity-check experiment.

    This file provides:
      - `amplitude(mps, bitstring)` : complex amplitude ⟨bitstring|ψ⟩
      - `amplitudes_for_bitstrings(mps, bitstrings)` : batch version
"""

using ..Yaqs.MPSModule: MPS, shift_orthogonality_center!

"""
    amplitude(mps::MPS, bitstring::String) -> ComplexF64

Compute the complex amplitude ⟨bitstring|ψ⟩ by contracting through the MPS.

`bitstring` is a string of '0' and '1' characters with `length(bitstring) == mps.length`.
`bitstring[i]` selects the physical index at MPS site `i`.
"""
function amplitude(mps::MPS{T}, bitstring::String) where T
    @assert length(bitstring) == mps.length "bitstring length $(length(bitstring)) != MPS length $(mps.length)"
    vec = ones(T, 1)
    for i in 1:mps.length
        idx = parse(Int, bitstring[i]) + 1  # '0' -> 1, '1' -> 2
        A = mps.tensors[i]  # (L, d, R)
        A_slice = @view A[:, idx, :]  # (L, R)
        vec = transpose(vec) * A_slice  # (1, R)
        vec = reshape(vec, size(A_slice, 2))
    end
    return vec[1]
end

"""
    amplitudes_for_bitstrings(mps, bitstrings; reverse_bits=false) -> Vector{ComplexF64}

Compute complex amplitudes for a list of bitstrings.

If `reverse_bits=true`, each bitstring is reversed before computing the amplitude,
which swaps between big-endian and little-endian qubit conventions.
"""
function amplitudes_for_bitstrings(mps::MPS, bitstrings::Vector{String}; reverse_bits::Bool=false)
    amps = Vector{ComplexF64}(undef, length(bitstrings))
    for (k, bs) in enumerate(bitstrings)
        bs_use = reverse_bits ? reverse(bs) : bs
        amps[k] = amplitude(mps, bs_use)
    end
    return amps
end
