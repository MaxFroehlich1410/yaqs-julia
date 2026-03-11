"""
    Amplitude comparison and fidelity diagnostics for the Israel check.
"""

using Printf, Dates

struct ComparisonResult
    circuit_label::String
    method::String
    num_bitstrings::Int
    fidelity::Float64
    fidelity_loss::Float64
    expected_fidelity_loss::Float64
    abs_deviation::Float64
    max_abs_error::Float64
    mean_abs_error::Float64
    relative_norm_diff::Float64
    phase_aligned_max_error::Float64
    phase_aligned_mean_error::Float64
    detected_convention::String
    runtime_s::Float64
    max_bond_dim_used::Int
    truncation_threshold::Float64
end

"""
    compare_amplitudes(a_ref, a_circ; label="", method="")

Compute fidelity, fidelity loss, and diagnostic metrics comparing reference
amplitudes `a_ref` to circuit amplitudes `a_circ`.

Global phase is accounted for in the phase-aligned diagnostics.
"""
function compare_amplitudes(a_ref::Vector{ComplexF64}, a_circ::Vector{ComplexF64})
    @assert length(a_ref) == length(a_circ)

    overlap = dot(a_ref, a_circ)  # sum(conj(a_ref) .* a_circ)
    fidelity = abs2(overlap)
    fidelity_loss = 1.0 - fidelity

    errors = abs.(a_ref .- a_circ)
    max_abs_error = maximum(errors)
    mean_abs_error = sum(errors) / length(errors)

    norm_ref = sqrt(sum(abs2, a_ref))
    norm_circ = sqrt(sum(abs2, a_circ))
    relative_norm_diff = abs(norm_ref - norm_circ) / max(norm_ref, 1e-30)

    # Phase-aligned comparison: remove global phase
    if abs(overlap) > 1e-30
        phase = overlap / abs(overlap)
        a_circ_aligned = a_circ .* conj(phase)
    else
        a_circ_aligned = a_circ
    end
    aligned_errors = abs.(a_ref .- a_circ_aligned)
    phase_aligned_max_error = maximum(aligned_errors)
    phase_aligned_mean_error = sum(aligned_errors) / length(aligned_errors)

    return (
        overlap = overlap,
        fidelity = fidelity,
        fidelity_loss = fidelity_loss,
        max_abs_error = max_abs_error,
        mean_abs_error = mean_abs_error,
        relative_norm_diff = relative_norm_diff,
        phase_aligned_max_error = phase_aligned_max_error,
        phase_aligned_mean_error = phase_aligned_mean_error,
    )
end

"""
    determine_bitstring_convention(mps, bitstrings, a_ref)

Test both native and reversed bitstring orderings and return the one
that gives higher fidelity, along with both results for diagnostics.
"""
function determine_bitstring_convention(mps, bitstrings::Vector{String}, a_ref::Vector{ComplexF64})
    println("  Testing bitstring conventions...")

    a_native = amplitudes_for_bitstrings(mps, bitstrings; reverse_bits=false)
    res_native = compare_amplitudes(a_ref, a_native)
    println("    Native order:   fidelity = $(res_native.fidelity), fidelity_loss = $(res_native.fidelity_loss)")

    a_reversed = amplitudes_for_bitstrings(mps, bitstrings; reverse_bits=true)
    res_reversed = compare_amplitudes(a_ref, a_reversed)
    println("    Reversed order: fidelity = $(res_reversed.fidelity), fidelity_loss = $(res_reversed.fidelity_loss)")

    if res_native.fidelity >= res_reversed.fidelity
        println("    -> Using NATIVE ordering (bitstring[1] = site 1 = qubit 0)")
        return "native", false, a_native, res_native
    else
        println("    -> Using REVERSED ordering (bitstring[1] = site N, bitstring[end] = site 1)")
        return "reversed", true, a_reversed, res_reversed
    end
end

"""
    build_result(circuit_label, method, num_bs, res, expected_fl, convention, runtime_s, max_bond, trunc_thresh)
"""
function build_result(circuit_label, method, num_bs, res, expected_fl, convention, runtime_s, max_bond, trunc_thresh)
    abs_dev = abs(res.fidelity_loss - expected_fl)
    return ComparisonResult(
        circuit_label, method, num_bs,
        res.fidelity, res.fidelity_loss, expected_fl, abs_dev,
        res.max_abs_error, res.mean_abs_error, res.relative_norm_diff,
        res.phase_aligned_max_error, res.phase_aligned_mean_error,
        convention, runtime_s, max_bond, trunc_thresh,
    )
end

"""
    print_result(r::ComparisonResult)
"""
function print_result(r::ComparisonResult)
    println("=" ^ 72)
    @printf("  Circuit:                %s\n", r.circuit_label)
    @printf("  Method:                 %s\n", r.method)
    @printf("  Max bond dim:           %d\n", r.max_bond_dim_used)
    @printf("  Truncation threshold:   %.2e\n", r.truncation_threshold)
    @printf("  Num bitstrings:         %d\n", r.num_bitstrings)
    @printf("  Bit ordering:           %s\n", r.detected_convention)
    @printf("  Runtime:                %.2f s\n", r.runtime_s)
    println("-" ^ 72)
    @printf("  Fidelity:               %.15f\n", r.fidelity)
    @printf("  Fidelity loss:          %.15f\n", r.fidelity_loss)
    @printf("  Expected fidelity loss: %.15f\n", r.expected_fidelity_loss)
    @printf("  Absolute deviation:     %.2e\n", r.abs_deviation)
    println("-" ^ 72)
    @printf("  Max abs error:          %.2e\n", r.max_abs_error)
    @printf("  Mean abs error:         %.2e\n", r.mean_abs_error)
    @printf("  Relative norm diff:     %.2e\n", r.relative_norm_diff)
    @printf("  Phase-aligned max err:  %.2e\n", r.phase_aligned_max_error)
    @printf("  Phase-aligned mean err: %.2e\n", r.phase_aligned_mean_error)
    println("=" ^ 72)
end

"""
    save_results_json(results::Vector{ComparisonResult}, filepath::String)
"""
function save_results_json(results::Vector{ComparisonResult}, filepath::String)
    open(filepath, "w") do io
        println(io, "[")
        for (i, r) in enumerate(results)
            println(io, "  {")
            @printf(io, "    \"circuit_label\": \"%s\",\n", r.circuit_label)
            @printf(io, "    \"method\": \"%s\",\n", r.method)
            @printf(io, "    \"max_bond_dim\": %d,\n", r.max_bond_dim_used)
            @printf(io, "    \"truncation_threshold\": %.2e,\n", r.truncation_threshold)
            @printf(io, "    \"num_bitstrings\": %d,\n", r.num_bitstrings)
            @printf(io, "    \"detected_convention\": \"%s\",\n", r.detected_convention)
            @printf(io, "    \"runtime_s\": %.4f,\n", r.runtime_s)
            @printf(io, "    \"fidelity\": %.17e,\n", r.fidelity)
            @printf(io, "    \"fidelity_loss\": %.17e,\n", r.fidelity_loss)
            @printf(io, "    \"expected_fidelity_loss\": %.17e,\n", r.expected_fidelity_loss)
            @printf(io, "    \"abs_deviation\": %.17e,\n", r.abs_deviation)
            @printf(io, "    \"max_abs_error\": %.17e,\n", r.max_abs_error)
            @printf(io, "    \"mean_abs_error\": %.17e,\n", r.mean_abs_error)
            @printf(io, "    \"relative_norm_diff\": %.17e,\n", r.relative_norm_diff)
            @printf(io, "    \"phase_aligned_max_error\": %.17e,\n", r.phase_aligned_max_error)
            @printf(io, "    \"phase_aligned_mean_error\": %.17e\n", r.phase_aligned_mean_error)
            print(io, "  }")
            if i < length(results)
                println(io, ",")
            else
                println(io)
            end
        end
        println(io, "]")
    end
    println("Results saved to $filepath")
end

"""
    save_results_summary(results::Vector{ComparisonResult}, filepath::String)
"""
function save_results_summary(results::Vector{ComparisonResult}, filepath::String)
    open(filepath, "w") do io
        println(io, "Israel Fidelity Check — Summary")
        println(io, "Generated: $(Dates.now())")
        println(io, "=" ^ 72)
        for r in results
            println(io)
            @printf(io, "Circuit: %s | Method: %s\n", r.circuit_label, r.method)
            @printf(io, "  Max bond dim:           %d\n", r.max_bond_dim_used)
            @printf(io, "  Truncation threshold:   %.2e\n", r.truncation_threshold)
            @printf(io, "  Num bitstrings:         %d\n", r.num_bitstrings)
            @printf(io, "  Bit ordering:           %s\n", r.detected_convention)
            @printf(io, "  Runtime:                %.2f s\n", r.runtime_s)
            @printf(io, "  Fidelity:               %.15f\n", r.fidelity)
            @printf(io, "  Fidelity loss:          %.15f\n", r.fidelity_loss)
            @printf(io, "  Expected fidelity loss: %.15f\n", r.expected_fidelity_loss)
            @printf(io, "  Absolute deviation:     %.2e\n", r.abs_deviation)
            @printf(io, "  Max abs error:          %.2e\n", r.max_abs_error)
            @printf(io, "  Mean abs error:         %.2e\n", r.mean_abs_error)
            @printf(io, "  Relative norm diff:     %.2e\n", r.relative_norm_diff)
            @printf(io, "  Phase-aligned max err:  %.2e\n", r.phase_aligned_max_error)
            @printf(io, "  Phase-aligned mean err: %.2e\n", r.phase_aligned_mean_error)
            consistent = r.abs_deviation < 0.05
            @printf(io, "  Consistent with expected: %s (deviation %.2e)\n",
                    consistent ? "YES" : "NO", r.abs_deviation)
            println(io, "-" ^ 72)
        end
    end
    println("Summary saved to $filepath")
end
