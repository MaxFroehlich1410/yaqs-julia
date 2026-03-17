"""
Diagnose whether initial computational-basis convention explains Israel mismatch.

Approach:
- Use only diagonal observables (Z_i and selected Z_i Z_j),
  so reference eval depends only on |a_ref|^2 over provided bitstrings.
- Evaluate PauliPropagation in Heisenberg picture with overlap against
  candidate computational basis states inferred from reference bitstrings.
"""

using Dates
using Printf
using PythonCall
using Statistics

include(joinpath(@__DIR__, "load_circuit.jl"))
include(joinpath(@__DIR__, "local_pauli_utils.jl"))

const DATA_DIR = joinpath(@__DIR__, "max")
const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

const CIRCUITS = ["4q", "6q"]
const PP_MIN_ABS_COEFF = 1e-12

function diagonal_probe_set(qc::Py; max_qubits::Int=16, max_pairs::Int=16)
    qsel, psel = choose_probe_qubits_and_pairs(qc; max_qubits=max_qubits, max_pairs=max_pairs)
    obs = ObservableSpec[]
    for q in qsel
        push!(obs, ObservableSpec("Z_$q", [:Z], [q], "1q_Z"))
    end
    for (i, j) in psel
        push!(obs, ObservableSpec("ZZ_$(i)_$(j)", [:Z, :Z], [i, j], "2q_ZZ"))
    end
    return obs
end

function candidate_initial_states(bitstrings::Vector{String}, a_ref::Vector{ComplexF64})
    i_max = argmax(abs.(a_ref))
    bs_first = bitstrings[1]
    bs_peak = bitstrings[i_max]

    cands = NamedTuple[]
    for src in (("first", bs_first), ("peak", bs_peak))
        for rev in (false, true)
            for inv in (false, true)
                name = "$(src[1])_rev$(rev)_inv$(inv)"
                onebits = bitstring_to_onebitinds(src[2]; reverse_bits=rev, invert_bits=inv)
                push!(cands, (name=name, onebits=onebits, source=src[1], reverse=rev, invert=inv))
            end
        end
    end
    return cands
end

function score_candidate(obs, states, amp_dict, pp_circ, pp_params, nqubits, cand)
    rows = NamedTuple[]
    for o in obs
        ref_val = expectation_from_sparse_states(o.symbols, o.qinds, states, amp_dict)
        pp_val = expectation_pp_computational(
            pp_circ, pp_params, nqubits, o.symbols, o.qinds, cand.onebits;
            min_abs_coeff=PP_MIN_ABS_COEFF
        )
        push!(rows, (name=o.name, family=o.family, ref=ref_val, pp=pp_val, abs_err=abs(ref_val - pp_val)))
    end
    errs = [r.abs_err for r in rows]
    fam = summarize_by_family(rows)
    return (
        candidate = cand,
        mae = mean(errs),
        maxae = maximum(errs),
        rmse = sqrt(mean(abs2, errs)),
        fam = fam,
        rows = rows,
    )
end

function run_single(label::String)
    println("\n", "="^88)
    println("Initial-state sector diagnostic for $label")
    println("="^88)

    qc = load_qpy_circuit(joinpath(DATA_DIR, "circuit $(label).qpy"))
    bitstrings, a_ref = load_npz_reference(joinpath(DATA_DIR, "circuit $(label).npz"))
    nqubits = pyconvert(Int, qc.num_qubits)
    pp_circ, pp_params, op_names, _ = qiskit_to_pauli_propagation(qc)

    obs = diagonal_probe_set(qc)
    # diagonal-only, for reference parsing we test 4 conventions as before
    convs = (:native, :reversed, :native_inverted, :reversed_inverted)
    candidates = candidate_initial_states(bitstrings, a_ref)

    best = nothing
    all_scores = NamedTuple[]
    for conv in convs
        _, states, amp_dict = parse_states(bitstrings, a_ref, conv)
        for cand in candidates
            sc = score_candidate(obs, states, amp_dict, pp_circ, pp_params, nqubits, cand)
            push!(all_scores, (conv=conv, score=sc))
            if isnothing(best) || sc.mae < best.score.mae
                best = (conv=conv, score=sc)
            end
        end
    end

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out = joinpath(RESULTS_DIR, "sector_diagnostic_$(label)_$(ts).txt")
    sorted = sort(all_scores; by=x -> x.score.mae)

    open(out, "w") do io
        println(io, "Initial-state / sector diagnostic")
        println(io, "Generated: $(Dates.now())")
        println(io, "Circuit: $label")
        println(io, "Qiskit ops: $op_names")
        println(io, "Observable set: diagonal only (Z, ZZ), count=$(length(obs))")
        println(io)
        println(io, "Best combination:")
        println(io, "  reference bit convention: $(best.conv)")
        println(io, "  initial-state candidate: $(best.score.candidate.name)")
        println(io, "  onebit count: $(length(best.score.candidate.onebits))")
        @printf(io, "  MAE=%.3e MAX=%.3e RMSE=%.3e\n", best.score.mae, best.score.maxae, best.score.rmse)
        for fam in ("1q_Z", "2q_ZZ")
            if haskey(best.score.fam, fam)
                m = best.score.fam[fam]
                @printf(io, "  %s: MAE=%.3e MAX=%.3e RMSE=%.3e\n", fam, m.mae, m.maxae, m.rmse)
            end
        end
        println(io)
        println(io, "Top 10 combinations by MAE:")
        for (k, x) in enumerate(first(sorted, min(10, length(sorted))))
            @printf(io, "  %2d) conv=%s cand=%s  MAE=%.3e MAX=%.3e RMSE=%.3e\n",
                    k, string(x.conv), x.score.candidate.name, x.score.mae, x.score.maxae, x.score.rmse)
        end
    end

    println("Saved: $out")
    return out
end

function main()
    println("Initial-state sector diagnostic — $(Dates.now())")
    for label in CIRCUITS
        run_single(label)
    end
    println("Done.")
end

main()
