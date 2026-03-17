"""
Strict gate-level validation for Israel -> PauliPropagation mapping.

Checks:
1) U(θ,φ,λ) vs Rz(φ)Ry(θ)Rz(λ) on random states
2) CX truth-table and Pauli-basis action for ordering conventions
3) Tiny extracted subcircuits: dense Qiskit vs dense mapped simulation
"""

using LinearAlgebra
using Random
using Dates
using Printf
using Statistics
using PythonCall
using PauliPropagation

include(joinpath(@__DIR__, "load_circuit.jl"))

const DATA_DIR = joinpath(@__DIR__, "max")
const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

const CIRCUITS = ["4q", "6q"]
const RNG = MersenneTwister(1234)

@inline function _phase_align(a::AbstractVector{ComplexF64}, b::AbstractVector{ComplexF64})
    ov = dot(a, b)
    if abs(ov) < 1e-30
        return copy(b)
    end
    return b .* conj(ov / abs(ov))
end

@inline function _state_err(a::AbstractVector{ComplexF64}, b::AbstractVector{ComplexF64})
    bal = _phase_align(a, b)
    return norm(a - bal)
end

function random_state(n::Int)
    v = randn(RNG, ComplexF64, n)
    v ./= norm(v)
    return v
end

function qiskit_u_matrix(θ::Float64, φ::Float64, λ::Float64)
    qiskit_lib = pyimport("qiskit.circuit.library")
    U = qiskit_lib.UGate(θ, φ, λ).to_matrix()
    return pyconvert(Matrix{ComplexF64}, U)
end

function pp_u_matrix(θ::Float64, φ::Float64, λ::Float64)
    rz1 = tomatrix(PauliRotation(:Z, 1), φ)
    ry = tomatrix(PauliRotation(:Y, 1), θ)
    rz2 = tomatrix(PauliRotation(:Z, 1), λ)
    return Matrix{ComplexF64}(rz1 * ry * rz2)
end

@inline function apply_1q!(ψ::Vector{ComplexF64}, U::Matrix{ComplexF64}, q::Int, n::Int)
    mask = 1 << q
    step = mask << 1
    @inbounds for base in 0:(step):(2^n - 1)
        for off in 0:(mask - 1)
            i0 = base + off
            i1 = i0 + mask
            a = ψ[i0 + 1]
            b = ψ[i1 + 1]
            ψ[i0 + 1] = U[1, 1] * a + U[1, 2] * b
            ψ[i1 + 1] = U[2, 1] * a + U[2, 2] * b
        end
    end
end

@inline function apply_cx!(ψ::Vector{ComplexF64}, c::Int, t::Int, n::Int)
    cmask = 1 << c
    tmask = 1 << t
    @inbounds for i in 0:(2^n - 1)
        if ((i & cmask) != 0) && ((i & tmask) == 0)
            j = i | tmask
            ψ[i + 1], ψ[j + 1] = ψ[j + 1], ψ[i + 1]
        end
    end
end

function cnot_matrix_little_endian(control::Int, target::Int, n::Int)
    U = Matrix{ComplexF64}(I, 2^n, 2^n)
    for col in 0:(2^n - 1)
        row = col
        if (((col >> control) & 1) == 1)
            row = col ⊻ (1 << target)
        end
        if row != col
            U[:, col + 1] .= 0
            U[row + 1, col + 1] = 1
        end
    end
    return U
end

function pauli_1q(sym::Symbol)
    if sym == :I
        return ComplexF64[1 0; 0 1]
    elseif sym == :X
        return ComplexF64[0 1; 1 0]
    elseif sym == :Y
        return ComplexF64[0 -im; im 0]
    elseif sym == :Z
        return ComplexF64[1 0; 0 -1]
    end
    error("unknown Pauli")
end

function pauli_term_matrix(term, n::Int)
    mats = Matrix{ComplexF64}[]
    # Qubit 1 in PauliPropagation is least-significant in bit indexing.
    for q in n:-1:1
        push!(mats, pauli_1q(inttosymbol(getpauli(term, q))))
    end
    out = mats[1]
    for k in 2:length(mats)
        out = kron(out, mats[k])
    end
    return out
end

function decompose_to_pauli_term(M::Matrix{ComplexF64}, n::Int)
    best_abs = -1.0
    best_term = zero(UInt8)
    best_coeff = 0.0 + 0.0im
    for term in 0:(4^n - 1)
        t = UInt8(term)
        P = pauli_term_matrix(t, n)
        c = tr(P' * M) / (2.0^n)
        if abs(c) > best_abs
            best_abs = abs(c)
            best_term = t
            best_coeff = c
        end
    end
    return best_term, best_coeff, best_abs
end

function validate_u_mapping(qc_list::Vector{Py}; samples::Int=40, states_per_sample::Int=8)
    params = Tuple{Float64,Float64,Float64}[]
    for qc in qc_list
        for instr in qc.data
            name = pyconvert(String, instr.operation.name)
            if name == "u"
                θ = pyconvert(Float64, instr.operation.params[0])
                φ = pyconvert(Float64, instr.operation.params[1])
                λ = pyconvert(Float64, instr.operation.params[2])
                push!(params, (θ, φ, λ))
            end
        end
    end
    @assert !isempty(params)

    chosen = [params[rand(RNG, 1:length(params))] for _ in 1:samples]
    # add random points as stress
    for _ in 1:5
        push!(chosen, (2π * rand(RNG), 2π * rand(RNG), 2π * rand(RNG)))
    end

    state_errs = Float64[]
    op_errs = Float64[]
    for (θ, φ, λ) in chosen
        Uq = qiskit_u_matrix(θ, φ, λ)
        Up = pp_u_matrix(θ, φ, λ)

        # operator error up to global phase
        α = tr(Uq' * Up)
        phase = abs(α) > 1e-30 ? α / abs(α) : 1.0 + 0im
        push!(op_errs, norm(Uq - Up * conj(phase)))

        for _ in 1:states_per_sample
            ψ = random_state(2)
            out_q = Uq * ψ
            out_p = Up * ψ
            push!(state_errs, _state_err(out_q, out_p))
        end
    end

    return (
        num_samples = length(chosen),
        state_max = maximum(state_errs),
        state_mean = mean(state_errs),
        op_max = maximum(op_errs),
        op_mean = mean(op_errs),
    )
end

function validate_cx_truth_and_pauli()
    # Qiskit reference for CX(control=0,target=1)
    Ucx = cnot_matrix_little_endian(0, 1, 2)

    # Truth-table check with bit convention (q0 LSB)
    truth_ok = true
    for b in 0:3
        q0 = (b >> 0) & 1
        q1 = (b >> 1) & 1
        expected_q0 = q0
        expected_q1 = q1 ⊻ q0
        expected = expected_q0 | (expected_q1 << 1)

        e = zeros(ComplexF64, 4)
        e[b + 1] = 1
        out = Ucx * e
        got = argmax(abs.(out)) - 1
        truth_ok &= (got == expected)
    end

    # Pauli-basis action check
    mismatches = String[]
    for s1 in (:I, :X, :Y, :Z), s2 in (:I, :X, :Y, :Z)
        p_in = PauliString(2, [s1, s2], [1, 2], 1.0)
        p_out = propagate([CliffordGate(:CNOT, [1, 2])], p_in; heisenberg=true)
        @assert length(p_out) == 1
        pp_term = first(paulis(p_out))
        pp_coeff = first(coefficients(p_out))

        Pin = pauli_term_matrix(symboltoint(2, [s1, s2], [1, 2]), 2)
        M = Ucx' * Pin * Ucx
        q_term, q_coeff, q_abs = decompose_to_pauli_term(M, 2)

        # should be one Pauli term with unit magnitude
        if !(abs(abs(q_coeff) - 1.0) < 1e-10 && abs(pp_coeff - q_coeff) < 1e-10 && pp_term == q_term)
            push!(mismatches,
                "$(s1)$(s2): pp=$(inttostring(pp_term,2)), coeff=$(pp_coeff); q=$(inttostring(q_term,2)), coeff=$(q_coeff), dom=$(q_abs)")
        end
    end

    return (
        truth_ok = truth_ok,
        pauli_mismatch_count = length(mismatches),
        pauli_mismatches = mismatches,
    )
end

function extract_tiny_subcircuit(qc::Py; max_qubits::Int=4, max_gates::Int=24)
    pool = Int[]
    selected = Vector{Tuple{String,Vector{Int},Vector{Float64}}}()
    for instr in qc.data
        name = pyconvert(String, instr.operation.name)
        name in ("u", "cx") || continue
        qs = [pyconvert(Int, q._index) for q in instr.qubits]
        union_pool = sort(unique(vcat(pool, qs)))
        if length(union_pool) <= max_qubits
            pool = union_pool
            params = [pyconvert(Float64, p) for p in instr.operation.params]
            push!(selected, (name, qs, params))
            if length(selected) >= max_gates
                break
            end
        elseif !isempty(pool) && all(q in pool for q in qs)
            params = [pyconvert(Float64, p) for p in instr.operation.params]
            push!(selected, (name, qs, params))
            if length(selected) >= max_gates
                break
            end
        end
    end
    qmap = Dict(q => i - 1 for (i, q) in enumerate(sort(pool))) # local 0-based
    return selected, sort(pool), qmap
end

function qiskit_state_for_tiny(selected, qmap, nsmall::Int)
    qiskit = pyimport("qiskit")
    qi = pyimport("qiskit.quantum_info")
    qc_small = qiskit.QuantumCircuit(nsmall)
    for (name, qs, params) in selected
        if name == "u"
            q = qmap[qs[1]]
            qc_small.u(params[1], params[2], params[3], q)
        elseif name == "cx"
            c = qmap[qs[1]]
            t = qmap[qs[2]]
            qc_small.cx(c, t)
        end
    end
    sv = qi.Statevector.from_label("0"^nsmall).evolve(qc_small).data
    return pyconvert(Vector{ComplexF64}, sv), qc_small
end

function mapped_dense_state_for_tiny(selected, qmap, nsmall::Int)
    ψ = zeros(ComplexF64, 2^nsmall)
    ψ[1] = 1.0 + 0im
    for (name, qs, params) in selected
        if name == "u"
            q = qmap[qs[1]]
            U = pp_u_matrix(params[1], params[2], params[3])
            apply_1q!(ψ, U, q, nsmall)
        elseif name == "cx"
            c = qmap[qs[1]]
            t = qmap[qs[2]]
            apply_cx!(ψ, c, t, nsmall)
        end
    end
    return ψ
end

function validate_tiny_subcircuits(qc_list::Vector{Tuple{String,Py}})
    rows = NamedTuple[]
    for (label, qc) in qc_list
        selected, pool, qmap = extract_tiny_subcircuit(qc; max_qubits=4, max_gates=24)
        nsmall = length(pool)
        @assert nsmall >= 2

        ψ_q, qc_small = qiskit_state_for_tiny(selected, qmap, nsmall)
        ψ_m = mapped_dense_state_for_tiny(selected, qmap, nsmall)
        err = _state_err(ψ_q, ψ_m)

        push!(rows, (
            label = label,
            nsmall = nsmall,
            ngates = length(selected),
            pool = pool,
            err = err,
        ))
    end
    return rows
end

function main()
    println("Gate-level validation — $(Dates.now())")
    qc_list = Tuple{String,Py}[]
    for label in CIRCUITS
        qpy_path = joinpath(DATA_DIR, "circuit $(label).qpy")
        qc = load_qpy_circuit(qpy_path)
        push!(qc_list, (label, qc))
    end

    ures = validate_u_mapping(last.(qc_list))
    cxres = validate_cx_truth_and_pauli()
    tiny = validate_tiny_subcircuits(qc_list)

    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_txt = joinpath(RESULTS_DIR, "gate_level_validation_$(ts).txt")
    out_json = joinpath(RESULTS_DIR, "gate_level_validation_$(ts).json")

    open(out_txt, "w") do io
        println(io, "Gate-level validation summary")
        println(io, "Generated: $(Dates.now())")
        println(io)
        println(io, "[1] U gate mapping: Qiskit U vs Rz*Ry*Rz")
        @printf(io, "  samples=%d\n", ures.num_samples)
        @printf(io, "  state error: mean=%.3e max=%.3e\n", ures.state_mean, ures.state_max)
        @printf(io, "  operator error (phase-aligned): mean=%.3e max=%.3e\n", ures.op_mean, ures.op_max)
        println(io)
        println(io, "[2] CX checks")
        println(io, "  truth table check (q0 LSB, control=0,target=1): $(cxres.truth_ok)")
        println(io, "  Pauli-basis mismatch count: $(cxres.pauli_mismatch_count)")
        for mm in cxres.pauli_mismatches
            println(io, "    mismatch: $mm")
        end
        println(io)
        println(io, "[3] Tiny extracted subcircuits (dense state)")
        for r in tiny
            @printf(io, "  %s: nsmall=%d ngates=%d pool=%s state_err=%.3e\n",
                    r.label, r.nsmall, r.ngates, string(r.pool), r.err)
        end
    end

    open(out_json, "w") do io
        println(io, "{")
        @printf(io, "  \"generated\": \"%s\",\n", string(Dates.now()))
        @printf(io, "  \"u_mapping\": {\"samples\": %d, \"state_mean\": %.16e, \"state_max\": %.16e, \"op_mean\": %.16e, \"op_max\": %.16e},\n",
                ures.num_samples, ures.state_mean, ures.state_max, ures.op_mean, ures.op_max)
        @printf(io, "  \"cx\": {\"truth_ok\": %s, \"pauli_mismatch_count\": %d},\n",
                string(cxres.truth_ok), cxres.pauli_mismatch_count)
        println(io, "  \"tiny_subcircuits\": [")
        for (i, r) in enumerate(tiny)
            @printf(io, "    {\"label\": \"%s\", \"nsmall\": %d, \"ngates\": %d, \"pool\": \"%s\", \"state_err\": %.16e}%s\n",
                    r.label, r.nsmall, r.ngates, string(r.pool), r.err, i < length(tiny) ? "," : "")
        end
        println(io, "  ]")
        println(io, "}")
    end

    println("Saved:")
    println("  $out_txt")
    println("  $out_json")
end

main()
