"""
    debug_pp_vs_tebd.jl — Systematic debugging of <Z_i> discrepancies

PHASE 1: PP vs TEBD on the same circuit (4q, 6q)
PHASE 2: Tiny hand-checkable gate-mapping tests
PHASE 3: Stress-test bitstring conventions (target vs TEBD vs PP)
PHASE 4: Active/inactive split analysis
PHASE 5: Fidelity interpretation check
"""

using LinearAlgebra
using Dates
using Printf
using Statistics
using Random

# ── Load Yaqs (TEBD, MPS, circuit ingestion) ─────────────────────────────
include(joinpath(@__DIR__, "..", "src", "Yaqs.jl"))
using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.GateLibrary
using .Yaqs.SimulationConfigs
using .Yaqs.DigitalTJM
using .Yaqs.CircuitIngestion: ingest_qiskit_circuit

# ── Load PauliPropagation ─────────────────────────────────────────────────
using PauliPropagation

# ── Load PythonCall helpers ───────────────────────────────────────────────
using PythonCall
include(joinpath(@__DIR__, "load_circuit.jl"))
include(joinpath(@__DIR__, "utils_local.jl"))

const DATA_DIR    = joinpath(@__DIR__, "max")
const REPORT_DIR  = joinpath(@__DIR__, "debug_reports")
mkpath(REPORT_DIR)

# ═══════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

function tebd_z(qc::Py; max_bond_dim=512, trunc=1e-12)
    digital_circuit = ingest_qiskit_circuit(qc)
    n = pyconvert(Int, qc.num_qubits)
    initial = MPS(n; state="zeros")
    sim = TimeEvolutionConfig(Observable[], 1.0;
        dt=1.0, num_traj=1, sample_timesteps=false,
        max_bond_dim=max_bond_dim, truncation_threshold=trunc)
    opts = TJMOptions(local_method=:TEBD, long_range_method=:TEBD)
    final_state, _, _ = run_digital_tjm(initial, digital_circuit, nothing, sim; alg_options=opts)
    Zop = Matrix(GateLibrary.matrix(ZGate()))
    z = zeros(Float64, n)
    for q in 1:n
        z[q] = real(MPSModule.local_expect(final_state, Zop, q))
    end
    return z, final_state
end

function qiskit_to_pp(qc::Py)
    gates  = Any[]
    params = Float64[]
    for instr in qc.data
        op   = instr.operation
        name = pyconvert(String, op.name)
        qargs = instr.qubits
        if name == "u"
            q = pyconvert(Int, qargs[0]._index) + 1
            θ = pyconvert(Float64, op.params[0])
            φ = pyconvert(Float64, op.params[1])
            λ = pyconvert(Float64, op.params[2])
            # U(θ,φ,λ) = Rz(φ)·Ry(θ)·Rz(λ)  (matrix product, rightmost applied first)
            # PP circuit list: first element = first gate applied in time
            # So the time order is: Rz(λ) first, Ry(θ) second, Rz(φ) last
            push!(gates, PauliRotation(:Z, q)); push!(params, λ)
            push!(gates, PauliRotation(:Y, q)); push!(params, θ)
            push!(gates, PauliRotation(:Z, q)); push!(params, φ)
        elseif name == "cx"
            c = pyconvert(Int, qargs[0]._index) + 1
            t = pyconvert(Int, qargs[1]._index) + 1
            push!(gates, CliffordGate(:CNOT, [c, t]))
        elseif name == "x"
            q = pyconvert(Int, qargs[0]._index) + 1
            push!(gates, CliffordGate(:X, [q]))
        elseif name == "h"
            q = pyconvert(Int, qargs[0]._index) + 1
            push!(gates, CliffordGate(:H, [q]))
        else
            error("Unsupported gate: $name")
        end
    end
    return gates, params
end

function pp_z(circuit, params::Vector{Float64}, n::Int)
    z = zeros(Float64, n)
    for q in 1:n
        obs = PauliString(n, :Z, q, 1.0)
        propagated = propagate(circuit, obs, params;
            min_abs_coeff=0.0, max_weight=Inf, max_freq=Inf, max_sins=Inf, heisenberg=true)
        z[q] = real(overlapwithzero(propagated))
    end
    return z
end

function reference_z(bitstrings::Vector{String}, a_ref::Vector{ComplexF64},
                     n::Int; convention::Symbol)
    z = zeros(Float64, n)
    for (k, bs) in enumerate(bitstrings)
        p = abs2(a_ref[k])
        for q in 1:n
            qiskit_q = q - 1
            if convention == :qiskit
                char_idx = n - qiskit_q
            else
                char_idx = qiskit_q + 1
            end
            bit = (bs[char_idx] == '1') ? 1 : 0
            z[q] += p * (1 - 2 * bit)
        end
    end
    return z
end

function save_csv(path, n, cols::Dict{String, Vector{Float64}})
    names = sort(collect(keys(cols)))
    open(path, "w") do io
        println(io, join(["qubit_qiskit"; "qubit_pp"; names], ","))
        for q in 1:n
            vals = join([@sprintf("%.16e", cols[c][q]) for c in names], ",")
            println(io, "$(q-1),$q,$vals")
        end
    end
end

function classify_qubits(qc::Py)
    cx_qubits = Set{Int}()
    for instr in qc.data
        if pyconvert(String, instr.operation.name) == "cx"
            for q_obj in instr.qubits
                push!(cx_qubits, pyconvert(Int, q_obj._index) + 1)
            end
        end
    end
    n = pyconvert(Int, qc.num_qubits)
    active   = sort(collect(cx_qubits))
    inactive = sort(collect(setdiff(1:n, cx_qubits)))
    return active, inactive
end

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: PP vs TEBD on the same circuit
# ═══════════════════════════════════════════════════════════════════════════

function phase1(label::String, io_report::IO)
    println("\n", "="^80)
    println("  PHASE 1 — PP vs TEBD on circuit $label")
    println("="^80)
    println(io_report, "\n## PHASE 1: PP vs TEBD — $label\n")

    qc = load_qpy_circuit(joinpath(DATA_DIR, "circuit $(label).qpy"))
    n = pyconvert(Int, qc.num_qubits)

    println("  Running TEBD ...")
    t0 = time()
    z_tebd, _ = tebd_z(qc; max_bond_dim=512, trunc=1e-12)
    println("  TEBD done in $(round(time()-t0; digits=2))s")

    println("  Running PP ...")
    circuit, params = qiskit_to_pp(qc)
    t0 = time()
    z_pp = pp_z(circuit, params, n)
    println("  PP done in $(round(time()-t0; digits=2))s")

    err = abs.(z_tebd .- z_pp)
    mae  = mean(err)
    maxe = maximum(err)
    rmse = sqrt(mean(abs2, err))

    @printf("  PP vs TEBD:  MAE = %.3e   MAX = %.3e   RMSE = %.3e\n", mae, maxe, rmse)
    @printf(io_report, "PP vs TEBD:  MAE = %.3e   MAX = %.3e   RMSE = %.3e\n", mae, maxe, rmse)

    csv = joinpath(REPORT_DIR, "phase1_pp_vs_tebd_$(label).csv")
    save_csv(csv, n, Dict("z_tebd" => z_tebd, "z_pp" => z_pp, "abs_error" => err))
    println("  Saved: $csv")

    active, inactive = classify_qubits(qc)
    mae_a  = length(active) > 0 ? mean(err[active]) : 0.0
    mae_i  = length(inactive) > 0 ? mean(err[inactive]) : 0.0
    @printf("  Active qubits (%d):   MAE = %.3e\n", length(active), mae_a)
    @printf("  Inactive qubits (%d): MAE = %.3e\n", length(inactive), mae_i)
    @printf(io_report, "  Active qubits (%d):   MAE = %.3e\n", length(active), mae_a)
    @printf(io_report, "  Inactive qubits (%d): MAE = %.3e\n", length(inactive), mae_i)

    if maxe < 1e-6
        println("  VERDICT: PP and TEBD AGREE to machine precision.")
        println(io_report, "  VERDICT: PP and TEBD AGREE to machine precision.\n")
    elseif maxe < 1e-2
        println("  VERDICT: PP and TEBD agree well (small numerical differences).")
        println(io_report, "  VERDICT: PP and TEBD agree well.\n")
    else
        println("  VERDICT: PP and TEBD DISAGREE. Bug is in circuit mapping.")
        println(io_report, "  VERDICT: PP and TEBD DISAGREE. Bug is in circuit mapping.\n")
    end

    return z_tebd, z_pp
end

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Tiny gate-mapping tests
# ═══════════════════════════════════════════════════════════════════════════

function phase2(io_report::IO)
    println("\n", "="^80)
    println("  PHASE 2 — Tiny gate-mapping tests")
    println("="^80)
    println(io_report, "\n## PHASE 2: Tiny gate-mapping tests\n")

    qiskit_circuit = pyimport("qiskit").QuantumCircuit
    sv_sim = pyimport("qiskit.quantum_info").Statevector

    all_pass = true
    tol = 1e-10

    # ── Test 1: single u gate ─────────────────────────────────────────
    println("\n  Test 1: u(1.2, 0.5, 0.8) on 1 qubit")
    qc = qiskit_circuit(1)
    qc.u(1.2, 0.5, 0.8, 0)

    sv = pyconvert(Vector{ComplexF64}, sv_sim.from_instruction(qc).data)
    qiskit_z = abs2(sv[1]) - abs2(sv[2])
    qiskit_x = 2 * real(conj(sv[1]) * sv[2])
    qiskit_y = 2 * imag(conj(sv[1]) * sv[2])

    pp_circ, pp_par = qiskit_to_pp(qc)
    pp_z_val = real(overlapwithzero(propagate(pp_circ, PauliString(1, :Z, 1, 1.0), pp_par;
        min_abs_coeff=0.0, heisenberg=true)))
    pp_x_val = real(overlapwithzero(propagate(pp_circ, PauliString(1, :X, 1, 1.0), pp_par;
        min_abs_coeff=0.0, heisenberg=true)))
    pp_y_val = real(overlapwithzero(propagate(pp_circ, PauliString(1, :Y, 1, 1.0), pp_par;
        min_abs_coeff=0.0, heisenberg=true)))

    z_tebd_v, _ = tebd_z(qc)
    tebd_z_val = z_tebd_v[1]

    @printf("    Qiskit <Z>=%.10f  PP <Z>=%.10f  TEBD <Z>=%.10f\n", qiskit_z, pp_z_val, tebd_z_val)
    @printf("    Qiskit <X>=%.10f  PP <X>=%.10f\n", qiskit_x, pp_x_val)
    @printf("    Qiskit <Y>=%.10f  PP <Y>=%.10f\n", qiskit_y, pp_y_val)

    t1_pass = abs(qiskit_z - pp_z_val) < tol && abs(qiskit_z - tebd_z_val) < tol &&
              abs(qiskit_x - pp_x_val) < tol && abs(qiskit_y - pp_y_val) < tol
    println("    PASS: $t1_pass")
    println(io_report, "Test 1 (u gate): $(t1_pass ? "PASS" : "FAIL")")
    @printf(io_report, "  |Z_qiskit - Z_pp| = %.3e, |Z_qiskit - Z_tebd| = %.3e\n",
            abs(qiskit_z - pp_z_val), abs(qiskit_z - tebd_z_val))
    all_pass &= t1_pass

    # ── Test 2: 2-qubit CX, test both control/target roles ───────────
    println("\n  Test 2a: CX(0→1) on |10⟩")
    qc2a = qiskit_circuit(2)
    qc2a.x(0)
    qc2a.cx(0, 1)

    sv2a = pyconvert(Vector{ComplexF64}, sv_sim.from_instruction(qc2a).data)
    qiskit_z0 = abs2(sv2a[1]) + abs2(sv2a[2]) - abs2(sv2a[3]) - abs2(sv2a[4])
    qiskit_z1 = abs2(sv2a[1]) - abs2(sv2a[2]) + abs2(sv2a[3]) - abs2(sv2a[4])

    pp_circ2a, pp_par2a = qiskit_to_pp(qc2a)
    pp_z0_2a = real(overlapwithzero(propagate(pp_circ2a, PauliString(2, :Z, 1, 1.0), pp_par2a;
        min_abs_coeff=0.0, heisenberg=true)))
    pp_z1_2a = real(overlapwithzero(propagate(pp_circ2a, PauliString(2, :Z, 2, 1.0), pp_par2a;
        min_abs_coeff=0.0, heisenberg=true)))

    z_tebd2a, _ = tebd_z(qc2a)

    @printf("    Expected: |11⟩ → Z0=-1, Z1=-1\n")
    @printf("    Qiskit Z0=%.6f  Z1=%.6f\n", qiskit_z0, qiskit_z1)
    @printf("    PP     Z0=%.6f  Z1=%.6f\n", pp_z0_2a, pp_z1_2a)
    @printf("    TEBD   Z0=%.6f  Z1=%.6f\n", z_tebd2a[1], z_tebd2a[2])

    t2a_pass = abs(qiskit_z0 - pp_z0_2a) < tol && abs(qiskit_z1 - pp_z1_2a) < tol &&
               abs(qiskit_z0 - z_tebd2a[1]) < tol && abs(qiskit_z1 - z_tebd2a[2]) < tol
    println("    PASS: $t2a_pass")
    println(io_report, "Test 2a (CX 0→1 on |10⟩): $(t2a_pass ? "PASS" : "FAIL")")
    all_pass &= t2a_pass

    println("\n  Test 2b: CX(1→0) on |01⟩")
    qc2b = qiskit_circuit(2)
    qc2b.x(1)
    qc2b.cx(1, 0)

    sv2b = pyconvert(Vector{ComplexF64}, sv_sim.from_instruction(qc2b).data)
    qiskit_z0_2b = abs2(sv2b[1]) + abs2(sv2b[2]) - abs2(sv2b[3]) - abs2(sv2b[4])
    qiskit_z1_2b = abs2(sv2b[1]) - abs2(sv2b[2]) + abs2(sv2b[3]) - abs2(sv2b[4])

    pp_circ2b, pp_par2b = qiskit_to_pp(qc2b)
    pp_z0_2b = real(overlapwithzero(propagate(pp_circ2b, PauliString(2, :Z, 1, 1.0), pp_par2b;
        min_abs_coeff=0.0, heisenberg=true)))
    pp_z1_2b = real(overlapwithzero(propagate(pp_circ2b, PauliString(2, :Z, 2, 1.0), pp_par2b;
        min_abs_coeff=0.0, heisenberg=true)))

    z_tebd2b, _ = tebd_z(qc2b)

    @printf("    Expected: |11⟩ → Z0=-1, Z1=-1\n")
    @printf("    Qiskit Z0=%.6f  Z1=%.6f\n", qiskit_z0_2b, qiskit_z1_2b)
    @printf("    PP     Z0=%.6f  Z1=%.6f\n", pp_z0_2b, pp_z1_2b)
    @printf("    TEBD   Z0=%.6f  Z1=%.6f\n", z_tebd2b[1], z_tebd2b[2])

    t2b_pass = abs(qiskit_z0_2b - pp_z0_2b) < tol && abs(qiskit_z1_2b - pp_z1_2b) < tol &&
               abs(qiskit_z0_2b - z_tebd2b[1]) < tol && abs(qiskit_z1_2b - z_tebd2b[2]) < tol
    println("    PASS: $t2b_pass")
    println(io_report, "Test 2b (CX 1→0 on |01⟩): $(t2b_pass ? "PASS" : "FAIL")")
    all_pass &= t2b_pass

    # ── Test 2c: CX(0→1) on |00⟩ (should be no-op) ──────────────────
    println("\n  Test 2c: CX(0→1) on |00⟩ — should be identity")
    qc2c = qiskit_circuit(2)
    qc2c.cx(0, 1)

    pp_circ2c, pp_par2c = qiskit_to_pp(qc2c)
    pp_z0_2c = real(overlapwithzero(propagate(pp_circ2c, PauliString(2, :Z, 1, 1.0), pp_par2c;
        min_abs_coeff=0.0, heisenberg=true)))
    pp_z1_2c = real(overlapwithzero(propagate(pp_circ2c, PauliString(2, :Z, 2, 1.0), pp_par2c;
        min_abs_coeff=0.0, heisenberg=true)))

    z_tebd2c, _ = tebd_z(qc2c)

    @printf("    Expected: Z0=+1, Z1=+1\n")
    @printf("    PP   Z0=%.6f  Z1=%.6f\n", pp_z0_2c, pp_z1_2c)
    @printf("    TEBD Z0=%.6f  Z1=%.6f\n", z_tebd2c[1], z_tebd2c[2])

    t2c_pass = abs(1.0 - pp_z0_2c) < tol && abs(1.0 - pp_z1_2c) < tol &&
               abs(1.0 - z_tebd2c[1]) < tol && abs(1.0 - z_tebd2c[2]) < tol
    println("    PASS: $t2c_pass")
    println(io_report, "Test 2c (CX on |00⟩ = identity): $(t2c_pass ? "PASS" : "FAIL")")
    all_pass &= t2c_pass

    # ── Test 3: u + cx combined ───────────────────────────────────────
    println("\n  Test 3: u(π/3, π/4, π/6) on q0 then CX(0→1)")
    qc3 = qiskit_circuit(2)
    qc3.u(π/3, π/4, π/6, 0)
    qc3.cx(0, 1)

    sv3 = pyconvert(Vector{ComplexF64}, sv_sim.from_instruction(qc3).data)
    qiskit_z0_3 = abs2(sv3[1]) + abs2(sv3[2]) - abs2(sv3[3]) - abs2(sv3[4])
    qiskit_z1_3 = abs2(sv3[1]) - abs2(sv3[2]) + abs2(sv3[3]) - abs2(sv3[4])

    pp_circ3, pp_par3 = qiskit_to_pp(qc3)
    pp_z0_3 = real(overlapwithzero(propagate(pp_circ3, PauliString(2, :Z, 1, 1.0), pp_par3;
        min_abs_coeff=0.0, heisenberg=true)))
    pp_z1_3 = real(overlapwithzero(propagate(pp_circ3, PauliString(2, :Z, 2, 1.0), pp_par3;
        min_abs_coeff=0.0, heisenberg=true)))

    z_tebd3, _ = tebd_z(qc3)

    @printf("    Qiskit Z0=%.10f  Z1=%.10f\n", qiskit_z0_3, qiskit_z1_3)
    @printf("    PP     Z0=%.10f  Z1=%.10f\n", pp_z0_3, pp_z1_3)
    @printf("    TEBD   Z0=%.10f  Z1=%.10f\n", z_tebd3[1], z_tebd3[2])

    t3_pass = abs(qiskit_z0_3 - pp_z0_3) < tol && abs(qiskit_z1_3 - pp_z1_3) < tol &&
              abs(qiskit_z0_3 - z_tebd3[1]) < tol && abs(qiskit_z1_3 - z_tebd3[2]) < tol
    println("    PASS: $t3_pass")
    println(io_report, "Test 3 (u + CX): $(t3_pass ? "PASS" : "FAIL")")
    @printf(io_report, "  |Z0_qiskit - Z0_pp| = %.3e, |Z1_qiskit - Z1_pp| = %.3e\n",
            abs(qiskit_z0_3 - pp_z0_3), abs(qiskit_z1_3 - pp_z1_3))
    all_pass &= t3_pass

    # ── Test 4: Bell state ────────────────────────────────────────────
    println("\n  Test 4: H on q0, CX(0→1) — Bell state")
    qc4 = qiskit_circuit(2)
    qc4.h(0)
    qc4.cx(0, 1)

    sv4 = pyconvert(Vector{ComplexF64}, sv_sim.from_instruction(qc4).data)
    qiskit_z0_4 = abs2(sv4[1]) + abs2(sv4[2]) - abs2(sv4[3]) - abs2(sv4[4])
    qiskit_z1_4 = abs2(sv4[1]) - abs2(sv4[2]) + abs2(sv4[3]) - abs2(sv4[4])

    pp_circ4, pp_par4 = qiskit_to_pp(qc4)
    pp_z0_4 = real(overlapwithzero(propagate(pp_circ4, PauliString(2, :Z, 1, 1.0), pp_par4;
        min_abs_coeff=0.0, heisenberg=true)))
    pp_z1_4 = real(overlapwithzero(propagate(pp_circ4, PauliString(2, :Z, 2, 1.0), pp_par4;
        min_abs_coeff=0.0, heisenberg=true)))

    @printf("    Expected: Z0=0, Z1=0 (maximally entangled)\n")
    @printf("    Qiskit Z0=%.10f  Z1=%.10f\n", qiskit_z0_4, qiskit_z1_4)
    @printf("    PP     Z0=%.10f  Z1=%.10f\n", pp_z0_4, pp_z1_4)

    t4_pass = abs(qiskit_z0_4 - pp_z0_4) < tol && abs(qiskit_z1_4 - pp_z1_4) < tol
    println("    PASS: $t4_pass")
    println(io_report, "Test 4 (Bell state): $(t4_pass ? "PASS" : "FAIL")")
    all_pass &= t4_pass

    println(io_report, "\nPhase 2 overall: $(all_pass ? "ALL PASS" : "FAILURES DETECTED")\n")
    return all_pass
end

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Stress-test bitstring conventions
# ═══════════════════════════════════════════════════════════════════════════

function phase3(label, z_tebd, z_pp, io_report)
    println("\n", "="^80)
    println("  PHASE 3 — Bitstring convention test — $label")
    println("="^80)
    println(io_report, "\n## PHASE 3: Bitstring convention — $label\n")

    bitstrings, a_ref = load_npz_reference(joinpath(DATA_DIR, "circuit $(label).npz"))
    n = length(z_tebd)

    for conv in [:qiskit, :native]
        z_ref = reference_z(bitstrings, a_ref, n; convention=conv)

        err_vs_tebd = abs.(z_ref .- z_tebd)
        err_vs_pp   = abs.(z_ref .- z_pp)

        mae_tebd = mean(err_vs_tebd)
        mae_pp   = mean(err_vs_pp)
        max_tebd = maximum(err_vs_tebd)
        max_pp   = maximum(err_vs_pp)

        @printf("  Convention %-8s: ref vs TEBD  MAE=%.3e  MAX=%.3e\n", conv, mae_tebd, max_tebd)
        @printf("                      ref vs PP    MAE=%.3e  MAX=%.3e\n", mae_pp, max_pp)
        @printf(io_report, "  Convention %-8s: ref vs TEBD  MAE=%.3e  MAX=%.3e\n", conv, mae_tebd, max_tebd)
        @printf(io_report, "                      ref vs PP    MAE=%.3e  MAX=%.3e\n", mae_pp, max_pp)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: Active/inactive split
# ═══════════════════════════════════════════════════════════════════════════

function phase4(label, z_tebd, z_pp, io_report)
    println("\n", "="^80)
    println("  PHASE 4 — Active/inactive split — $label")
    println("="^80)
    println(io_report, "\n## PHASE 4: Active/inactive split — $label\n")

    qc = load_qpy_circuit(joinpath(DATA_DIR, "circuit $(label).qpy"))
    bitstrings, a_ref = load_npz_reference(joinpath(DATA_DIR, "circuit $(label).npz"))
    n = pyconvert(Int, qc.num_qubits)
    active, inactive = classify_qubits(qc)

    z_ref_q = reference_z(bitstrings, a_ref, n; convention=:qiskit)
    z_ref_n = reference_z(bitstrings, a_ref, n; convention=:native)

    for (conv_name, z_ref) in [("qiskit", z_ref_q), ("native", z_ref_n)]
        println("\n  Convention: $conv_name")
        println(io_report, "\n  Convention: $conv_name")

        for (grp_name, idxs) in [("active", active), ("inactive", inactive)]
            if isempty(idxs) continue end
            e_pp_tebd = abs.(z_pp[idxs] .- z_tebd[idxs])
            e_ref_tebd = abs.(z_ref[idxs] .- z_tebd[idxs])
            e_ref_pp = abs.(z_ref[idxs] .- z_pp[idxs])

            @printf("    %8s (%d qubits):\n", grp_name, length(idxs))
            @printf("      PP vs TEBD:   MAE=%.3e  MAX=%.3e\n", mean(e_pp_tebd), maximum(e_pp_tebd))
            @printf("      ref vs TEBD:  MAE=%.3e  MAX=%.3e\n", mean(e_ref_tebd), maximum(e_ref_tebd))
            @printf("      ref vs PP:    MAE=%.3e  MAX=%.3e\n", mean(e_ref_pp), maximum(e_ref_pp))

            @printf(io_report, "    %8s (%d qubits):\n", grp_name, length(idxs))
            @printf(io_report, "      PP vs TEBD:   MAE=%.3e  MAX=%.3e\n", mean(e_pp_tebd), maximum(e_pp_tebd))
            @printf(io_report, "      ref vs TEBD:  MAE=%.3e  MAX=%.3e\n", mean(e_ref_tebd), maximum(e_ref_tebd))
            @printf(io_report, "      ref vs PP:    MAE=%.3e  MAX=%.3e\n", mean(e_ref_pp), maximum(e_ref_pp))
        end
    end

    csv = joinpath(REPORT_DIR, "phase4_all_z_$(label).csv")
    save_csv(csv, n, Dict(
        "z_tebd" => z_tebd, "z_pp" => z_pp,
        "z_ref_qiskit" => z_ref_q, "z_ref_native" => z_ref_n,
        "err_pp_tebd" => abs.(z_pp .- z_tebd),
        "err_ref_qiskit_tebd" => abs.(z_ref_q .- z_tebd),
        "err_ref_native_tebd" => abs.(z_ref_n .- z_tebd),
    ))
    println("  Saved: $csv")
end

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: Fidelity interpretation
# ═══════════════════════════════════════════════════════════════════════════

function phase5(label, z_tebd, io_report)
    println("\n", "="^80)
    println("  PHASE 5 — Fidelity interpretation — $label")
    println("="^80)
    println(io_report, "\n## PHASE 5: Fidelity interpretation — $label\n")

    qc = load_qpy_circuit(joinpath(DATA_DIR, "circuit $(label).qpy"))
    bitstrings, a_ref = load_npz_reference(joinpath(DATA_DIR, "circuit $(label).npz"))
    n = pyconvert(Int, qc.num_qubits)

    expected_fl = Dict("4q" => 0.14101255424513792, "6q" => 0.13516477073819633)
    fl = get(expected_fl, label, NaN)
    expected_fidelity = 1.0 - fl

    _, tebd_state = tebd_z(qc; max_bond_dim=512, trunc=1e-12)
    a_circ = amplitudes_for_bitstrings(tebd_state, bitstrings; reverse_bits=true)
    overlap = sum(conj(a_ref[k]) * a_circ[k] for k in 1:length(a_ref))
    fidelity = abs2(overlap)

    @printf("  Expected fidelity loss: %.6f  → expected F = %.6f\n", fl, expected_fidelity)
    @printf("  Computed fidelity (TEBD vs ref): F = %.10f\n", fidelity)
    @printf(io_report, "  Expected fidelity loss: %.6f  → expected F = %.6f\n", fl, expected_fidelity)
    @printf(io_report, "  Computed fidelity (TEBD vs ref): F = %.10f\n", fidelity)

    z_ref_q = reference_z(bitstrings, a_ref, n; convention=:qiskit)
    active, _ = classify_qubits(qc)
    err_active = abs.(z_ref_q[active] .- z_tebd[active])

    println(io_report, "\n  For fidelity ≈ $(round(fidelity; digits=4)):")
    println(io_report, "    The maximum possible |ΔZ_i| = 2(1-√F) ≈ $(round(2*(1-sqrt(fidelity)); digits=4))")
    println(io_report, "    Observed max |ΔZ_i| on active qubits = $(round(maximum(err_active); digits=4))")
    if maximum(err_active) > 2 * (1 - sqrt(fidelity)) + 0.01
        println(io_report, "    WARNING: Observed errors EXCEED the fidelity bound.")
        println(io_report, "    This suggests the target state and circuit output are NOT related by the reported fidelity.")
    else
        println(io_report, "    Errors are consistent with the fidelity bound.")
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

function main()
    report_path = joinpath(REPORT_DIR, "debug_summary.txt")
    open(report_path, "w") do io
        println(io, "Debug Summary — PP vs TEBD vs Target")
        println(io, "Generated: $(Dates.now())\n")

        results = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()

        for label in ["4q", "6q"]
            z_tebd, z_pp = phase1(label, io)
            results[label] = (z_tebd, z_pp)
        end

        phase2(io)

        for label in ["4q", "6q"]
            z_tebd, z_pp = results[label]
            phase3(label, z_tebd, z_pp, io)
            phase4(label, z_tebd, z_pp, io)
            phase5(label, z_tebd, io)
        end

        println(io, "\n" * "="^80)
        println(io, "END OF REPORT")
    end
    println("\nFull report: $report_path")
    println("Done.")
end

main()
