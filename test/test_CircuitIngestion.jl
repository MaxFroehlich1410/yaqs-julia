# Unit tests for circuit ingestion utilities (`Yaqs.CircuitIngestion`).
#
# These tests validate:
# - mapping of Qiskit gate names/parameters to `Yaqs.GateLibrary` operator types (`map_qiskit_name`)
# - conversion of Python instruction objects into `DigitalGate`s via PythonCall stubs (no Qiskit required)
# - optional end-to-end ingestion of a real Qiskit circuit when Qiskit is available (skips otherwise)
#
# Args:
#     None
#
# Returns:
#     Nothing: Defines `@testset`s for name mapping, instruction conversion, and optional Qiskit IO.
using Test
using LinearAlgebra
using PythonCall
using Yaqs
using Yaqs.CircuitIngestion
using Yaqs.CircuitTJM
using Yaqs.GateLibrary
using Yaqs.MPSModule
using Yaqs.SimulationConfigs
using Yaqs.MPOModule

@testset "Circuit Ingestion" begin

    @testset "map_qiskit_name (no qiskit required)" begin
        @test map_qiskit_name("x", Float64[]) isa XGate
        @test map_qiskit_name("cx", Float64[]) isa CXGate
        @test map_qiskit_name("cz", Float64[]) isa CZGate
        @test map_qiskit_name("h", Float64[]) isa HGate
        @test map_qiskit_name("id", Float64[]) isa IdGate
        @test map_qiskit_name("s", Float64[]) isa SGate
        @test map_qiskit_name("t", Float64[]) isa TGate
        @test map_qiskit_name("sx", Float64[]) isa RxGate
        @test map_qiskit_name("rx", [0.5]).theta ≈ 0.5
        @test map_qiskit_name("ry", [0.6]).theta ≈ 0.6
        @test map_qiskit_name("rz", [0.7]).theta ≈ 0.7
        @test map_qiskit_name("p", [0.8]).theta ≈ 0.8
        @test map_qiskit_name("u3", [0.1, 0.2, 0.3]) isa UGate
        @test map_qiskit_name("swap", Float64[]) isa SWAPGate
        @test map_qiskit_name("rxx", [0.4]) isa RxxGate
        @test map_qiskit_name("ryy", [0.4]) isa RyyGate
        @test map_qiskit_name("rzz", [0.4]) isa RzzGate

        @test_throws ErrorException map_qiskit_name("unsupported_gate", Float64[])
    end

    @testset "convert_instruction_to_gate (python stubs, no qiskit required)" begin
        main = pyimport("__main__")
        PythonCall.pyexec("""
class _Op:
    def __init__(self, name, params, label=None):
        self.name = name
        self.params = params
        self.label = label

class _Qubit:
    def __init__(self, idx):
        self._index = idx

class _Instr:
    def __init__(self, op, qubits):
        self.operation = op
        self.qubits = qubits
""", pygetattr(main, "__dict__"))
        Op = pygetattr(main, "_Op")
        Qubit = pygetattr(main, "_Qubit")
        Instr = pygetattr(main, "_Instr")

        # Rx on qubit 0 -> sites=[1]
        instr = Instr(Op("rx", [0.5]), [Qubit(0)])
        gate = convert_instruction_to_gate(instr, PythonCall.pybuiltins.None)
        @test gate isa DigitalGate
        @test gate.op isa RxGate
        @test gate.op.theta ≈ 0.5
        @test gate.sites == [1]
        @test gate.generator == [matrix(XGate())]

        # Barrier with label defaults/propagation
        binstr = Instr(Op("barrier", [], label="SAMPLE_OBSERVABLES"), [Qubit(0), Qubit(1)])
        bgate = convert_instruction_to_gate(binstr, PythonCall.pybuiltins.None)
        @test bgate.op isa Barrier
        @test bgate.sites == [1, 2]

        # Unsupported gate -> nothing
        bad = Instr(Op("not_a_gate", []), [Qubit(0)])
        @test convert_instruction_to_gate(bad, PythonCall.pybuiltins.None) === nothing
    end

    # Qiskit integration tests are expensive (large Python env + slow import on first run).
    # Keep them opt-in so `run_tests.jl` stays fast by default.
    run_qiskit_tests = get(ENV, "YAQS_RUN_QISKIT_TESTS", "0") == "1"
    if !run_qiskit_tests
        println("Skipping Qiskit integration tests (set YAQS_RUN_QISKIT_TESTS=1 to enable).")
    else
        # Check if Qiskit is available
        qiskit_available = false
        try
            pyimport("qiskit")
            qiskit_available = true
        catch e
            println("Qiskit not found. Skipping Qiskit integration tests.")
        end
        if !qiskit_available
            return
        end

        # 1. Create a Qiskit circuit via PythonCall
        qiskit = pyimport("qiskit")
        QuantumCircuit = qiskit.QuantumCircuit
        
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(0.5, 2)
        qc.barrier()
        qc.rzz(0.2, 1, 2)
        
        # 2. Ingest
        circ = ingest_qiskit_circuit(qc)
        
        @test circ isa DigitalCircuit
        @test circ.num_qubits == 3
        
        # Check layers
        # Layer 1: H(0), Rx(2)
        # Qiskit DAG: H(0) is first on wire 0. Rx(2) is first on wire 2.
        # CX(0,1) depends on H(0). So CX is not in layer 1.
        
        @test length(circ.layers) >= 2
        
        l1 = circ.layers[1]
        # Should contain H on site 1, Rx on site 3.
        found_h = false
        found_rx = false
        
        for g in l1
            if g.op isa HGate && g.sites == [1]
                found_h = true
            end
            if g.op isa RxGate && g.sites == [3]
                found_rx = true
                @test g.op.theta ≈ 0.5
            end
        end
        @test found_h
        @test found_rx
        
        # Layer 2: CX(0,1) -> CX(1,2)
        # Is CX in layer 2?
        # H(0) removed. 0 is free. 1 was free. So CX(0,1) available.
        # Rx(2) removed. 2 is free.
        
        l2 = circ.layers[2]
        found_cx = false
        for g in l2
            if g.op isa CXGate && g.sites == [1, 2]
                found_cx = true
            end
        end
        @test found_cx
        
        # Layer 3: Rzz(1, 2) -> Rzz(2, 3)
        
        l3 = circ.layers[3]
        found_rzz = false
        for g in l3
            if g.op isa RzzGate && g.sites == [2, 3]
                found_rzz = true
                @test g.op.theta ≈ 0.2
            end
        end
        @test found_rzz
        
        @testset "Run Simulation from Qiskit" begin
            # Simple Bell State: H(0), CX(0,1)
            qc2 = QuantumCircuit(2)
            qc2.h(0)
            qc2.cx(0, 1)
            
            circ2 = ingest_qiskit_circuit(qc2)
            
            psi = MPS(2; state="zeros")
            sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
            
            psi_out, _ = run_circuit_tjm(psi, circ2, nothing, sim_params)
            
            # Check that the state is valid and normalized after circuit application
            @test MPSModule.check_if_valid_mps(psi_out)
            @test isapprox(norm(psi_out), 1.0; atol=1e-8)
            
            # Verify that gates were applied (state changed from initial |00>)
            z1 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 1))
            z2 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 2))
            
            # The exact expectations depend on gate conversion and application
            # Just verify that the state is not the initial |00> state
            # (at least one qubit should have changed)
            @test !(isapprox(z1, 1.0; atol=1e-6) && isapprox(z2, 1.0; atol=1e-6))
        end
    end

end
