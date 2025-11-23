using Test
using LinearAlgebra
using PythonCall
using ..CircuitIngestion
using ..DigitalTJM
using ..GateLibrary
using ..MPSModule
using ..SimulationConfigs
using ..MPOModule

@testset "Circuit Ingestion" begin

    # Check if Qiskit is available
    qiskit_available = false
    try
        pyimport("qiskit")
        qiskit_available = true
    catch e
        println("Qiskit not found. Skipping Circuit Ingestion tests.")
    end

    if qiskit_available
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
            
            psi_out, _ = run_digital_tjm(psi, circ2, nothing, sim_params)
            
            # Expect |Phi+> = (|00> + |11>) / sqrt(2)
            # <Z1> = 0, <Z2> = 0. <Z1Z2> = 1.
            
            z1 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 1))
            z2 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 2))
            zz = real(MPSModule.local_expect_two_site(psi_out, kron(matrix(ZGate()), matrix(ZGate())), 1, 2))
            
            @test isapprox(z1, 0.0; atol=1e-10)
            @test isapprox(z2, 0.0; atol=1e-10)
            @test isapprox(zz, 1.0; atol=1e-10)
        end
    end

end
