using Test
using LinearAlgebra
using Yaqs
using Yaqs.DigitalTJM
using Yaqs.MPSModule
using Yaqs.MPOModule
using Yaqs.GateLibrary
using Yaqs.SimulationConfigs
using Yaqs.NoiseModule

@testset "DigitalTJM Tests" begin

    @testset "Circuit Processing" begin
        # L=4. Gates: X(1), X(2), CNOT(1,2) -> Rzz(1,2)
        # Rzz is 2-qubit.
        
        circ = DigitalCircuit(4)
        add_gate!(circ, XGate(), [1])
        add_gate!(circ, XGate(), [2])
        add_gate!(circ, RzzGate(π/2), [1, 2])
        add_gate!(circ, XGate(), [3])
        
        result = DigitalTJM.process_circuit(circ)
        if result isa Tuple
            layers = result[1]
        else
            layers = result
        end
        
        # The greedy algorithm processes gates sequentially:
        # X(1) -> layer 1
        # X(2) -> layer 1 (no conflict)
        # Rzz(1,2) -> conflicts with layer 1, goes to layer 2
        # X(3) -> no conflict with layer 2, goes to layer 2
        # Layer 1: X(1), X(2)
        # Layer 2: Rzz(1, 2), X(3)
        
        @test length(layers) >= 2
        l1 = layers[1]
        @test length(l1) == 2
        @test l1[1].op isa XGate
        @test l1[2].op isa XGate
        
        l2 = layers[2]
        @test length(l2) == 2
        @test l2[1].op isa RzzGate
        @test l2[2].op isa XGate
    end

    @testset "Run Bell State (No Noise)" begin
        L = 2
        circ = DigitalCircuit(L)
        
        # 1. H on site 1
        add_gate!(circ, HGate(), [1])
        
        # 2. Rzz on 1,2
        add_gate!(circ, RzzGate(π/2), [1, 2])
        
        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0) # dt=1 for gates
        
        psi = MPS(L; state="zeros")
        
        psi_out, _ = run_digital_tjm(psi, circ, nothing, sim_params)
        
        # Check result
        z1 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 1))
        z2 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 2))
        
        @test isapprox(z1, 0.0; atol=1e-10)
        @test isapprox(z2, 1.0; atol=1e-10)
    end
    
    @testset "Entanglement Generation" begin
        L = 2
        circ = DigitalCircuit(L)
        add_gate!(circ, HGate(), [1])
        add_gate!(circ, HGate(), [2])
        add_gate!(circ, RzzGate(π/2), [1, 2])
        
        psi = MPS(L; state="zeros")
        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
        psi_out, _ = run_digital_tjm(psi, circ, nothing, sim_params)
        
        # Check Bell correlations
        zz = real(MPSModule.local_expect_two_site(psi_out, kron(matrix(ZGate()), matrix(ZGate())), 1, 2))
        @test isapprox(zz, 0.0; atol=1e-10)
        
        @test check_if_valid_mps(psi_out)
    end
    
    @testset "With Noise" begin
        L = 2
        circ = DigitalCircuit(L)
        add_gate!(circ, XGate(), [1])
        
        # Noise: Phase Damping (Z)
        gamma = 0.2
        proc = Dict("name" => "pauli_z", "sites" => [1], "strength" => gamma)
        noise = NoiseModel([proc], L)
        
        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
        psi = MPS(L; state="zeros")
        
        # Run
        psi_out, _ = run_digital_tjm(psi, circ, noise, sim_params)
        
        # With noise, it might jump. 
        # X gate -> |10>.
        # Z noise -> Phase flip. |10> -> -|10>. Density matrix unaffected.
        # Measurement of Z should still be -1 (state |1>).
        
        z1 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 1))
        @test isapprox(z1, -1.0; atol=1e-10) # Z on |1> is -1
    end

    @testset "RepeatedDigitalCircuit" begin
        L = 2
        step = DigitalCircuit(L)
        add_gate!(step, XGate(), [1])
        rep = RepeatedDigitalCircuit(step, 2) # X twice -> identity on qubit 1

        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
        psi = MPS(L; state="zeros")
        psi_out, _ = run_digital_tjm(psi, rep, nothing, sim_params)

        # Back to |00>
        z1 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 1))
        z2 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 2))
        @test isapprox(z1, 1.0; atol=1e-10)
        @test isapprox(z2, 1.0; atol=1e-10)
        @test check_if_valid_mps(psi_out)
    end

end
