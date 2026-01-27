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

    @testset "TDVP truncation timing (during vs after_window)" begin
        # Long-range 2-qubit gate triggers the TDVP window path.
        L = 3
        circ = DigitalCircuit(L)
        add_gate!(circ, HGate(), [1])
        add_gate!(circ, RzzGate(π/3), [1, 3])  # long-range

        # Disable threshold-based truncation so both modes should match up to gauge/global phase.
        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0, max_bond_dim=64, truncation_threshold=0.0)
        psi0 = MPS(L; state="zeros")
        pad_bond_dimension!(psi0, 2; noise_scale=0.0)

        opts_during = TJMOptions(local_method=:TDVP, long_range_method=:TDVP, tdvp_truncation_timing=:during)
        opts_after  = TJMOptions(local_method=:TDVP, long_range_method=:TDVP, tdvp_truncation_timing=:after_window)

        psi_during, _ = run_digital_tjm(psi0, circ, nothing, sim_params; alg_options=opts_during)
        psi_after,  _ = run_digital_tjm(psi0, circ, nothing, sim_params; alg_options=opts_after)

        @test check_if_valid_mps(psi_during)
        @test check_if_valid_mps(psi_after)

        # Fidelity should be ~1 (allow global phase).
        ov = abs(MPSModule.scalar_product(psi_during, psi_after))
        @test isapprox(ov, 1.0; atol=1e-10)
    end

    @testset "SRC method for long-range 2-qubit gate matches TEBD (up to tolerance)" begin
        using Random

        L = 3
        circ = DigitalCircuit(L)
        add_gate!(circ, HGate(), [1])
        add_gate!(circ, CZGate(), [1, 3])  # long-range

        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0, max_bond_dim=128, truncation_threshold=1e-12)
        psi0 = MPS(L; state="zeros")

        rng = MersenneTwister(2026)

        opts_tebd = TJMOptions(local_method=:TEBD, long_range_method=:TEBD)
        opts_src  = TJMOptions(local_method=:SRC,  long_range_method=:SRC)

        psi_tebd, _ = run_digital_tjm(psi0, circ, nothing, sim_params; alg_options=opts_tebd, rng=rng)
        psi_src,  _ = run_digital_tjm(psi0, circ, nothing, sim_params; alg_options=opts_src,  rng=rng)

        @test check_if_valid_mps(psi_tebd)
        @test check_if_valid_mps(psi_src)

        ov = abs(MPSModule.scalar_product(psi_tebd, psi_src))
        @test ov ≥ 1 - 1e-6
    end

    @testset "ZIPUP method for long-range 2-qubit gate matches TEBD (up to tolerance)" begin
        L = 3
        circ = DigitalCircuit(L)
        add_gate!(circ, HGate(), [1])
        add_gate!(circ, CZGate(), [1, 3])  # long-range

        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0, max_bond_dim=128, truncation_threshold=1e-12)
        psi0 = MPS(L; state="zeros")

        opts_tebd  = TJMOptions(local_method=:TEBD,  long_range_method=:TEBD)
        opts_zipup = TJMOptions(local_method=:ZIPUP, long_range_method=:ZIPUP)

        psi_tebd,  _ = run_digital_tjm(psi0, circ, nothing, sim_params; alg_options=opts_tebd)
        psi_zipup, _ = run_digital_tjm(psi0, circ, nothing, sim_params; alg_options=opts_zipup)

        @test check_if_valid_mps(psi_tebd)
        @test check_if_valid_mps(psi_zipup)

        ov = abs(MPSModule.scalar_product(psi_tebd, psi_zipup))
        @test ov ≥ 1 - 1e-8
    end

end
