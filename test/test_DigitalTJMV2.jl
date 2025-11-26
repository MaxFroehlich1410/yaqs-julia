using Test
using LinearAlgebra
using Yaqs
using Yaqs.DigitalTJM: DigitalCircuit, add_gate!, DigitalGate, run_digital_tjm, process_circuit
using Yaqs.DigitalTJMV2: run_digital_tjm_v2
using Yaqs.MPSModule
using Yaqs.GateLibrary
using Yaqs.SimulationConfigs

@testset "DigitalTJMV2 Tests" begin

    @testset "Comparison V1 vs V2 (Noise Free)" begin
        L = 4
        circ = DigitalCircuit(L)
        
        # Create an entangled state GHZ-like
        # H on 1
        add_gate!(circ, HGate(), [1])
        # CNOT 1->2 (Modeled as Rxx or similar entangling, or just CNOT if Generator defined)
        # Note: DigitalTJM usually expects gates with generators for 2-qubit gates.
        # RzzGate is a good candidate.
        # CNOT is not directly supported unless we define a generator for it (it's not Hamiltonian evolution naturally).
        # But Rzz(pi/2) is exp(-i * pi/4 * Z \otimes Z). 
        # Let's use Rzz for testing 2-qubit gates.
        
        add_gate!(circ, RzzGate(π/2), [1, 2])
        add_gate!(circ, RzzGate(π/2), [2, 3])
        add_gate!(circ, RzzGate(π/2), [3, 4])
        
        # Single qubit gates mixed in
        add_gate!(circ, XGate(), [2])
        add_gate!(circ, ZGate(), [4])
        
        psi_init = MPS(L; state="zeros")
        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
        
        # Run V1
        psi_v1, _ = run_digital_tjm(psi_init, circ, nothing, sim_params)
        
        # Run V2
        psi_v2, _ = run_digital_tjm_v2(psi_init, circ, nothing, sim_params)
        
        # Compare Overlap |<psi_v1 | psi_v2>|^2 should be 1
        ov = MPSModule.scalar_product(psi_v1, psi_v2)
        overlap = abs2(ov)
        
        @test isapprox(overlap, 1.0; atol=1e-8)
        
        # Compare Local Expectations
        for i in 1:L
            z1 = MPSModule.local_expect(psi_v1, matrix(ZGate()), i)
            z2 = MPSModule.local_expect(psi_v2, matrix(ZGate()), i)
            @test isapprox(z1, z2; atol=1e-8)
        end
    end
    
    @testset "Windowing Correctness" begin
        # Test a gate on sites 2,3 in a 10-qubit chain.
        # V2 should only touch 2,3 (and maybe 1,4 due to padding/orthogonality).
        # We can check if sites far away are untouched?
        # Actually TDVP preserves state on non-acting sites if H is identity.
        # But V2 is explicitly windowed.
        
        L = 10
        circ = DigitalCircuit(L)
        add_gate!(circ, RzzGate(π/2), [2, 3])
        
        psi = MPS(L; state="x+") # |+> state
        
        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
        psi_out, _ = run_digital_tjm_v2(psi, circ, nothing, sim_params)
        
        # Check site 9 is still |+>
        x9 = real(MPSModule.local_expect(psi_out, matrix(XGate()), 9))
        @test isapprox(x9, 1.0; atol=1e-8)
        
        # Check sites 2,3 changed
        # |++> -> exp(-i pi/4 ZZ) |++> 
        # ZZ |++> = |++> (since Z|+> = |->, Z|-> = |+>? No. Z|+> = |->)
        # Wait. Z|+> = |->. Z|-> = |+>.
        # So ZZ |++> = ZZ (|0>+|1>)(|0>+|1>) = (|00> - |01> - |10> + |11>) != |++>
        # So state changes.
        # Expect <XX> should change?
        # Let's just check it changed from 1.0
        
        x2 = real(MPSModule.local_expect(psi_out, matrix(XGate()), 2))
        @test !isapprox(x2, 1.0; atol=1e-8)
    end
    
    @testset "Disjoint Gates in Layer" begin
        # Sites (1,2) and (3,4) in same layer
        L = 4
        circ = DigitalCircuit(L)
        add_gate!(circ, RzzGate(π/2), [1, 2])
        add_gate!(circ, RzzGate(π/2), [3, 4])
        
        psi = MPS(L; state="zeros")
        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
        
        psi_v2, _ = run_digital_tjm_v2(psi, circ, nothing, sim_params)
        
        # Check it ran correctly
        @test MPSModule.check_if_valid_mps(psi_v2)
    end

end

