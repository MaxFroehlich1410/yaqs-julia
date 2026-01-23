# Legacy/compatibility regression tests for CircuitTJM noise-window selection.
#
# This file overlaps with `test_CircuitTJM_Noise.jl` and exists to exercise internal noise-window
# filtering logic used during circuit TJM execution. It validates that only noise processes supported
# on the active two-site window are selected by the helper under test.
#
# Args:
#     None
#
# Returns:
#     Nothing: Defines `@testset`s checking local-noise-model extraction for a moving window.
using Test
using LinearAlgebra
using Random
using Yaqs
using Yaqs.MPSModule
using Yaqs.MPOModule
using Yaqs.NoiseModule
using Yaqs.GateLibrary
using Yaqs.SimulationConfigs
using Yaqs.CircuitTJM

# Access internal functions for testing
const create_local_noise_model = Yaqs.DigitalTJM.create_local_noise_model

@testset "CircuitTJM Noise Tests" begin

    @testset "create_local_noise_model" begin
        # Setup: Noise model with processes on (1), (2), (1,2), (3,4)
        procs = AbstractNoiseProcess{ComplexF64}[
            LocalNoiseProcess("p1", [1], 0.1, matrix(XGate())),
            LocalNoiseProcess("p2", [2], 0.1, matrix(ZGate())),
            LocalNoiseProcess("p12", [1, 2], 0.1, kron(matrix(XGate()), matrix(XGate()))),
            LocalNoiseProcess("p34", [3, 4], 0.1, kron(matrix(ZGate()), matrix(ZGate())))
        ]
        
        nm = NoiseModel(procs)
        
        # Test 1: Window (1, 2)
        local_nm = create_local_noise_model(nm, 1, 2)
        @test length(local_nm.processes) == 3
        names = [p.name for p in local_nm.processes]
        @test "p1" in names
        @test "p2" in names
        @test "p12" in names
        @test !("p34" in names)
        
        # Test 2: Window (3, 4)
        local_nm_34 = create_local_noise_model(nm, 3, 4)
        @test length(local_nm_34.processes) == 1
        @test local_nm_34.processes[1].name == "p34"
        
        # Test 3: Window (2, 3) - Should only pick up single site proc on 2?
        # Current implementation logic: 
        # p_sites subset of affected_sites (Set([s1, s2]))
        # p2 on [2] -> Yes.
        # p1 on [1] -> No.
        # p12 on [1,2] -> No (1 not in {2,3}).
        local_nm_23 = create_local_noise_model(nm, 2, 3)
        @test length(local_nm_23.processes) == 1
        @test local_nm_23.processes[1].name == "p2"
    end

    @testset "Full Digital Simulation with Noise" begin
        
        L = 2
        circ = DigitalCircuit(L)
        # Use a gate that triggers the window logic. Rzz(0) is Identity effectively.
        add_gate!(circ, RzzGate(0.0), [1, 2])
        

        
        proc = LocalNoiseProcess("flip", [1], 10.0, matrix(XGate())) # Very Strong noise to ensure jump
        nm = NoiseModel(AbstractNoiseProcess{ComplexF64}[proc])
        
        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
        psi_init = MPS(L; state="zeros") # |00>
        
        # Run
        psi_out, _ = run_circuit_tjm(psi_init, circ, nm, sim_params)
    
        
        z1 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 1))
        
        # Check it's a valid MPS
        @test MPSModule.check_if_valid_mps(psi_out)
        @test isapprox(norm(psi_out), 1.0; atol=1e-8)
    end
end

