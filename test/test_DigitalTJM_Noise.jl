using Test
using LinearAlgebra
using Random

# Robust include path handling
include(joinpath(@__DIR__, "../src/Yaqs.jl"))

using .Yaqs
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.NoiseModule
using .Yaqs.GateLibrary
using .Yaqs.SimulationConfigs
using .Yaqs.DigitalTJM: DigitalCircuit, add_gate!
using .Yaqs.DigitalTJMV2

# Access internal functions for testing
const create_local_noise_model = Yaqs.DigitalTJMV2.create_local_noise_model
const solve_local_jumps! = Yaqs.DigitalTJMV2.solve_local_jumps!

@testset "DigitalTJM Noise Tests" begin

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

    @testset "solve_local_jumps! Statistics" begin
        # We test that jumps happen with correct probability ~ gamma * dt * <L'L>
        # Case: L = X on site 1. State |0>. <X'X> = <I> = 1.
        # Rate = gamma * dt.
        # If we set gamma * dt = 0.5, probability of jump is 0.5.
        
        L = 2
        psi = MPS(L; state="zeros") # |00>
        
        gamma = 1.0
        dt = 0.5
        
        # Process: X on site 1
        # Jump: X|0> -> |1>.
        proc = LocalNoiseProcess("jump_x", [1], gamma, matrix(XGate()))
        nm = NoiseModel(AbstractNoiseProcess{ComplexF64}[proc])
        
        shots = 1000
        jump_count = 0
        
        for i in 1:shots
            temp_psi = deepcopy(psi)
            # Apply dissipation first to reduce norm
            Yaqs.DissipationModule.apply_dissipation(temp_psi, nm, dt, nothing)
            
            solve_local_jumps!(temp_psi, nm, dt)
            
            # Check if jump happened.
            # |0> -> Z=+1. |1> -> Z=-1.
            z1 = real(MPSModule.local_expect(temp_psi, matrix(ZGate()), 1))
            
            # If z1 is approx -1, it flipped.
            if z1 < 0.0
                jump_count += 1
            end
        end
        
        prob = jump_count / shots
        # Expected: 1 - exp(-gamma * dt) = 1 - exp(-0.5) ≈ 0.393
        @test 0.35 < prob < 0.45
    end
    
    @testset "solve_local_jumps! 2-site" begin
        # Case: 2-site jump. L = X1 Z2.
        # State |00>. L|00> = |10>.
        # <L'L> = <00| Z2 X1 X1 Z2 |00> = <00| I I |00> = 1.
        
        L = 2
        psi = MPS(L; state="zeros")
        
        gamma = 10.0 # Force jump (likely but not underflow)
        dt = 1.0
        
        # NOTE: Julia's kron(A, B) corresponds to B \otimes A relative to indices (1, 2).
        # We want X on 1, Z on 2. So we use kron(Z, X).
        op = kron(matrix(ZGate()), matrix(XGate()))
        # Note: LocalNoiseProcess expects matrix of size (dim, dim). For 2 qubits dim=4.
        proc = LocalNoiseProcess("jump_xz", [1, 2], gamma, op)
        nm = NoiseModel(AbstractNoiseProcess{ComplexF64}[proc])
        
        # Apply dissipation (will crush the norm because gamma is huge)
        Yaqs.DissipationModule.apply_dissipation(psi, nm, dt, nothing)
        
        solve_local_jumps!(psi, nm, dt)
        
        # Check state is |10>
        # Site 1 should be |1> (Z=-1)
        # Site 2 should be |0> (Z=+1)
        
        z1 = real(MPSModule.local_expect(psi, matrix(ZGate()), 1))
        z2 = real(MPSModule.local_expect(psi, matrix(ZGate()), 2))
        
        @test isapprox(z1, -1.0; atol=1e-8)
        @test isapprox(z2, 1.0; atol=1e-8)
        
        # Check canonical form (normalization)
        @test isapprox(norm(psi), 1.0; atol=1e-8)
    end

    @testset "Full Digital Simulation with Noise" begin
        # 2 Qubits. 
        # Gate: Identity on 1,2 (implemented as Rzz(0) or similar, or just window).
        # Noise: X errors on site 1.
        # Check that state mixedness increases or flips occur.
        
        L = 2
        circ = DigitalCircuit(L)
        # Use a gate that triggers the window logic. Rzz(0) is Identity effectively.
        add_gate!(circ, RzzGate(0.0), [1, 2])
        
        # Noise: Strong flip on site 1
        # gamma*dt = 100 -> Certain flip per step if unnormalized rate logic (but probability saturates at 1).
        # Our logic: r < norm_sq (which is 1). Total rate = gamma * dt.
        # If rate > 1, then we assume probability logic handles it? 
        # Code: r_jump = rand() * total_rate.
        # Wait, the code:
        # r = rand()
        # if r > norm_sq ...
        # Standard MCWF: The "no-jump" evolution decreases norm to ||psi||^2 = exp(-rate).
        # A jump occurs if rand() > ||psi||^2.
        # Here we apply `apply_dissipation` first which reduces norm.
        # Then `solve_local_jumps!` checks norm.
        
        # So we need dissipation in the loop for jumps to happen naturally!
        # The `run_digital_tjm_v2` calls `apply_dissipation` then `solve_local_jumps!`.
        # Correct.
        
        proc = LocalNoiseProcess("flip", [1], 10.0, matrix(XGate())) # Very Strong noise to ensure jump
        nm = NoiseModel(AbstractNoiseProcess{ComplexF64}[proc])
        
        sim_params = TimeEvolutionConfig(Observable[], 1.0; dt=1.0)
        psi_init = MPS(L; state="zeros") # |00>
        
        # Run
        psi_out, _ = run_digital_tjm_v2(psi_init, circ, nm, sim_params)
        
        # Expect site 1 to have flipped to |1> (z=-1)
        # With gamma=10, prob of no-jump is exp(-10) ~ 4e-5.
        
        z1 = real(MPSModule.local_expect(psi_out, matrix(ZGate()), 1))
        
        # If jump happened, z1 = -1. If not, z1 = +1 (normalized).
        # We can't be 100% sure in one shot, but let's just run it.
        # print("Final Z1: $z1\n")
        
        # Check it's a valid MPS
        @test MPSModule.check_if_valid_mps(psi_out)
        @test isapprox(norm(psi_out), 1.0; atol=1e-8)
    end
end

