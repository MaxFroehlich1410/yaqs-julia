using Test
using LinearAlgebra
using Statistics
using StaticArrays
using Yaqs
using Yaqs.StochasticProcessModule
using Yaqs.MPSModule
using Yaqs.MPOModule
using Yaqs.NoiseModule
using Yaqs.GateLibrary
using Yaqs.SimulationConfigs
using Yaqs.DissipationModule

@testset "Stochastic Process Refactored" begin

    @testset "Stochastic Factor" begin
        L = 2
        psi = MPS(L; state="ones")
        # Norm is 1.0. dp should be 0.
        dp = calculate_stochastic_factor(psi)
        @test isapprox(dp, 0.0; atol=1e-12)
        
        # Decay the state
        psi.tensors[1] .*= 0.5
        # Norm is 0.5. Squared norm is 0.25.
        # dp = 1 - 0.25 = 0.75
        dp = calculate_stochastic_factor(psi)
        @test isapprox(dp, 0.75; atol=1e-12)
    end
    
    @testset "Probability Distribution - 1-site Pauli" begin
        L = 2
        gamma = 0.5
        dt = 0.1
        # Process: gamma * Z on site 1
        proc_list = [Dict("name" => "pauli_z", "sites" => [1], "strength" => gamma)]
        noise = NoiseModel(proc_list, L)
        
        psi = MPS(L; state="x+") 
        
        sim_params = TimeEvolutionConfig(Observable[], 1.0)
        
        probs, candidates = create_probability_distribution(psi, noise, dt, sim_params)
        
        @test length(probs) == 1
        @test isapprox(probs[1], 1.0)
    end
    
    @testset "Probability Distribution - 2-site Pauli" begin
        L = 2
        gamma = 0.2
        dt = 0.1
        # Process: gamma * Z1 Z2
        ZZ = kron(matrix(ZGate()), matrix(ZGate()))
        # Manually create process
        proc = LocalNoiseProcess("custom_zz", [1, 2], gamma, SMatrix{4,4,ComplexF64}(ZZ))
        # Explicit type for vector
        procs = Vector{AbstractNoiseProcess{ComplexF64}}([proc])
        noise = NoiseModel(procs)
        
        psi = MPS(L; state="ones")
        sim_params = TimeEvolutionConfig(Observable[], 1.0)
        
        probs, candidates = create_probability_distribution(psi, noise, dt, sim_params)
        
        @test length(probs) == 1
        @test isapprox(probs[1], 1.0)
        
        # Candidate checks
        (p, op, type, sites) = candidates[1]
        @test type == "local_2"
    end
    
    @testset "Long Range Pauli (Factors)" begin
        L = 3
        gamma = 0.1
        dt = 0.1
        # Z1 Z3
        # Must pass 'factors' for optimized path
        sigma = Matrix(matrix(ZGate())) # Convert to Matrix
        tau = Matrix(matrix(ZGate()))
        factors = [sigma, tau]
        
        mpo = MPO(L)
        proc = MPONoiseProcess("crosstalk_zz", [1, 3], gamma, mpo, factors)
        procs = Vector{AbstractNoiseProcess{ComplexF64}}([proc])
        noise = NoiseModel(procs)
        
        psi = MPS(L; state="ones")
        sim_params = TimeEvolutionConfig(Observable[], 1.0)
        
        probs, candidates = create_probability_distribution(psi, noise, dt, sim_params)
        
        @test length(probs) == 1
        (p, op, type, sites) = candidates[1]
        @test type == "pauli_long_range"
        @test length(op) == 2 # factors
    end
    
    @testset "Stochastic Process Execution" begin
        L = 2
        psi = MPS(L; state="ones")
        # Apply decay manually to trigger jump
        psi.tensors[1] .*= 0.5 
        
        # Empty noise model -> Just normalize
        procs = Vector{AbstractNoiseProcess{ComplexF64}}()
        noise = NoiseModel(procs)
        sim_params = TimeEvolutionConfig(Observable[], 1.0)
        
        psi_new = stochastic_process!(psi, noise, 0.1, sim_params)
        @test isapprox(norm(psi_new), 1.0; atol=1e-12)
        @test isapprox(real(scalar_product(psi_new, psi_new)), 1.0; atol=1e-12)
    end

end
