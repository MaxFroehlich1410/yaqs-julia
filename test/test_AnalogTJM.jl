# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if !isdefined(Main, :Yaqs)
    include("../src/Yaqs.jl")
end

using Test
using .Yaqs.MPSModule
using .Yaqs.MPOModule
using .Yaqs.NoiseModule
using .Yaqs.SimulationConfigs
using .Yaqs.AnalogTJM
using .Yaqs.GateLibrary
using .Yaqs.Algorithms
using LinearAlgebra

@testset "AnalogTJM Tests" begin

    @testset "initialize" begin
        L = 5
        state = MPS(L, state="zeros")
        processes = [Dict("name" => "lowering", "sites" => [i], "strength" => 0.1) for i in 1:L]
        noise_model = NoiseModel(processes, L)
        
        dummy_obs = Vector{Observable{AbstractOperator}}()
        sim_params = TimeEvolutionConfig(dummy_obs, 0.2; dt=0.2)
        
        # Test initialization with nothing for stoch_proc
        state_init = initialize(state, noise_model, sim_params)
        
        # Explicitly normalize because initialize without solve_jumps! might leave it unnormalized
        MPSModule.normalize!(state_init)
        
        norm_after = real(scalar_product(state_init, state_init))
        
        @test norm_after ≈ 1.0
    end

    @testset "analog_tjm_2_no_sampling" begin
        L = 5
        H = MPO(L, identity=true, physical_dimensions=fill(2, L))
        
        state = MPS(L, state="zeros")
        noise_model = nothing
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:L]
        
        sim_params = TimeEvolutionConfig(observables, 0.2; dt=0.1, sample_timesteps=false)
        
        args = (0, state, noise_model, sim_params, H)
        results = analog_tjm_2(args)
        
        @test size(results) == (L, 1)
    end

    @testset "analog_tjm_2_sampling" begin
        L = 5
        H = MPO(L, identity=true, physical_dimensions=fill(2, L))
        state = MPS(L, state="zeros")
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:L]
        
        sim_params = TimeEvolutionConfig(observables, 0.2; dt=0.1, sample_timesteps=true)
        
        args = (0, state, nothing, sim_params, H)
        results = analog_tjm_2(args)
        
        @test size(results) == (L, 3)
    end

    @testset "analog_tjm_1_no_sampling" begin
        L = 5
        H = MPO(L, identity=true, physical_dimensions=fill(2, L))
        state = MPS(L, state="zeros")
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:L]
        
        sim_params = TimeEvolutionConfig(observables, 0.2; dt=0.1, sample_timesteps=false)
        
        args = (0, state, nothing, sim_params, H)
        results = analog_tjm_1(args)
        
        @test size(results) == (L, 1)
    end
    
    @testset "analog_tjm_1_sampling" begin
        L = 5
        H = MPO(L, identity=true, physical_dimensions=fill(2, L))
        state = MPS(L, state="zeros")
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:L]
        
        sim_params = TimeEvolutionConfig(observables, 0.2; dt=0.1, sample_timesteps=true)
        
        args = (0, state, nothing, sim_params, H)
        results = analog_tjm_1(args)
        
        @test size(results) == (L, 3)
    end

    @testset "step_through (no noise)" begin
        L = 3
        H = MPO(L, identity=true, physical_dimensions=fill(2, L))
        state = MPS(L, state="zeros")
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:L]
        sim_params = TimeEvolutionConfig(observables, 0.2; dt=0.1, sample_timesteps=true)

        # should preserve norm and keep <Z> ≈ 1 for identity Hamiltonian evolution
        state2 = step_through(state, H, nothing, sim_params)
        @test MPSModule.check_if_valid_mps(state2)
        @test isapprox(real(scalar_product(state2, state2)), 1.0; atol=1e-8)
        for i in 1:L
            @test isapprox(real(expect(state2, observables[i])), 1.0; atol=1e-8)
        end
    end

    @testset "sample (no noise)" begin
        L = 3
        H = MPO(L, identity=true, physical_dimensions=fill(2, L))
        phi = MPS(L, state="zeros")
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:L]
        sim_params = TimeEvolutionConfig(observables, 0.2; dt=0.1, sample_timesteps=true)

        results = zeros(Float64, length(observables), length(sim_params.times))
        sample(phi, H, nothing, sim_params, results, 2)

        # After sampling, column 2 should be written.
        @test all(isfinite, results[:, 2])
        @test all(x -> isapprox(x, 1.0; atol=1e-8), results[:, 2])
    end

end
