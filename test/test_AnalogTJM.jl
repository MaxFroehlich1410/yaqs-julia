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

end
