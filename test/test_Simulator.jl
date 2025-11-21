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
using .Yaqs.Simulator
using .Yaqs.GateLibrary
using LinearAlgebra

@testset "Simulator Tests" begin

    @testset "available_cpus" begin
        if haskey(ENV, "SLURM_CPUS_ON_NODE")
            pop!(ENV, "SLURM_CPUS_ON_NODE")
        end
        @test available_cpus() == Sys.CPU_THREADS
        
        ENV["SLURM_CPUS_ON_NODE"] = "8"
        @test available_cpus() == 8
        pop!(ENV, "SLURM_CPUS_ON_NODE")
    end

    @testset "analog_simulation" begin
        L = 5
        initial_state = MPS(L, state="zeros")
        
        H = MPO(L, identity=true, physical_dimensions=fill(2, L))
        
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:L]
        
        sim_params = TimeEvolutionConfig(observables, 1.0;
                                         dt=0.1,
                                         num_traj=10,
                                         max_bond_dim=4,
                                         truncation_threshold=1e-6,
                                         order=2,
                                         sample_timesteps=false)
                                         
        gamma = 0.1
        processes = [Dict("name" => "lowering", "sites" => [i], "strength" => gamma) for i in 1:L]
        append!(processes, [Dict("name" => "pauli_z", "sites" => [i], "strength" => gamma) for i in 1:L])
        
        noise_model = NoiseModel(processes, L)
        
        Simulator.run(initial_state, H, sim_params, noise_model)
        
        for (i, obs) in enumerate(sim_params.observables)
            @test !isempty(obs.results)
            @test !isempty(obs.trajectories)
            @test size(obs.trajectories, 1) == sim_params.num_traj
            @test Base.length(obs.results) == 1
        end
    end
    
    @testset "analog_simulation_parallel_off" begin
        L = 5
        initial_state = MPS(L, state="zeros")
        H = MPO(L, identity=true, physical_dimensions=fill(2, L))
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:L]
        
        sim_params = TimeEvolutionConfig(observables, 1.0;
                                         dt=0.1,
                                         num_traj=10,
                                         max_bond_dim=4,
                                         truncation_threshold=1e-6,
                                         order=2,
                                         sample_timesteps=false)
        gamma = 0.1
        processes = [Dict("name" => "lowering", "sites" => [i], "strength" => gamma) for i in 1:L]
        noise_model = NoiseModel(processes, L)
        
        Simulator.run(initial_state, H, sim_params, noise_model, parallel=false)
        
        for obs in sim_params.observables
            @test !isempty(obs.results)
            @test size(obs.trajectories, 1) == sim_params.num_traj
        end
    end
    
    @testset "analog_simulation_error_mismatch" begin
        state = MPS(2)
        obs = Vector{Observable{AbstractOperator}}() # Explicit typing
        sim_params = TimeEvolutionConfig(obs, 1.0)
        @test_throws ErrorException Simulator.run(state, "InvalidOperator", sim_params)
    end

end
