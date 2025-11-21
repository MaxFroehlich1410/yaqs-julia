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
        # Test without SLURM
        if haskey(ENV, "SLURM_CPUS_ON_NODE")
            pop!(ENV, "SLURM_CPUS_ON_NODE")
        end
        @test available_cpus() == Sys.CPU_THREADS
        
        # Test with SLURM
        ENV["SLURM_CPUS_ON_NODE"] = "8"
        @test available_cpus() == 8
        pop!(ENV, "SLURM_CPUS_ON_NODE")
    end

    @testset "analog_simulation" begin
        # Test branch for AnalogSimParams
        length = 5
        initial_state = MPS(length, state="zeros")
        
        # Create Identity MPO (since we don't have full Ising init logic yet, keeping it simple)
        # Or construct basic Ising manually for meaningful results?
        # Let's use Identity for mechanics check.
        H = MPO(length, identity=true, physical_dimensions=fill(2, length))
        
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:length]
        
        sim_params = TimeEvolutionConfig(observables, 1.0;
                                         dt=0.1,
                                         num_traj=10,
                                         max_bond_dim=4,
                                         truncation_threshold=1e-6,
                                         order=2,
                                         sample_timesteps=false)
                                         
        gamma = 0.1
        processes = [Dict("name" => "lowering", "sites" => [i], "strength" => gamma) for i in 1:length]
        append!(processes, [Dict("name" => "pauli_z", "sites" => [i], "strength" => gamma) for i in 1:length])
        
        noise_model = NoiseModel(processes, length)
        
        Simulator.run(initial_state, H, sim_params, noise_model)
        
        for (i, obs) in enumerate(sim_params.observables)
            @test !isempty(obs.results)
            @test !isempty(obs.trajectories)
            @test size(obs.trajectories, 1) == sim_params.num_traj
            @test length(obs.results) == 1
        end
    end
    
    @testset "analog_simulation_parallel_off" begin
        length = 5
        initial_state = MPS(length, state="zeros")
        H = MPO(length, identity=true, physical_dimensions=fill(2, length))
        observables = [Observable("Z_$i", ZGate(), i) for i in 1:length]
        
        sim_params = TimeEvolutionConfig(observables, 1.0;
                                         dt=0.1,
                                         num_traj=10,
                                         max_bond_dim=4,
                                         truncation_threshold=1e-6,
                                         order=2,
                                         sample_timesteps=false)
        gamma = 0.1
        processes = [Dict("name" => "lowering", "sites" => [i], "strength" => gamma) for i in 1:length]
        noise_model = NoiseModel(processes, length)
        
        Simulator.run(initial_state, H, sim_params, noise_model, parallel=false)
        
        for obs in sim_params.observables
            @test !isempty(obs.results)
            @test size(obs.trajectories, 1) == sim_params.num_traj
        end
    end
    
    @testset "analog_simulation_error_mismatch" begin
        # Test passing QuantumCircuit (mock) or invalid operator
        state = MPS(2)
        sim_params = TimeEvolutionConfig([], 1.0)
        # Pass random object as operator
        @test_throws ErrorException Simulator.run(state, "InvalidOperator", sim_params)
    end

end

