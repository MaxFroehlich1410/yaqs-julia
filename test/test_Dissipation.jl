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
using .Yaqs.NoiseModule
using .Yaqs.SimulationConfigs
using .Yaqs.DissipationModule
using LinearAlgebra
using Random

rng = MersenneTwister(1234)

@testset "Dissipation Tests" begin

    @testset "apply_dissipation_one_site_canonical_0" begin
        # 1) Create a simple product-state MPS of length 3.
        L = 3
        pdim = 2
        tensors = Vector{Array{ComplexF64, 3}}(undef, L)
        
        for i in 1:L
            vec = rand(rng, ComplexF64, pdim)
            LinearAlgebra.normalize!(vec)
            tensors[i] = reshape(vec, 1, pdim, 1)
        end
        
        state = MPS(L, tensors, fill(pdim, L))
        
        processes = Vector{Dict{String, Any}}()
        for i in 1:L
            push!(processes, Dict("name" => "lowering", "sites" => [i], "strength" => 0.1))
            push!(processes, Dict("name" => "pauli_z", "sites" => [i], "strength" => 0.1))
        end
        
        noise_model = NoiseModel(processes, L)
        dt = 0.1
        dummy_obs = Vector{Observable{AbstractOperator}}()
        sim_params = TimeEvolutionConfig(dummy_obs, 0.0; max_bond_dim=10, truncation_threshold=1e-10)
        
        apply_dissipation(state, noise_model, dt, sim_params)
        
        # Check canonical form
        canonical_indices = check_canonical_form(state)
        @test 1 in canonical_indices # 1-based index 1 == 0-based index 0
    end
    
    @testset "apply_dissipation_two_site_canonical_0" begin
        # 1) Create a simple product-state MPS of length 3.
        L = 3
        pdim = 2
        tensors = Vector{Array{ComplexF64, 3}}(undef, L)
        
        for i in 1:L
            vec = rand(rng, ComplexF64, pdim)
            LinearAlgebra.normalize!(vec)
            tensors[i] = reshape(vec, 1, pdim, 1)
        end
        
        state = MPS(L, tensors, fill(pdim, L))
        
        processes = Vector{Dict{String, Any}}()
        for i in 1:L-1
             push!(processes, Dict("name" => "crosstalk_xx", "sites" => [i, i+1], "strength" => 0.1))
             push!(processes, Dict("name" => "crosstalk_yy", "sites" => [i, i+1], "strength" => 0.1))
        end
        
        noise_model = NoiseModel(processes, L)
        dt = 0.1
        dummy_obs = Vector{Observable{AbstractOperator}}()
        sim_params = TimeEvolutionConfig(dummy_obs, 0.0; max_bond_dim=10, truncation_threshold=1e-10)
        
        apply_dissipation(state, noise_model, dt, sim_params)
        
        # Check canonical form
        canonical_indices = check_canonical_form(state)
        @test 1 in canonical_indices
    end
    
    @testset "is_adjacent_and_is_longrange" begin
        # Direct test of helpers via internal access or via NoiseProcess properties
        # Since we use type dispatch and functions are inside module, we test via usage or if exported.
        # They are not exported, so we skip direct unit test of helper functions
        # unless we access them via DissipationModule.is_adjacent etc.
        
        # Testing via NoiseProcess creation
        proc_adj = LocalNoiseProcess("test", [1, 2], 0.1, zeros(2,2))
        proc_long = LocalNoiseProcess("test", [1, 3], 0.1, zeros(2,2))
        
        @test DissipationModule.is_adjacent(proc_adj) == true
        @test DissipationModule.is_longrange(proc_long) == true
        
        # Testing MPONoiseProcess
        # mpo = MPO(...) # Need dummy MPO
        # proc_mpo_long = MPONoiseProcess("test", [1, 3], 0.1, mpo)
        # @test DissipationModule.is_longrange(proc_mpo_long) == true
    end

end

