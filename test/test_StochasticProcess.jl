using Test
using LinearAlgebra
using Statistics
using ..StochasticProcessModule
using ..MPSModule
using ..MPOModule
using ..NoiseModule
using ..GateLibrary
using ..SimulationConfigs

@testset "Stochastic Process" begin

    @testset "Effective Hamiltonian Construction" begin
        L = 2
        gamma = 0.5
        proc_list = [Dict("name" => "lowering", "sites" => [1], "strength" => gamma)]
        noise = NoiseModel(proc_list, L)
        
        H_sys = MPO(L) 
        
        psi = MPS(L; state="ones")
        # Use standard constructor
        sp = StochasticProcess(H_sys, noise)
        
        val = expect_mpo(sp.H_eff, psi)
        target = -0.5im * gamma
        
        # Relax tolerance slightly if needed, but it should be exact
        @test isapprox(val, target; atol=1e-10)
    end
    
    @testset "Trajectory Evolution - Relaxation" begin
        L = 1
        gamma = 1.0
        proc_list = [Dict("name" => "lowering", "sites" => [1], "strength" => gamma)]
        noise = NoiseModel(proc_list, L)
        H_sys = MPO(L) 
        
        psi = MPS(L; state="ones")
        sp = StochasticProcess(H_sys, noise)
        
        obs = Observable("Z", ZGate(), 1)
        
        t_final = 0.5
        dt = 0.1
        
        res = trajectory_evolution(sp, psi, t_final, dt; observables=[obs])
        
        @test haskey(res, "Z")
        @test length(res["Z"]) == 5
        @test haskey(res, "time")
        @test length(res["time"]) == 5
        @test res["Z"][1] isa ComplexF64
    end
    
    @testset "Trajectory Evolution - MPO Noise (Long Range)" begin
        L = 3
        proc_list = [Dict("name" => "crosstalk_xy", "sites" => [1, 3], "strength" => 0.2, "unraveling" => "projector")]
        noise = NoiseModel(proc_list, L)
        H_sys = MPO(L)
        
        psi = MPS(L; state="zeros")
        sp = StochasticProcess(H_sys, noise)
        
        has_mpo_jump = false
        for (proc, op, type) in sp.jump_ops
            if type == "mpo"
                has_mpo_jump = true
            end
        end
        @test has_mpo_jump
        
        res = trajectory_evolution(sp, psi, 0.1, 0.05; observables=[Observable("Z1", ZGate(), 1)])
        @test length(res["time"]) == 2
    end

end
