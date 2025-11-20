using Test
using LinearAlgebra

if !isdefined(Main, :Yaqs)
    include("../src/Yaqs.jl")
    using .Yaqs
end

using .Yaqs.GateLibrary
using .Yaqs.Decompositions
using .Yaqs.MPSModule
using .Yaqs.SimulationConfigs

@testset "SimulationConfigs Tests" begin

    @testset "Observable Creation" begin
        gate = XGate()
        site = 1
        obs = Observable("X", gate, site)

        @test obs.name == "X"
        @test obs.op == gate
        @test obs.sites == [1]
        @test isempty(obs.results)
        @test isempty(obs.trajectories)
    end

    @testset "TimeEvolutionConfig Basic" begin
        obs_list = [Observable("X", XGate(), 1)]
        elapsed_time = 1.0
        dt = 0.2
        config = TimeEvolutionConfig(obs_list, elapsed_time; dt=dt, sample_timesteps=true, num_traj=50)

        @test config.observables == obs_list
        @test config.total_time == elapsed_time
        @test config.dt == dt
        
        expected_times = collect(0.0:0.2:1.0)
        @test length(config.times) == length(expected_times)
        @test isapprox(config.times, expected_times; atol=1e-10)
        
        @test config.sample_timesteps == true
        @test config.num_traj == 50
    end

    @testset "TimeEvolutionConfig Defaults" begin
        obs_list = Observable[]
        elapsed_time = 2.0
        config = TimeEvolutionConfig(obs_list, elapsed_time)

        @test config.observables == obs_list
        @test config.total_time == 2.0
        @test config.dt == 0.1
        @test config.sample_timesteps == true
        @test isapprox(config.times[end], 2.0; atol=1e-10)
        @test config.num_traj == 1000
        @test config.max_bond_dim == 4096
        @test config.truncation_threshold == 1e-9
    end

    @testset "Initialize with Sample Timesteps" begin
        obs = Observable("X", XGate(), 1)
        config = TimeEvolutionConfig([obs], 1.0; dt=0.5, sample_timesteps=true, num_traj=10)
        # times: 0.0, 0.5, 1.0 (length 3)

        SimulationConfigs.initialize!(obs, config)
        
        @test length(obs.results) == 3
        @test size(obs.trajectories) == (10, 3)
    end

    @testset "Initialize without Sample Timesteps" begin
        obs = Observable("X", XGate(), 1)
        config = TimeEvolutionConfig([obs], 1.0; dt=0.25, sample_timesteps=false, num_traj=5)
        # times length 5

        SimulationConfigs.initialize!(obs, config)
        
        @test length(obs.results) == 1
        @test size(obs.trajectories) == (5, 1)
    end

    @testset "Expectation Value Integration" begin
        # Setup MPS
        L = 4
        mps = MPS(L; state="x+") # |++++>
        
        # Observable X on site 1
        obs = Observable("X", XGate(), 1)
        
        val = expect(mps, obs)
        @test isapprox(val, 1.0; atol=1e-10)
        
        # Observable Z on site 1 -> should be 0 for |+>
        obs_z = Observable("Z", ZGate(), 1)
        val_z = expect(mps, obs_z)
        @test isapprox(val_z, 0.0; atol=1e-10)
    end

    @testset "Aggregation Logic" begin
        obs = Observable("X", XGate(), 1)
        config = TimeEvolutionConfig([obs], 1.0; num_traj=2, sample_timesteps=false)
        SimulationConfigs.initialize!(obs, config)
        
        # Mock trajectories
        # Shape (2, 1)
        obs.trajectories[1, 1] = 1.0
        obs.trajectories[2, 1] = 0.5
        
        aggregate_trajectories!(config)
        
        # Mean should be 0.75
        @test isapprox(obs.results[1], 0.75; atol=1e-10)
    end
    
    @testset "MeasurementConfig" begin
        shots = 10
        config = MeasurementConfig(shots)
        
        @test length(config.measurements) == shots
        
        # Mock results
        config.measurements[1] = Dict(0=>1)
        config.measurements[2] = Dict(1=>1)
        
        aggregate_measurements!(config)
        
        @test config.results[0] == 1
        @test config.results[1] == 1
    end

end
