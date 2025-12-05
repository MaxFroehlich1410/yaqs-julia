using Test
using Yaqs

@testset "Yaqs Tests" begin
    # List of test files to run
    tests = [
        "test_GateLibrary.jl",
        "test_Decompositions.jl",
        "test_MPS.jl",
        "test_MPO.jl",
        "test_SimulationConfigs.jl",
        "test_Algorithms.jl",
        "test_Noise.jl",
        "test_StochasticProcess.jl",
        "test_Dissipation.jl",
        "test_AnalogTJM.jl",
        "test_DigitalTJM.jl",
        "test_DigitalTJM_Noise.jl",
        "test_Simulator.jl",
        "test_CircuitIngestion.jl",
        "test_CircuitLibrary.jl"
    ]

    for t in tests
        @testset "$t" begin
            include(t)
        end
    end
end

