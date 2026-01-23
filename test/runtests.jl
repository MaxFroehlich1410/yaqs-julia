# Entry-point for the Julia test suite.
#
# This file defines the top-level `@testset` and includes individual test files in a fixed order.
# It is the primary driver invoked by `Pkg.test()` for the `Yaqs` package.
#
# Args:
#     None
#
# Returns:
#     Nothing: Executes all included test files under a single root testset.
using Test
using Yaqs

@testset "Yaqs Tests" begin
    # List of test files to run
    tests = [
        "test_Timing.jl",
        "test_InternalCore.jl",
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
        "test_CircuitTJM.jl",
        "test_CircuitTJM_Noise.jl",
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

