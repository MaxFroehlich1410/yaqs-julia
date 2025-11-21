using Test

include("src/Yaqs.jl")
using .Yaqs

@testset "Yaqs.jl Tests" begin
    include("test/test_GateLibrary.jl")
    include("test/test_Decompositions.jl")
    include("test/test_MPS.jl")
    include("test/test_MPO.jl")
    include("test/test_SimulationConfigs.jl")
    include("test/test_Algorithms.jl")
    include("test/test_Noise.jl")
    include("test/test_StochasticProcess.jl")
    include("test/test_Dissipation.jl")
    include("test/test_AnalogTJM.jl")
    include("test/test_Simulator.jl")
end
