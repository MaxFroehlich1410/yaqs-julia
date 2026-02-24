using Test

using Yaqs

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
    include("test/test_CircuitTJM.jl")
    include("test/test_CircuitIngestion.jl")
    include("test/test_CircuitLibrary.jl")
    include("test/test_Ising_Circuit_TJM.jl")
end
