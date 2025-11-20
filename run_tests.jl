using Pkg
Pkg.activate(@__DIR__)

# Check if dependencies are installed, if not, install them
try
    using TensorOperations
    using StaticArrays
    using KrylovKit
catch
    println("Installing dependencies...")
    Pkg.add("TensorOperations")
    Pkg.add("StaticArrays")
    Pkg.add("KrylovKit")
end

println("Loading Yaqs Package...")
include("src/Yaqs.jl")
using .Yaqs

# Test files should now use .Yaqs or assume modules are available if we export them to Main?
# We can't export to Main easily from script.
# But we can modify tests to use `Yaqs`.

println("Running GateLibrary Tests...")
include("test/test_GateLibrary.jl")

println("\nRunning Decompositions Tests...")
include("test/test_Decompositions.jl")

println("\nRunning MPS Tests...")
include("test/test_MPS.jl")

println("\nRunning MPO Tests...")
include("test/test_MPO.jl")

println("\nRunning SimulationConfigs Tests...")
include("test/test_SimulationConfigs.jl")

println("\nRunning Algorithms Tests...")
include("test/test_Algorithms.jl")

println("\nRunning Noise Tests...")
include("test/test_Noise.jl")

println("\nRunning StochasticProcess Tests...")
include("test/test_StochasticProcess.jl")
