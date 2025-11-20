using Pkg
Pkg.activate(@__DIR__)

# Check if dependencies are installed, if not, install them
try
    using TensorOperations
    using StaticArrays
catch
    println("Installing dependencies...")
    Pkg.add("TensorOperations")
    Pkg.add("StaticArrays")
end

println("Running MPS Tests...")
include("test/test_MPS.jl")

println("\nRunning MPO Tests...")
include("test/test_MPO.jl")

