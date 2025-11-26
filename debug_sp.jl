using Test
using LinearAlgebra
using ..Yaqs.StochasticProcessModule
using ..Yaqs.MPSModule
using ..Yaqs.MPOModule
using ..Yaqs.NoiseModule
using ..Yaqs.GateLibrary

# Replicate test case
L = 2
gamma = 0.5
proc_list = [Dict("name" => "lowering", "sites" => [1], "strength" => gamma)]
noise = NoiseModel(proc_list, L)

H_sys = MPO(L) 

psi = MPS(L; state="ones")
sp = StochasticProcess(H_sys, noise)

# Inspect H_eff tensors
println("H_eff length: ", sp.H_eff.length)
for i in 1:L
    T = sp.H_eff.tensors[i]
    println("Tensor $i size: ", size(T))
    # Print norms of blocks
    # Tensor (L, Po, Pi, R)
    # i=1: (1, 2, 2, 2) (Left=1, Right=2). [I, term]
    # i=2: (2, 2, 2, 1) (Left=2, Right=1). [0; I]
    
    # Check values
    if i == 1
        # Check T[1, :, :, 2] -> term
        term = T[1, :, :, 2]
        println("Tensor 1 [1, :, :, 2] (Expected term): ")
        display(term)
    elseif i == 2
        # Check T[2, :, :, 1] -> I
        id_blk = T[2, :, :, 1]
        println("Tensor 2 [2, :, :, 1] (Expected I): ")
        display(id_blk)
    end
end

val = expect_mpo(sp.H_eff, psi)
target = -0.5im * gamma

println("Val: $val")
println("Target: $target")
println("IsApprox: ", isapprox(val, target; atol=1e-10))

