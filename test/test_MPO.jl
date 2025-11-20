using Test
using LinearAlgebra
using Random

if !isdefined(Main, :MPSModule)
    include("../src/MPS.jl")
end
if !isdefined(Main, :MPOModule)
    include("../src/MPO.jl")
end

using .MPSModule
using .MPOModule

# Helper (same as in test_MPS.jl)
function crandn(shape)
    return (randn(ComplexF64, shape) .+ 1im .* randn(ComplexF64, shape)) ./ sqrt(2)
end

function random_mps(length, phys_dim, bond_dim)
    # Simplified random MPS generator
    tensors = [crandn((1, phys_dim, bond_dim))]
    for _ in 2:length-1
        push!(tensors, crandn((bond_dim, phys_dim, bond_dim)))
    end
    push!(tensors, crandn((bond_dim, phys_dim, 1)))
    
    mps = MPS(length, tensors, [phys_dim for _ in 1:length])
    normalize!(mps)
    return mps
end

@testset "MPO Tests" begin

    @testset "Initialization" begin
        L = 4
        mpo = MPO(L; identity=true)
        @test mpo.length == L
        @test length(mpo.tensors) == L
        
        # Check Identity Structure (1, d, d, 1)
        for t in mpo.tensors
            @test size(t, 1) == 1
            @test size(t, 4) == 1
            d = size(t, 2)
            @test size(t, 3) == d
            
            # Check it acts as Identity
            mat = reshape(t, d, d)
            @test isapprox(mat, Matrix(I, d, d))
        end
    end

    @testset "Expectation Value (expect_mpo)" begin
        L = 4
        mps = random_mps(L, 2, 4)
        mpo = MPO(L; identity=true)
        
        # <psi| I |psi> = <psi|psi> = 1
        val = expect_mpo(mpo, mps)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
        
        # Scale MPO
        mpo.tensors[1] .*= 2.0
        val = expect_mpo(mpo, mps)
        @test isapprox(val, 2.0 + 0.0im; atol=1e-12)
    end

    @testset "Contract MPO x MPS" begin
        L = 3
        mps = random_mps(L, 2, 2)
        mpo = MPO(L; identity=true)
        
        # Apply Identity
        new_mps = contract_mpo_mps(mpo, mps)
        
        # Check overlap <psi|new_psi> approx 1
        overlap = scalar_product(mps, new_mps)
        @test isapprox(overlap, 1.0 + 0.0im; atol=1e-12)
        
        # Check Bond Dimensions increased (Identity MPO has bond 1, so bonds should match original)
        # But generally, D_new = D_mpo * D_mps
        for i in 1:L
            # Identity MPO bonds are 1
            # So output bonds should match input MPS bonds
            @test size(new_mps.tensors[i]) == size(mps.tensors[i])
        end
    end
    
    @testset "Contract MPO x MPO" begin
        L = 3
        mpo1 = MPO(L; identity=true)
        mpo2 = MPO(L; identity=true)
        
        mpo3 = contract_mpo_mpo(mpo1, mpo2)
        
        # Id * Id = Id
        mps = random_mps(L, 2, 2)
        val = expect_mpo(mpo3, mps)
        @test isapprox(val, 1.0 + 0.0im; atol=1e-12)
    end

end

