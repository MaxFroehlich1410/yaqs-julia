using Test
using LinearAlgebra
using StaticArrays

include("../src/GateLibrary.jl")
using .GateLibrary

@testset "GateLibrary" begin
    
    @testset "Pauli Matrices" begin
        @test matrix(XGate()) == [0 1; 1 0]
        
        # Pauli Y = [0 -i; i 0]
        # Expected matches my implementation
        expected_Y = [0 -1im; 1im 0]
        @test matrix(YGate()) ≈ expected_Y
        
        @test matrix(ZGate()) == [1 0; 0 -1]
        @test matrix(IdGate()) == [1 0; 0 1]
    end

    @testset "Hadamard" begin
        inv_sqrt2 = 1/sqrt(2)
        expected = [inv_sqrt2 inv_sqrt2; inv_sqrt2 -inv_sqrt2]
        @test matrix(HGate()) ≈ expected
    end
    
    @testset "Rotations" begin
        theta = π/2
        
        # Rx(pi/2) = [cos(pi/4) -i sin(pi/4); -i sin(pi/4) cos(pi/4)]
        #          = 1/sqrt(2) * [1 -i; -i 1]
        val = 1/sqrt(2)
        expected_Rx = [val -val*1im; -val*1im val]
        @test matrix(RxGate(theta)) ≈ expected_Rx
        
        # Rz(pi) = [-i 0; 0 i] (up to global phase? No, def is e^(-i t/2))
        # Rz(pi) -> e^(-i pi/2) = -i.
        # Diag(-i, i)
        @test matrix(RzGate(π)) ≈ [-1im 0; 0 1im]
    end
    
    @testset "Two Qubit Gates" begin
        # CX
        # [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        expected_CX = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        @test matrix(CXGate()) ≈ expected_CX
        
        # CZ
        expected_CZ = diagm([1, 1, 1, -1])
        @test matrix(CZGate()) ≈ expected_CZ
    end
    
    @testset "Allocation Check" begin
        # Ensure SMatrix is used (0 allocations for call)
        g = XGate()
        @test @allocated(matrix(g)) == 0
    end

end

