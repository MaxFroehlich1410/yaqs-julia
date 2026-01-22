using Test
using LinearAlgebra
using StaticArrays

if !isdefined(Main, :Yaqs)
    include("../src/Yaqs.jl")
    using .Yaqs
end
using .Yaqs.GateLibrary

@testset "GateLibrary" begin
    
    @testset "Abstract Types & Metadata" begin
        @test AbstractGate <: AbstractOperator
        @test AbstractNoise <: AbstractOperator

        @test XGate() isa AbstractGate
        @test RaisingGate() isa AbstractOperator
        @test !(RaisingGate() isa AbstractGate)

        # Defaults
        @test is_unitary(XGate()) == true
        @test hamiltonian_coeff(XGate()) == 1.0
    end

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
    
    @testset "Phase Gates" begin
        @test matrix(SGate()) ≈ [1 0; 0 1im]
        @test matrix(SdgGate()) ≈ [1 0; 0 -1im]
        @test matrix(TGate()) ≈ [1 0; 0 exp(1im*π/4)]
        @test matrix(TdgGate()) ≈ [1 0; 0 exp(-1im*π/4)]
        θ = 0.3
        @test matrix(PhaseGate(θ)) ≈ [1 0; 0 exp(1im*θ)]
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

        # Ry(pi) maps |0> -> |1> up to phase (here: [[0,1],[-1,0]])
        @test matrix(RyGate(π)) ≈ [0.0 1.0; -1.0 0.0]
    end
    
    @testset "UGate" begin
        # Spot-check against the definition in GateLibrary.matrix(::UGate).
        θ, ϕ, λ = 0.7, 0.2, -0.4
        U = matrix(UGate(θ, ϕ, λ))
        c = cos(θ/2)
        s = sin(θ/2)
        expected = [c exp(1im*ϕ)*s; -exp(1im*λ)*s exp(1im*(ϕ+λ))*c]
        @test U ≈ expected
    end

    @testset "Generators" begin
        θ = 0.9
        @test generator(RxGate(θ)) == [matrix(XGate())]
        @test generator(RyGate(θ)) == [matrix(YGate())]
        @test generator(RzGate(θ)) == [matrix(ZGate())]
        @test_throws ErrorException generator(SGate())
    end

    @testset "Two Qubit Gates" begin
        # CX
        # [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        expected_CX = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        @test matrix(CXGate()) ≈ expected_CX
        
        # CZ
        expected_CZ = diagm([1, 1, 1, -1])
        @test matrix(CZGate()) ≈ expected_CZ

        # CY
        expected_CY = [1 0 0 0;
                       0 1 0 0;
                       0 0 0 -1im;
                       0 0 1im 0]
        @test matrix(CYGate()) ≈ expected_CY

        # CH: controlled-H on target when control is |1>
        inv_sqrt2 = 1/sqrt(2)
        Hm = [inv_sqrt2 inv_sqrt2; inv_sqrt2 -inv_sqrt2]
        expected_CH = [1 0 0 0;
                       0 1 0 0;
                       0 0 Hm[1,1] Hm[1,2];
                       0 0 Hm[2,1] Hm[2,2]]
        @test matrix(CHGate()) ≈ expected_CH

        # SWAP and iSWAP
        expected_SWAP = [1 0 0 0;
                         0 0 1 0;
                         0 1 0 0;
                         0 0 0 1]
        @test matrix(SWAPGate()) ≈ expected_SWAP

        expected_iSWAP = [1 0 0 0;
                          0 0 1im 0;
                          0 1im 0 0;
                          0 0 0 1]
        @test matrix(iSWAPGate()) ≈ expected_iSWAP
    end
    
    @testset "Non-Unitary Operators" begin
        @test is_unitary(RaisingGate()) == false
        @test is_unitary(LoweringGate()) == false
        @test matrix(RaisingGate()) ≈ [0 0; 1 0]
        @test matrix(LoweringGate()) ≈ [0 1; 0 0]
    end

    @testset "Allocation Check" begin
        # Ensure SMatrix is used (0 allocations for call)
        g = XGate()
        @test @allocated(matrix(g)) == 0
    end

end

