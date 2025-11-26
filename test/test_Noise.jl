using Test
using LinearAlgebra
using StaticArrays
using Yaqs
using Yaqs.NoiseModule
using Yaqs.GateLibrary
using Yaqs.MPOModule

@testset "Noise Model & Unraveling" begin

    @testset "Basic Process Parsing" begin
        # Test Pauli X noise
        proc_list = [Dict("name" => "pauli_x", "sites" => [1], "strength" => 0.1)]
        model = NoiseModel(proc_list, 2)
        
        @test length(model.processes) == 1
        p = model.processes[1]
        @test p isa LocalNoiseProcess
        @test p.name == "pauli_x"
        @test p.strength == 0.1
        @test p.matrix ≈ matrix(XGate())
    end
    
    @testset "Projector Unraveling" begin
        # Test Projector unraveling for Pauli Z
        # Should expand to 2 processes: (I+Z) and (I-Z)
        proc_list = [Dict("name" => "pauli_z", "sites" => [1], "strength" => 0.2, "unraveling" => "projector")]
        model = NoiseModel(proc_list, 2)
        
        @test length(model.processes) == 2
        
        p1 = model.processes[1]
        p2 = model.processes[2]
        
        @test startswith(p1.name, "projector_plus") || startswith(p1.name, "projector_minus")
        @test p1.strength == 0.1 # gamma/2
        
        # Check matrices: I+Z and I-Z
        Id = [1 0; 0 1]
        Z = [1 0; 0 -1]
        
        mat1 = p1.matrix
        mat2 = p2.matrix
        
        target1 = Id + Z
        target2 = Id - Z
        
        # Order might vary
        if p1.matrix ≈ target1
             @test p2.matrix ≈ target2
        else
             @test p1.matrix ≈ target2
             @test p2.matrix ≈ target1
        end
    end
    
    @testset "Unitary 2-Point Unraveling" begin
         proc_list = [Dict("name" => "pauli_x", "sites" => [1], "strength" => 0.2, 
                           "unraveling" => "unitary_2pt", "theta0" => 0.1)]
         model = NoiseModel(proc_list, 2)
         
         @test length(model.processes) == 2
         p1 = model.processes[1]
         @test p1.strength ≈ (0.2 / sin(0.1)^2) / 2
         
         # Check unitary: exp(±i theta X)
         X = [0 1; 1 0]
         U_plus = exp(1im * 0.1 * X)
         
         # Check if p1 or p2 matches U_plus
         matches = (p1.matrix ≈ U_plus) || (model.processes[2].matrix ≈ U_plus)
         @test matches
    end
    
    @testset "Long Range MPO Noise" begin
         # Long range crosstalk XY on sites 1 and 3
         proc_list = [Dict("name" => "crosstalk_xy", "sites" => [1, 3], "strength" => 0.05, "unraveling" => "projector")]
         model = NoiseModel(proc_list, 4)
         
         # Should produce 2 MPO processes
         @test length(model.processes) == 2
         p1 = model.processes[1]
         @test p1 isa MPONoiseProcess
         @test p1.mpo isa MPO
         @test p1.mpo.length == 4
         @test p1.sites == [1, 3]
         
         # Verify MPO structure logic (basic check)
         # Projector (I + XY). 
         # Site 1: [I, X]. Site 2: [I, I]. Site 3: [I, Y].
         # Just check dimensions and types
         @test length(p1.mpo.tensors) == 4
    end

end

