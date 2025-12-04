using Test
using Yaqs
using Yaqs.CircuitLibrary
using Yaqs.DigitalTJM
using Yaqs.GateLibrary

@testset "CircuitLibrary Tests" begin
    @testset "Ising Circuit" begin
        L = 4
        J = 1.0
        g = 0.5
        dt = 0.1
        timesteps = 2
        circ = create_ising_circuit(L, J, g, dt, timesteps; periodic=false)
        @test circ.num_qubits == L
        # Check 2-site gate. 
        # Gates: 1 Barrier, 4 Rx, then Rzz. So index 6.
        @test circ.gates[6].op isa RzzGate
    end
    
    @testset "2D Ising Circuit" begin
        rows = 2
        cols = 2
        circ = create_2d_ising_circuit(rows, cols, 1.0, 0.5, 0.1, 1)
        @test circ.num_qubits == 4
        # Should have RX on all 4
        # Then interactions
        # Horizontal: Row 1 (1-2), Row 2 (4-3 -> 8,7? No 3,4)
        # site_index(1,1,2)=1, site_index(1,2,2)=2.
        # site_index(2,1,2)=4, site_index(2,2,2)=3.
        # Horizontal: (1,2) and (4,3).
        # Vertical: (1,4) and (2,3).
        
        found_h = 0
        found_v = 0
        for g in circ.gates
            if g.op isa RzzGate
                s = sort(g.sites)
                if s == [1, 2] || s == [3, 4]
                    found_h += 1
                elseif s == [1, 4] || s == [2, 3]
                    found_v += 1
                end
            end
        end
        @test found_h >= 2
        @test found_v >= 2
    end
    
    @testset "Fermi Hubbard 1D" begin
        L = 2
        circ = create_1d_fermi_hubbard_circuit(L, 1.0, 1.0, 0.0, 1, 0.1, 1)
        @test circ.num_qubits == 4
        # Interactions between Up and Down (on-site)
        # Up: 1,2. Down: 3,4.
        # On-site: (1,3), (2,4).
        # Hopping: (1,2), (3,4).
        
        found_onsite = 0
        found_hopping = 0
        
        for g in circ.gates
            s = sort(g.sites)
            if g.op isa CPhaseGate
                if s == [1, 3] || s == [2, 4]
                    found_onsite += 1
                end
            elseif g.op isa RxxGate
                if s == [1, 2] || s == [3, 4]
                    found_hopping += 1
                end
            end
        end
        @test found_onsite >= 2
        @test found_hopping >= 2
    end
    
    @testset "Fermi Hubbard 2D" begin
        Lx = 2
        Ly = 1 # Essentially 1D 2-site
        circ = create_2d_fermi_hubbard_circuit(Lx, Ly, 1.0, 1.0, 0.0, 1, 0.1, 1)
        # 2 sites. 4 qubits.
        # Interleaved: Up1(1), Dn1(2), Up2(3), Dn2(4).
        # Onsite: (1,2), (3,4).
        # Hopping: (1,3), (2,4).
        
        found_onsite = 0
        found_hopping_gates = 0
        
        for g in circ.gates
            s = sort(g.sites)
            if g.op isa CPhaseGate
                if s == [1, 2] || s == [3, 4]
                    found_onsite += 1
                end
            end
            # Hopping adds long range interaction -> decomposed into many gates
            # We check for Rz (part of interaction) or CNOTs
            if g.op isa CXGate
               found_hopping_gates += 1
            end
        end
        @test found_onsite >= 2
        @test found_hopping_gates > 0
    end
    
    @testset "QAOA" begin
        circ = qaoa_ising_layer(4)
        @test circ.num_qubits == 4
    end
    
    @testset "Brickwork" begin
        circ = create_cz_brickwork_circuit(4, 1)
        found_cz = 0
        for g in circ.gates; if g.op isa CZGate; found_cz += 1; end; end
        @test found_cz == 3 # (1,2), (3,4), (2,3)
    end
    
    @testset "Frames" begin
        circ = create_sy_cz_parity_frame(2, 1)
        # H, Sdg, CZ
        has_h = false
        has_sdg = false
        has_cz = false
        for g in circ.gates
            if g.op isa HGate; has_h = true; end
            if g.op isa SdgGate; has_sdg = true; end
            if g.op isa CZGate; has_cz = true; end
        end
        @test has_h && has_sdg && has_cz
    end
end
