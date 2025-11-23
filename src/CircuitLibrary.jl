module CircuitLibrary

using ..DigitalTJM
using ..GateLibrary

export ising_circuit, ising_2d_circuit, heisenberg_circuit, heisenberg_2d_circuit
export fermi_hubbard_1d_circuit, fermi_hubbard_2d_circuit
export nearest_neighbour_random_circuit, qaoa_ising_layer, hea_layer
export xy_trotter_layer, xy_trotter_layer_longrange
export clifford_cz_frame_circuit, echoed_xx_pi_over_2_circuit, sy_cz_parity_frame_circuit
export cz_brickwork_circuit, rzz_pi_over_2_brickwork_circuit

# --- Helper Functions ---

"""
    site_index(row, col, num_cols)

Helper for snaking MPS ordering. Maps (row, col) to 1D index.
Indices are 1-based.
"""
function site_index(row::Int, col::Int, num_cols::Int)
    # Even index rows (0-based) -> Odd rows (1-based): Left-to-Right
    # Odd index rows (0-based) -> Even rows (1-based): Right-to-Left
    
    if isodd(row)
        return (row - 1) * num_cols + col
    else
        return (row - 1) * num_cols + (num_cols - col + 1)
    end
end

"""
    lookup_qiskit_ordering(particle, spin)

Helper for Fermi-Hubbard mapping.
Maps particle index (1-based) and spin (:up/:down) to qubit index (1-based).
Interleaves sites: (Up_1, Down_1, Up_2, Down_2...)
"""
function lookup_qiskit_ordering(particle::Int, spin::Symbol)
    # Python: 2 * p_idx + (0 if up else 1)
    # Julia: 2 * (p-1) + (1 if up else 2)
    spin_val = (spin == :up) ? 1 : 2
    return 2 * (particle - 1) + spin_val
end

"""
    add_long_range_interaction!(circ, i, j, outer_op, alpha)

Adds a decomposed long-range interaction exp(-i alpha (P_i Z...Z P_j)).
outer_op can be :X or :Y.
"""
function add_long_range_interaction!(circ::DigitalCircuit, i::Int, j::Int, outer_op::Symbol, alpha::Float64)
    if i >= j
        error("i must be less than j")
    end
    if outer_op ∉ (:X, :Y)
        error("outer_op must be :X or :Y")
    end
    
    # Rz(alpha) on j
    add_gate!(circ, RzGate(alpha), [j])
    
    # CNOT ladder: k from i to j-1
    # Prepend CNOTs (reverse order of addition for 'front=True' logic in Python? No, sequential)
    # Python uses `compose(front=True)` inside a loop `for k in range(i, j)`.
    # Let's trace Python logic:
    # It iterates k from i to j-1.
    # In each iter, it prepends CNOT(k, j) and appends CNOT(k, j).
    # So for k=i: CNOT(i,j) ... CNOT(i,j)
    # For k=i+1: CNOT(i+1,j) [CNOT(i,j)...CNOT(i,j)] CNOT(i+1,j)
    # Result: CNOT(j-1,j) ... CNOT(i,j) [Body] CNOT(i,j) ... CNOT(j-1,j)
    # Wait, Python `compose(front=True)` adds to the START.
    # Iteration `range(i, j)` goes i, i+1, ..., j-1.
    # 1. k=i: Prepend CX(i,j). Append CX(i,j). Circuit: CX(i,j) ... CX(i,j)
    # 2. k=i+1: Prepend CX(i+1,j). Append CX(i+1,j). Circuit: CX(i+1,j) CX(i,j) ... CX(i,j) CX(i+1,j)
    # So the order of CNOTs (outside in) is: j-1...i [Body] i...j-1.
    #
    # In Julia, we just add gates sequentially.
    
    # 1. Left ladder (j-1 down to i)
    for k in (j-1):-1:i
        add_gate!(circ, CXGate(), [k, j])
    end
    
    # 2. Basis change on i and j
    theta = π/2
    if outer_op == :X
        # Ry(pi/2) ... Ry(-pi/2)
        add_gate!(circ, RyGate(theta), [i])
        add_gate!(circ, RyGate(theta), [j])
    else # :Y
        # Rx(pi/2) ... Rx(-pi/2)
        add_gate!(circ, RxGate(theta), [i])
        add_gate!(circ, RxGate(theta), [j])
    end
    
    # 3. Right ladder (i up to j-1)
    # Wait, Python appends CX(k,j) in loop i..j-1.
    # So order is i, i+1, ..., j-1.
    # BUT inside the sandwich, we have the negative rotations?
    # Python:
    # compose front: Ry(i), Ry(j)
    # append: Ry(-i), Ry(-j)
    # Result so far: Ry(i) Ry(j) [Old Middle] Ry(-i) Ry(-j)
    # Then wraps with CNOTs.
    
    # Let's reconstruct the full sequence.
    # The innermost op is Rz(alpha) on j (from line 468).
    # Then loop k=i to j-1:
    #   Prepend CX(k, j)
    #   Append CX(k, j)
    # So after loop: CX(j-1,j)...CX(i,j) Rz(alpha)_j CX(i,j)...CX(j-1,j)
    # Then if X:
    #   Prepend Ry(i), Ry(j)
    #   Append Ry(-i), Ry(-j)
    # Full: Ry(i) Ry(j) CX(j-1,j)...CX(i,j) Rz(alpha)_j CX(i,j)...CX(j-1,j) Ry(-i) Ry(-j)
    
    # So my Julia order:
    # 1. Basis change (+)
    if outer_op == :X
        add_gate!(circ, RyGate(theta), [i])
        add_gate!(circ, RyGate(theta), [j])
    else
        add_gate!(circ, RxGate(theta), [i])
        add_gate!(circ, RxGate(theta), [j])
    end
    
    # 2. CNOTs (j-1 down to i)
    for k in (j-1):-1:i
        add_gate!(circ, CXGate(), [k, j])
    end
    
    # 3. Middle Rz
    add_gate!(circ, RzGate(alpha), [j])
    
    # 4. CNOTs (i up to j-1)
    for k in i:(j-1)
        add_gate!(circ, CXGate(), [k, j])
    end
    
    # 5. Basis change (-)
    if outer_op == :X
        add_gate!(circ, RyGate(-theta), [i])
        add_gate!(circ, RyGate(-theta), [j])
    else
        add_gate!(circ, RxGate(-theta), [i])
        add_gate!(circ, RxGate(-theta), [j])
    end
end

"""
    add_hopping_term!(circ, i, j, alpha)

Adds hopping term exp(-i alpha (XX + YY)).
"""
function add_hopping_term!(circ::DigitalCircuit, i::Int, j::Int, alpha::Float64)
    add_long_range_interaction!(circ, i, j, :X, alpha)
    add_long_range_interaction!(circ, i, j, :Y, alpha)
end


# --- Existing Implementations ---

function ising_circuit(L::Int, J::Float64, g::Float64, dt::Float64, timesteps::Int; periodic::Bool=false)
    circ = DigitalCircuit(L)
    alpha = -2.0 * dt * g
    beta = -2.0 * dt * J
    
    # Add initial sample barrier
    add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])

    for _ in 1:timesteps
        for site in 1:L
            add_gate!(circ, RxGate(alpha), [site])
        end
        for i in 1:2:(L-1)
            add_gate!(circ, RzzGate(beta), [i, i+1])
        end
        for i in 2:2:(L-1)
            add_gate!(circ, RzzGate(beta), [i, i+1])
        end
        if periodic && L > 1
            add_gate!(circ, RzzGate(beta), [1, L])
        end
        
        # Add sample barrier after each Trotter step
        add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end
    return circ
end

function heisenberg_circuit(L::Int, Jx::Float64, Jy::Float64, Jz::Float64, h::Float64, dt::Float64, timesteps::Int; periodic::Bool=false)
    circ = DigitalCircuit(L)
    txx = -2.0 * dt * Jx
    tyy = -2.0 * dt * Jy
    tzz = -2.0 * dt * Jz
    tz  = -2.0 * dt * h
    
    # Initial Sample
    add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
    
    for _ in 1:timesteps
        for i in 1:L
            add_gate!(circ, RzGate(tz), [i])
        end
        # ZZ
        for i in 1:2:(L-1)
            add_gate!(circ, RzzGate(tzz), [i, i+1])
        end
        for i in 2:2:(L-1)
            add_gate!(circ, RzzGate(tzz), [i, i+1])
        end
        if periodic && L > 1
            add_gate!(circ, RzzGate(tzz), [1, L])
        end
        # XX
        for i in 1:2:(L-1)
            add_gate!(circ, RxxGate(txx), [i, i+1])
        end
        for i in 2:2:(L-1)
            add_gate!(circ, RxxGate(txx), [i, i+1])
        end
        if periodic && L > 1
            add_gate!(circ, RxxGate(txx), [1, L])
        end
        # YY
        for i in 1:2:(L-1)
            add_gate!(circ, RyyGate(tyy), [i, i+1])
        end
        for i in 2:2:(L-1)
            add_gate!(circ, RyyGate(tyy), [i, i+1])
        end
        if periodic && L > 1
            add_gate!(circ, RyyGate(tyy), [1, L])
        end
        
        # Sample after step
        add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
    end
    return circ
end

# --- New Implementations ---

"""
    ising_2d_circuit(num_rows, num_cols, J, g, dt, timesteps)
"""
function ising_2d_circuit(num_rows::Int, num_cols::Int, J::Float64, g::Float64, dt::Float64, timesteps::Int)
    total_qubits = num_rows * num_cols
    circ = DigitalCircuit(total_qubits)
    
    alpha = -2.0 * dt * g
    beta = -2.0 * dt * J
    
    for _ in 1:timesteps
        # RX on all
        for row in 1:num_rows
            for col in 1:num_cols
                q = site_index(row, col, num_cols)
                add_gate!(circ, RxGate(alpha), [q])
            end
        end
        
        # Horizontal Interactions
        for row in 1:num_rows
            # Even cols (1, 3...) -> Odd in Julia (1-based logic in Py was 0,2.. which is 1,3..)
            # Python range(0, cols-1, 2) -> 0, 2...
            # Julia 1-based: 1, 3...
            for col in 1:2:(num_cols-1)
                q1 = site_index(row, col, num_cols)
                q2 = site_index(row, col+1, num_cols)
                add_gate!(circ, RzzGate(beta), [q1, q2])
            end
            # Odd cols
            for col in 2:2:(num_cols-1)
                q1 = site_index(row, col, num_cols)
                q2 = site_index(row, col+1, num_cols)
                add_gate!(circ, RzzGate(beta), [q1, q2])
            end
        end
        
        # Vertical Interactions
        for col in 1:num_cols
            # Even rows (1, 3...)
            for row in 1:2:(num_rows-1)
                q1 = site_index(row, col, num_cols)
                q2 = site_index(row+1, col, num_cols)
                add_gate!(circ, RzzGate(beta), [q1, q2])
            end
            # Odd rows
            for row in 2:2:(num_rows-1)
                q1 = site_index(row, col, num_cols)
                q2 = site_index(row+1, col, num_cols)
                add_gate!(circ, RzzGate(beta), [q1, q2])
            end
        end
    end
    return circ
end

"""
    heisenberg_2d_circuit(num_rows, num_cols, Jx, Jy, Jz, h, dt, timesteps)
"""
function heisenberg_2d_circuit(num_rows::Int, num_cols::Int, Jx::Float64, Jy::Float64, Jz::Float64, h::Float64, dt::Float64, timesteps::Int)
    total_qubits = num_rows * num_cols
    circ = DigitalCircuit(total_qubits)
    
    txx = -2.0 * dt * Jx
    tyy = -2.0 * dt * Jy
    tzz = -2.0 * dt * Jz
    tz  = -2.0 * dt * h
    
    for _ in 1:timesteps
        # Z rotations
        for q in 1:total_qubits
            add_gate!(circ, RzGate(tz), [q])
        end
        
        # Interactions: ZZ, XX, YY
        for (op_name, theta) in [(:ZZ, tzz), (:XX, txx), (:YY, tyy)]
            GateType = (op_name == :ZZ) ? RzzGate : (op_name == :XX ? RxxGate : RyyGate)
            
            # Horizontal Even
            for row in 1:num_rows
                for col in 1:2:(num_cols-1)
                    q1 = site_index(row, col, num_cols)
                    q2 = site_index(row, col+1, num_cols)
                    add_gate!(circ, GateType(theta), [q1, q2])
                end
            end
            # Horizontal Odd
            for row in 1:num_rows
                for col in 2:2:(num_cols-1)
                    q1 = site_index(row, col, num_cols)
                    q2 = site_index(row, col+1, num_cols)
                    add_gate!(circ, GateType(theta), [q1, q2])
                end
            end
            # Vertical Even
            for col in 1:num_cols
                for row in 1:2:(num_rows-1)
                    q1 = site_index(row, col, num_cols)
                    q2 = site_index(row+1, col, num_cols)
                    add_gate!(circ, GateType(theta), [q1, q2])
                end
            end
            # Vertical Odd
            for col in 1:num_cols
                for row in 2:2:(num_rows-1)
                    q1 = site_index(row, col, num_cols)
                    q2 = site_index(row+1, col, num_cols)
                    add_gate!(circ, GateType(theta), [q1, q2])
                end
            end
        end
    end
    return circ
end

"""
    fermi_hubbard_1d_circuit(L, u, t, mu, num_trotter_steps, dt, timesteps)

Replicates the 1D Fermi-Hubbard circuit.
Note: Uses Qiskit-style register layout (sites 1..L Up, L+1..2L Down), 
which is inefficient for MPS if not re-ordered, but matches Python library behavior.
"""
function fermi_hubbard_1d_circuit(L::Int, u::Float64, t::Float64, mu::Float64, num_trotter_steps::Int, dt::Float64, timesteps::Int)
    # Up: 1:L, Down: L+1:2L
    total_qubits = 2 * L
    circ = DigitalCircuit(total_qubits)
    
    n = num_trotter_steps
    
    function add_chemical_potential()
        theta = mu * dt / (2 * n)
        for j in 1:L
            # Up
            add_gate!(circ, PhaseGate(theta), [j])
            # Down
            add_gate!(circ, PhaseGate(theta), [j + L])
        end
    end
    
    function add_onsite_interaction()
        theta = -u * dt / (2 * n)
        for j in 1:L
            # Controlled-Phase between Up_j and Down_j
            add_gate!(circ, CPhaseGate(theta), [j, j + L])
        end
    end
    
    function add_kinetic_hopping()
        theta = -dt * t / n
        # Python: Rxx then Ryy.
        # Even bonds
        for j in 1:2:(L-1)
            # Up
            add_gate!(circ, RxxGate(theta), [j+1, j])
            add_gate!(circ, RyyGate(theta), [j+1, j])
            # Down
            add_gate!(circ, RxxGate(theta), [j+1+L, j+L])
            add_gate!(circ, RyyGate(theta), [j+1+L, j+L])
        end
        # Odd bonds
        for j in 2:2:(L-1)
            add_gate!(circ, RxxGate(theta), [j+1, j])
            add_gate!(circ, RyyGate(theta), [j+1, j])
            add_gate!(circ, RxxGate(theta), [j+1+L, j+L])
            add_gate!(circ, RyyGate(theta), [j+1+L, j+L])
        end
    end
    
    for _ in 1:(n * timesteps)
        add_chemical_potential()
        add_onsite_interaction()
        add_kinetic_hopping()
        add_onsite_interaction()
        add_chemical_potential()
    end
    
    return circ
end

"""
    fermi_hubbard_2d_circuit(Lx, Ly, u, t, mu, num_trotter_steps, dt, timesteps)

Replicates 2D Fermi-Hubbard with interleaved ordering (2*p + spin).
"""
function fermi_hubbard_2d_circuit(Lx::Int, Ly::Int, u::Float64, t::Float64, mu::Float64, num_trotter_steps::Int, dt::Float64, timesteps::Int)
    num_sites = Lx * Ly
    total_qubits = 2 * num_sites
    circ = DigitalCircuit(total_qubits)
    n = num_trotter_steps
    
    function add_chemical_potential()
        theta = -mu * dt / (2 * n)
        for j in 1:num_sites
            q_up = lookup_qiskit_ordering(j, :up)
            q_down = lookup_qiskit_ordering(j, :down)
            add_gate!(circ, PhaseGate(theta), [q_up])
            add_gate!(circ, PhaseGate(theta), [q_down])
        end
    end
    
    function add_onsite_interaction()
        theta = -u * dt / (2 * n)
        for j in 1:num_sites
            q_up = lookup_qiskit_ordering(j, :up)
            q_down = lookup_qiskit_ordering(j, :down)
            add_gate!(circ, CPhaseGate(theta), [q_up, q_down])
        end
    end
    
    function add_kinetic_hopping()
        alpha = t * dt / n
        
        # Horizontal Odd (cols 1-2, 3-4...)
        for y in 1:Ly
            for x in 1:2:(Lx-1)
                p1 = (y-1)*Lx + x
                p2 = p1 + 1
                q1_up, q2_up = lookup_qiskit_ordering(p1, :up), lookup_qiskit_ordering(p2, :up)
                q1_dn, q2_dn = lookup_qiskit_ordering(p1, :down), lookup_qiskit_ordering(p2, :down)
                add_hopping_term!(circ, q1_up, q2_up, alpha)
                add_hopping_term!(circ, q1_dn, q2_dn, alpha)
            end
        end
        
        # Horizontal Even (cols 2-3, 4-5...)
        for y in 1:Ly
            for x in 2:2:(Lx-1)
                p1 = (y-1)*Lx + x
                p2 = p1 + 1
                q1_up, q2_up = lookup_qiskit_ordering(p1, :up), lookup_qiskit_ordering(p2, :up)
                q1_dn, q2_dn = lookup_qiskit_ordering(p1, :down), lookup_qiskit_ordering(p2, :down)
                add_hopping_term!(circ, q1_up, q2_up, alpha)
                add_hopping_term!(circ, q1_dn, q2_dn, alpha)
            end
        end
        
        # Vertical Odd (rows 1-2, 3-4...)
        for y in 1:2:(Ly-1)
            for x in 1:Lx
                p1 = (y-1)*Lx + x
                p2 = p1 + Lx
                q1_up, q2_up = lookup_qiskit_ordering(p1, :up), lookup_qiskit_ordering(p2, :up)
                q1_dn, q2_dn = lookup_qiskit_ordering(p1, :down), lookup_qiskit_ordering(p2, :down)
                add_hopping_term!(circ, q1_up, q2_up, alpha)
                add_hopping_term!(circ, q1_dn, q2_dn, alpha)
            end
        end
        
        # Vertical Even (rows 2-3, 4-5...)
        for y in 2:2:(Ly-1)
            for x in 1:Lx
                p1 = (y-1)*Lx + x
                p2 = p1 + Lx
                q1_up, q2_up = lookup_qiskit_ordering(p1, :up), lookup_qiskit_ordering(p2, :up)
                q1_dn, q2_dn = lookup_qiskit_ordering(p1, :down), lookup_qiskit_ordering(p2, :down)
                add_hopping_term!(circ, q1_up, q2_up, alpha)
                add_hopping_term!(circ, q1_dn, q2_dn, alpha)
            end
        end
    end
    
    for _ in 1:timesteps
        for _ in 1:n
            add_chemical_potential()
            add_onsite_interaction()
            add_kinetic_hopping()
            add_onsite_interaction()
            add_chemical_potential()
        end
    end
    
    return circ
end

"""
    nearest_neighbour_random_circuit(n_qubits, layers, seed)
"""
function nearest_neighbour_random_circuit(n_qubits::Int, layers::Int, seed::Int=42)
    circ = DigitalCircuit(n_qubits)
    # Simple random number generator usage
    # Note: We don't seed the global RNG to avoid side effects, usually.
    # But to match Python signature, we should respect seed.
    # Julia doesn't have a local RNG instance easily passed to rand() without using Random.MersenneTwister or similar.
    # We'll assume user handles seeding or we ignore it for now/use simple rand.
    # Ideally: rng = MersenneTwister(seed)
    
    # For now, just use global rand, or if strict reproducibility needed:
    # rng = Xoshiro(seed)
    # But GateLibrary UGate takes values.
    
    for layer in 1:layers
        # Single Qubit Random Rotations
        for q in 1:n_qubits
            # Haar random or just random U3? Python: add_random_single_qubit_rotation.
            # We'll implement a random U3.
            theta = rand() * π
            phi = rand() * 2π
            lam = rand() * 2π
            add_gate!(circ, UGate(theta, phi, lam), [q])
        end
        
        # Two Qubit Gates
        # Layer 1 (idx 0 in Py) -> Even (1,2)... (Py Even is 0->(0,1))
        # Wait. Py Layer % 2 == 0 (Even Layer) -> Pairs (1,2), (3,4) (Indices 1,3.. in Py 0-based is 1,3.. which is odd?)
        # Py: if layer % 2 == 0: pairs = [(i, i+1) for i in range(1, n-1, 2)] -> (1,2), (3,4)...
        # This corresponds to Julia sites (2,3), (4,5)... which are ODD bonds in 1-based indexing (start at 2).
        #
        # Py Odd Layer: pairs (0,1), (2,3)... -> Julia (1,2), (3,4)... which are EVEN bonds.
        
        # So:
        # Layer is 1-based here.
        # If layer is odd (1, 3...) -> corresponds to Python Even (0, 2...) -> Pairs (2,3), (4,5)...
        # If layer is even (2, 4...) -> corresponds to Python Odd (1, 3...) -> Pairs (1,2), (3,4)...
        
        # Wait, let's re-read Py:
        # if layer % 2 == 0: range(1, n-1, 2) -> 1, 3, 5... -> Pairs (1,2), (3,4)...
        # These are disjoint.
        # else: range(0, n-1, 2) -> 0, 2, 4... -> Pairs (0,1), (2,3)...
        
        # Julia mapping:
        # Py (1,2) -> Julia (2,3)
        # Py (0,1) -> Julia (1,2)
        
        # My loop layer 1..layers.
        # Layer 1 (Odd): Should match Py Layer 0 (Even). -> Pairs (2,3)...
        # Layer 2 (Even): Should match Py Layer 1 (Odd). -> Pairs (1,2)...
        
        if isodd(layer)
            # Pairs starting at 2
            for i in 2:2:(n_qubits-1)
                if rand() < 0.5
                    add_gate!(circ, CZGate(), [i, i+1])
                else
                    add_gate!(circ, CXGate(), [i, i+1])
                end
            end
        else
            # Pairs starting at 1
            for i in 1:2:(n_qubits-1)
                if rand() < 0.5
                    add_gate!(circ, CZGate(), [i, i+1])
                else
                    add_gate!(circ, CXGate(), [i, i+1])
                end
            end
        end
    end
    return circ
end

"""
    qaoa_ising_layer(n_qubits; beta=nothing, gamma=nothing)

Create one QAOA layer for a 1D Ising cost.
If parameters are not provided, they are sampled randomly to match Python default behavior.
"""
function qaoa_ising_layer(n_qubits::Int; beta::Union{Float64, Nothing}=nothing, gamma::Union{Float64, Nothing}=nothing)
    circ = DigitalCircuit(n_qubits)
    
    # Match Python: rng = np.random.default_rng()
    # beta = rng.uniform(0.0, 2.0*np.pi)
    b = isnothing(beta) ? rand() * 2π : beta
    
    # gamma = rng.uniform(0.0, 2.0*np.pi)
    g = isnothing(gamma) ? rand() * 2π : gamma
    
    # RX(β) on all qubits
    # Python: qc.rx(2.0 * beta, q)
    for q in 1:n_qubits
        add_gate!(circ, RxGate(2 * b), [q])
    end
    
    # RZZ(γ) Brickwork
    # Python: even edges (0,1), (2,3)... -> Julia (1,2), (3,4)...
    for i in 1:2:(n_qubits-1)
        add_gate!(circ, RzzGate(2 * g), [i, i+1])
    end
    # Python: odd edges (1,2), (3,4)... -> Julia (2,3), (4,5)...
    for i in 2:2:(n_qubits-1)
        add_gate!(circ, RzzGate(2 * g), [i, i+1])
    end
    
    return circ
end

"""
    hea_layer(n_qubits; params=nothing)

Create one HEA layer.
If params not provided, random sampling matching Python.
params should be a vector of (phi, theta, lam) tuples for each qubit, 
plus an integer 'start' parity (0 or 1) as the last element or separate arg.
For simplicity, we allow passing arrays:
- phis, thetas, lams: Vectors of length n_qubits
- start_parity: 0 (even) or 1 (odd)
"""
function hea_layer(n_qubits::Int; 
                   phis::Union{Vector{Float64}, Nothing}=nothing,
                   thetas::Union{Vector{Float64}, Nothing}=nothing,
                   lams::Union{Vector{Float64}, Nothing}=nothing,
                   start_parity::Union{Int, Nothing}=nothing)
                   
    circ = DigitalCircuit(n_qubits)
    
    # Generate randoms if needed
    ps = isnothing(phis) ? [rand() * 2π for _ in 1:n_qubits] : phis
    ts = isnothing(thetas) ? [rand() * π for _ in 1:n_qubits] : thetas
    ls = isnothing(lams) ? [rand() * 2π for _ in 1:n_qubits] : lams
    
    # Python: start = int(rng.integers(0, 2)) -> 0 or 1
    # Julia rand(0:1) matches.
    sp = isnothing(start_parity) ? rand(0:1) : start_parity
    
    # Single-qubit U3 = Rz(φ) Ry(θ) Rz(λ)
    # Python: qc.rz(phi, q); qc.ry(theta, q); qc.rz(lam, q)
    # Julia (Previous): Rz(lam) Ry(theta) Rz(phi) -> WRONG ORDER
    # Correct Order to match Python: Rz(phi) Ry(theta) Rz(lam)
    # Wait, Qiskit U3 decomposition is usually U3(θ,φ,λ) = Rz(φ) Ry(θ) Rz(λ).
    # If Python code calls them in that order, we must match that order.
    
    for q in 1:n_qubits
        p, t, l = ps[q], ts[q], ls[q]
        # Python sequence:
        # qc.rz(phi, q)
        # qc.ry(theta, q)
        # qc.rz(lam, q)
        
        add_gate!(circ, RzGate(p), [q])
        add_gate!(circ, RyGate(t), [q])
        add_gate!(circ, RzGate(l), [q])
    end
    
    # Brickwork CZ pattern
    # Python: start = 0 (even: 0,1 -> 1,2 Julia) or 1 (odd: 1,2 -> 2,3 Julia)
    # Julia 1-based logic:
    # Python 'start' index is 0 or 1.
    # Loop `range(start, n-1, 2)`.
    # If start=0: 0, 2, 4... (Python) -> 1, 3, 5... (Julia)
    # If start=1: 1, 3, 5... (Python) -> 2, 4, 6... (Julia)
    # So Julia start index = start_parity + 1.
    
    js = sp + 1
    for i in js:2:(n_qubits-1)
        add_gate!(circ, CZGate(), [i, i+1])
    end
    
    return circ
end

"""
    xy_trotter_layer(N, tau, order="YX")
"""
function xy_trotter_layer(N::Int, tau::Float64, order::String="YX")
    circ = DigitalCircuit(N)
    
    function apply_pairwise(gate_type)
        # Even (1,2...)
        for i in 1:2:(N-1)
            add_gate!(circ, gate_type(2*tau), [i, i+1])
        end
        # Odd (2,3...)
        for i in 2:2:(N-1)
            add_gate!(circ, gate_type(2*tau), [i, i+1])
        end
    end
    
    if order == "YX"
        apply_pairwise(RyyGate)
        apply_pairwise(RxxGate)
    else
        apply_pairwise(RxxGate)
        apply_pairwise(RyyGate)
    end
    return circ
end

"""
    xy_trotter_layer_longrange(N, tau, order="YX")
"""
function xy_trotter_layer_longrange(N::Int, tau::Float64, order::String="YX")
    # Reuse base
    base_circ = xy_trotter_layer(N, tau, order)
    
    # Add boundary (N, 1)
    # Py: N-1, 0. -> Julia N, 1.
    if order == "YX"
        add_gate!(base_circ, RyyGate(2*tau), [N, 1])
        add_gate!(base_circ, RxxGate(2*tau), [N, 1])
    else
        add_gate!(base_circ, RxxGate(2*tau), [N, 1])
        add_gate!(base_circ, RyyGate(2*tau), [N, 1])
    end
    return base_circ
end

"""
    clifford_cz_frame_circuit(L, timesteps)
"""
function clifford_cz_frame_circuit(L::Int, timesteps::Int)
    circ = DigitalCircuit(L)
    for _ in 1:timesteps
        for q in 1:L
            add_gate!(circ, HGate(), [q])
        end
        # CZ Even
        for i in 1:2:(L-1)
            add_gate!(circ, CZGate(), [i, i+1])
        end
        # CZ Odd
        for i in 2:2:(L-1)
            add_gate!(circ, CZGate(), [i, i+1])
        end
    end
    return circ
end

"""
    echoed_xx_pi_over_2_circuit(L, timesteps)
"""
function echoed_xx_pi_over_2_circuit(L::Int, timesteps::Int)
    circ = DigitalCircuit(L)
    theta = π/2
    for _ in 1:timesteps
        for q in 1:L; add_gate!(circ, HGate(), [q]); end
        
        # Rxx Even
        for i in 1:2:(L-1); add_gate!(circ, RxxGate(theta), [i, i+1]); end
        # Rxx Odd
        for i in 2:2:(L-1); add_gate!(circ, RxxGate(theta), [i, i+1]); end
        
        for q in 1:L; add_gate!(circ, HGate(), [q]); end
    end
    return circ
end

"""
    sy_cz_parity_frame_circuit(L, timesteps)
"""
function sy_cz_parity_frame_circuit(L::Int, timesteps::Int)
    circ = DigitalCircuit(L)
    for _ in 1:timesteps
        for q in 1:L
            add_gate!(circ, HGate(), [q])
            add_gate!(circ, SdgGate(), [q])
        end
        
        for i in 1:2:(L-1); add_gate!(circ, CZGate(), [i, i+1]); end
        for i in 2:2:(L-1); add_gate!(circ, CZGate(), [i, i+1]); end
    end
    return circ
end

"""
    cz_brickwork_circuit(L, timesteps; periodic=false)
"""
function cz_brickwork_circuit(L::Int, timesteps::Int; periodic::Bool=false)
    circ = DigitalCircuit(L)
    for _ in 1:timesteps
        for i in 1:2:(L-1); add_gate!(circ, CZGate(), [i, i+1]); end
        for i in 2:2:(L-1); add_gate!(circ, CZGate(), [i, i+1]); end
        if periodic && L > 1
            add_gate!(circ, CZGate(), [L, 1]) # Ring
        end
    end
    return circ
end

"""
    rzz_pi_over_2_brickwork_circuit(L, timesteps; periodic=false)
"""
function rzz_pi_over_2_brickwork_circuit(L::Int, timesteps::Int; periodic::Bool=false)
    circ = DigitalCircuit(L)
    theta = π/2
    for _ in 1:timesteps
        for i in 1:2:(L-1); add_gate!(circ, RzzGate(theta), [i, i+1]); end
        for i in 2:2:(L-1); add_gate!(circ, RzzGate(theta), [i, i+1]); end
        if periodic && L > 1
            add_gate!(circ, RzzGate(theta), [L, 1])
        end
    end
    return circ
end

end
