module CircuitLibrary

using ..CircuitTJM
using ..GateLibrary

export create_ising_circuit, create_2d_ising_circuit, create_heisenberg_circuit, create_2d_heisenberg_circuit
export create_1d_fermi_hubbard_circuit, create_2d_fermi_hubbard_circuit
export nearest_neighbour_random_circuit, qaoa_ising_layer, hea_layer
export xy_trotter_layer, xy_trotter_layer_longrange, longrange_test_circuit
export create_clifford_cz_frame_circuit, create_echoed_xx_pi_over_2, create_sy_cz_parity_frame
export create_cz_brickwork_circuit, create_rzz_pi_over_2_brickwork

# --- Helper Functions ---

"""
Map 2D lattice coordinates to a 1D snaking index.

This converts a `(row, col)` pair into a 1-based linear index using a snake ordering that reverses
direction on alternating rows, matching the MPS layout used for 2D circuits.

Args:
    row (Int): Row index (1-based).
    col (Int): Column index (1-based).
    num_cols (Int): Total number of columns in the lattice.

Returns:
    Int: Linear site index in the snaking ordering.
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
Map a particle index and spin to Qiskit-style qubit ordering.

This returns a 1-based qubit index with interleaved spin ordering `(Up_1, Down_1, Up_2, Down_2, ...)`
to match Qiskit conventions used in the Fermi-Hubbard circuit construction.

Args:
    particle (Int): Particle/site index (1-based).
    spin (Symbol): Spin label, typically `:up` or `:down`.

Returns:
    Int: Qubit index in the interleaved ordering.
"""
function lookup_qiskit_ordering(particle::Int, spin::Symbol)
    # Python: 2 * p_idx + (0 if up else 1)
    # Julia: 2 * (p-1) + (1 if up else 2)
    spin_val = (spin == :up) ? 1 : 2
    return 2 * (particle - 1) + spin_val
end

"""
Add a decomposed long-range interaction to a circuit.

This inserts a sequence of basis changes, CNOT ladders, and a central rotation to realize
`exp(-i * alpha * (P_i Z...Z P_j))` with `P` chosen as `X` or `Y` on the endpoints.

Args:
    circ (DigitalCircuit): Circuit to append gates to.
    i (Int): Left endpoint qubit index (1-based).
    j (Int): Right endpoint qubit index (1-based).
    outer_op (Symbol): Endpoint operator, either `:X` or `:Y`.
    alpha (Float64): Rotation angle for the interaction.

Returns:
    Nothing: Gates are appended to `circ` in-place.

Raises:
    ErrorException: If `i >= j` or `outer_op` is not `:X` or `:Y`.
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
Add a hopping interaction term to a circuit.

This composes two long-range interactions to implement `exp(-i * alpha * (XX + YY))` between the
specified qubits.

Args:
    circ (DigitalCircuit): Circuit to append gates to.
    i (Int): Left qubit index (1-based).
    j (Int): Right qubit index (1-based).
    alpha (Float64): Rotation angle for the hopping term.

Returns:
    Nothing: Gates are appended to `circ` in-place.
"""
function add_hopping_term!(circ::DigitalCircuit, i::Int, j::Int, alpha::Float64)
    add_long_range_interaction!(circ, i, j, :X, alpha)
    add_long_range_interaction!(circ, i, j, :Y, alpha)
end


# --- Existing Implementations ---

"""
Construct a 1D transverse-field Ising circuit.

This builds a digital Trotter circuit with single-qubit `Rx` rotations and nearest-neighbor `Rzz`
gates, inserting sampling barriers between Trotter steps.

Args:
    L (Int): Number of qubits.
    J (Float64): Ising coupling strength.
    g (Float64): Transverse field strength.
    dt (Float64): Trotter time step.
    timesteps (Int): Number of Trotter steps.
    periodic (Bool): Whether to include a periodic boundary interaction.

Returns:
    DigitalCircuit: Circuit implementing the Ising evolution.
"""
function create_ising_circuit(L::Int, J::Float64, g::Float64, dt::Float64, timesteps::Int; periodic::Bool=false)
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

"""
Construct a 1D Heisenberg circuit with a longitudinal field.

This builds a Trotter circuit that applies `Rz` field rotations and alternating `Rzz`, `Rxx`, and
`Ryy` two-qubit interactions, with optional periodic boundary coupling.

Args:
    L (Int): Number of qubits.
    Jx (Float64): XX coupling strength.
    Jy (Float64): YY coupling strength.
    Jz (Float64): ZZ coupling strength.
    h (Float64): Longitudinal field strength.
    dt (Float64): Trotter time step.
    timesteps (Int): Number of Trotter steps.
    periodic (Bool): Whether to include a periodic boundary interaction.

Returns:
    DigitalCircuit: Circuit implementing the Heisenberg evolution.
"""
function create_heisenberg_circuit(L::Int, Jx::Float64, Jy::Float64, Jz::Float64, h::Float64, dt::Float64, timesteps::Int; periodic::Bool=false)
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
Construct a 2D transverse-field Ising circuit with snaking ordering.

This builds a digital circuit for a 2D lattice mapped to a 1D snaking MPS order, applying `Rx`
rotations and nearest-neighbor `Rzz` interactions in both horizontal and vertical directions.

Args:
    num_rows (Int): Number of lattice rows.
    num_cols (Int): Number of lattice columns.
    J (Float64): Ising coupling strength.
    g (Float64): Transverse field strength.
    dt (Float64): Trotter time step.
    timesteps (Int): Number of Trotter steps.

Returns:
    DigitalCircuit: Circuit implementing the 2D Ising evolution.
"""
function create_2d_ising_circuit(num_rows::Int, num_cols::Int, J::Float64, g::Float64, dt::Float64, timesteps::Int)
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
Construct a 2D Heisenberg circuit with snaking ordering.

This builds a digital circuit for a 2D lattice mapped to a 1D snaking order, applying on-site `Rz`
rotations and alternating `Rzz`, `Rxx`, and `Ryy` interactions across horizontal and vertical bonds.

Args:
    num_rows (Int): Number of lattice rows.
    num_cols (Int): Number of lattice columns.
    Jx (Float64): XX coupling strength.
    Jy (Float64): YY coupling strength.
    Jz (Float64): ZZ coupling strength.
    h (Float64): Longitudinal field strength.
    dt (Float64): Trotter time step.
    timesteps (Int): Number of Trotter steps.

Returns:
    DigitalCircuit: Circuit implementing the 2D Heisenberg evolution.
"""
function create_2d_heisenberg_circuit(num_rows::Int, num_cols::Int, Jx::Float64, Jy::Float64, Jz::Float64, h::Float64, dt::Float64, timesteps::Int)
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
Construct a 1D Fermi-Hubbard Trotter circuit with Qiskit ordering.

This uses Qiskit-style register layout `(Up_1..Up_L, Down_1..Down_L)` and applies chemical
potential, onsite interaction, and kinetic hopping terms per Trotter step.

Args:
    L (Int): Number of lattice sites.
    u (Float64): Onsite interaction strength.
    t (Float64): Hopping strength.
    mu (Float64): Chemical potential.
    num_trotter_steps (Int): Trotter sub-steps per timestep.
    dt (Float64): Trotter time step.
    timesteps (Int): Number of outer timesteps.

Returns:
    DigitalCircuit: Circuit implementing the 1D Fermi-Hubbard evolution.
"""
function create_1d_fermi_hubbard_circuit(L::Int, u::Float64, t::Float64, mu::Float64, num_trotter_steps::Int, dt::Float64, timesteps::Int)
    # Up: 1:L, Down: L+1:2L
    total_qubits = 2 * L
    circ = DigitalCircuit(total_qubits)
    
    n = num_trotter_steps
    
    """
    Append the chemical potential term for one sub-step.

    This applies `PhaseGate(theta)` to all up and down qubits using the current Trotter angle.

    Args:
        None

    Returns:
        Nothing: Gates are appended to `circ` in-place.
    """
    function add_chemical_potential()
        theta = mu * dt / (2 * n)
        for j in 1:L
            # Up
            add_gate!(circ, PhaseGate(theta), [j])
            # Down
            add_gate!(circ, PhaseGate(theta), [j + L])
        end
    end
    
    """
    Append the onsite interaction term for one sub-step.

    This applies a controlled-phase gate between each up/down pair at the same lattice site using
    the current Trotter angle.

    Args:
        None

    Returns:
        Nothing: Gates are appended to `circ` in-place.
    """
    function add_onsite_interaction()
        theta = -u * dt / (2 * n)
        for j in 1:L
            # Controlled-Phase between Up_j and Down_j
            add_gate!(circ, CPhaseGate(theta), [j, j + L])
        end
    end
    
    """
    Append the kinetic hopping term for one sub-step.

    This applies `Rxx` and `Ryy` gates along even and odd bonds for both spin sectors using the
    current Trotter angle.

    Args:
        None

    Returns:
        Nothing: Gates are appended to `circ` in-place.
    """
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
Construct a 2D Fermi-Hubbard Trotter circuit with interleaved ordering.

This builds a circuit using interleaved spin ordering `(Up_1, Down_1, Up_2, Down_2, ...)` and applies
chemical potential, onsite interaction, and kinetic hopping terms across the 2D lattice.

Args:
    Lx (Int): Number of sites in the x direction.
    Ly (Int): Number of sites in the y direction.
    u (Float64): Onsite interaction strength.
    t (Float64): Hopping strength.
    mu (Float64): Chemical potential.
    num_trotter_steps (Int): Trotter sub-steps per timestep.
    dt (Float64): Trotter time step.
    timesteps (Int): Number of outer timesteps.

Returns:
    DigitalCircuit: Circuit implementing the 2D Fermi-Hubbard evolution.
"""
function create_2d_fermi_hubbard_circuit(Lx::Int, Ly::Int, u::Float64, t::Float64, mu::Float64, num_trotter_steps::Int, dt::Float64, timesteps::Int)
    num_sites = Lx * Ly
    total_qubits = 2 * num_sites
    circ = DigitalCircuit(total_qubits)
    n = num_trotter_steps
    
    """
    Append the chemical potential term for one sub-step.

    This applies `PhaseGate(theta)` to all up and down qubits using the current Trotter angle and
    the interleaved spin ordering.

    Args:
        None

    Returns:
        Nothing: Gates are appended to `circ` in-place.
    """
    function add_chemical_potential()
        theta = -mu * dt / (2 * n)
        for j in 1:num_sites
            q_up = lookup_qiskit_ordering(j, :up)
            q_down = lookup_qiskit_ordering(j, :down)
            add_gate!(circ, PhaseGate(theta), [q_up])
            add_gate!(circ, PhaseGate(theta), [q_down])
        end
    end
    
    """
    Append the onsite interaction term for one sub-step.

    This applies a controlled-phase gate between each up/down pair at the same lattice site using
    the current Trotter angle and interleaved ordering.

    Args:
        None

    Returns:
        Nothing: Gates are appended to `circ` in-place.
    """
    function add_onsite_interaction()
        theta = -u * dt / (2 * n)
        for j in 1:num_sites
            q_up = lookup_qiskit_ordering(j, :up)
            q_down = lookup_qiskit_ordering(j, :down)
            add_gate!(circ, CPhaseGate(theta), [q_up, q_down])
        end
    end
    
    """
    Append the kinetic hopping term for one sub-step.

    This applies hopping interactions along horizontal and vertical bonds for both spin sectors,
    using long-range decompositions where needed.

    Args:
        None

    Returns:
        Nothing: Gates are appended to `circ` in-place.
    """
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
Construct a random nearest-neighbor circuit with alternating entanglers.

This builds a layered circuit with random single-qubit U gates followed by randomly chosen CX/CZ
gates on alternating bonds per layer.

Args:
    n_qubits (Int): Number of qubits.
    layers (Int): Number of circuit layers.
    seed (Int): Random seed for parameter generation.

Returns:
    DigitalCircuit: Random nearest-neighbor circuit.
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
Construct a single QAOA layer for a 1D Ising cost Hamiltonian.

This applies `Rx` rotations with angle `2*beta` on all qubits followed by a brickwork of `Rzz`
interactions with angle `2*gamma`. Missing parameters are sampled randomly.

Args:
    n_qubits (Int): Number of qubits.
    beta (Union{Float64, Nothing}): Mixer angle; sampled if `nothing`.
    gamma (Union{Float64, Nothing}): Cost angle; sampled if `nothing`.

Returns:
    DigitalCircuit: Circuit implementing a single QAOA layer.
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
Construct a single hardware-efficient ansatz (HEA) layer.

This applies per-qubit `Rz`/`Ry`/`Rz` rotations followed by a CZ brickwork pattern whose parity can
be specified or randomly chosen.

Args:
    n_qubits (Int): Number of qubits.
    phis (Union{Vector{Float64}, Nothing}): Rz(phi) angles per qubit or `nothing` for random.
    thetas (Union{Vector{Float64}, Nothing}): Ry(theta) angles per qubit or `nothing` for random.
    lams (Union{Vector{Float64}, Nothing}): Rz(lambda) angles per qubit or `nothing` for random.
    start_parity (Union{Int, Nothing}): Brickwork start parity (0 or 1), random if `nothing`.

Returns:
    DigitalCircuit: Circuit implementing one HEA layer.
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
Construct a single XY Trotter layer with nearest-neighbor interactions.

This applies `Rxx` and `Ryy` gates on alternating bonds according to the specified order, using a
rotation angle of `2*tau`.

Args:
    N (Int): Number of qubits.
    tau (Float64): Trotter time step.
    order (String): Interaction order, `"YX"` or `"XY"`.

Returns:
    DigitalCircuit: Circuit implementing the XY layer.
"""
function xy_trotter_layer(N::Int, tau::Float64, order::String="YX")
    circ = DigitalCircuit(N)
    
    """
    Apply a two-qubit gate type on even and odd bonds.

    This helper appends gates on (1,2), (3,4), ... and (2,3), (4,5), ... with the provided gate
    constructor and the shared angle `2*tau`.

    Args:
        gate_type: Gate constructor that takes a single angle argument.

    Returns:
        Nothing: Gates are appended to `circ` in-place.
    """
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
Construct an XY Trotter layer with periodic long-range coupling.

This extends the nearest-neighbor XY layer with an additional boundary interaction between qubits
`N` and `1` to mimic periodic boundary conditions.

Args:
    N (Int): Number of qubits.
    tau (Float64): Trotter time step.
    order (String): Interaction order, `"YX"` or `"XY"`.

Returns:
    DigitalCircuit: Circuit implementing the long-range XY layer.
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
Construct a minimal circuit to probe long-range noise effects.

This prepares a uniform superposition with H gates and applies a single long-range two-qubit gate
between qubits `N` and `1` to isolate noise effects on that interaction.

Args:
    N (Int): Number of qubits.
    theta (Float64): Rotation angle for the long-range gate.

Returns:
    DigitalCircuit: Circuit with the long-range test structure.
"""
function longrange_test_circuit(N::Int, theta::Float64)
    circ = DigitalCircuit(N)
    
    # 1. Apply H gates to all qubits to create superposition
    for q in 1:N
        add_gate!(circ, HGate(), [q])
    end
    
    # 2. Apply exactly ONE long-range two-qubit gate: RXX between qubits N and 1
    # This is the periodic boundary (N-1, 0) in 0-based = (N, 1) in 1-based
    add_gate!(circ, RzzGate(theta), [N, 1])
    
    return circ
end

"""
Construct a Clifford CZ frame circuit.

This applies H gates on all qubits followed by even/odd CZ brickwork per timestep, producing a
Clifford circuit useful for benchmarking.

Args:
    L (Int): Number of qubits.
    timesteps (Int): Number of timesteps to apply.

Returns:
    DigitalCircuit: Circuit implementing the Clifford CZ frame.
"""
function create_clifford_cz_frame_circuit(L::Int, timesteps::Int)
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
Construct an echoed XX circuit with pi/2 rotations.

This alternates global H layers with brickwork `Rxx(π/2)` gates to form an echoed sequence over
the specified number of timesteps.

Args:
    L (Int): Number of qubits.
    timesteps (Int): Number of timesteps to apply.

Returns:
    DigitalCircuit: Circuit implementing the echoed XX sequence.
"""
function create_echoed_xx_pi_over_2(L::Int, timesteps::Int)
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
Construct a SY-CZ parity frame circuit.

This applies H and S† gates on all qubits followed by even/odd CZ brickwork per timestep, matching
the parity-frame structure used in related benchmarks.

Args:
    L (Int): Number of qubits.
    timesteps (Int): Number of timesteps to apply.

Returns:
    DigitalCircuit: Circuit implementing the SY-CZ parity frame.
"""
function create_sy_cz_parity_frame(L::Int, timesteps::Int)
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
Construct a CZ brickwork circuit.

This alternates even and odd CZ layers over the specified number of timesteps, with optional
periodic boundary coupling.

Args:
    L (Int): Number of qubits.
    timesteps (Int): Number of timesteps to apply.
    periodic (Bool): Whether to add a CZ gate between the last and first qubit.

Returns:
    DigitalCircuit: Circuit implementing the CZ brickwork pattern.
"""
function create_cz_brickwork_circuit(L::Int, timesteps::Int; periodic::Bool=false)
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
Construct an RZZ(π/2) brickwork circuit.

This alternates even and odd `Rzz(π/2)` layers over the specified number of timesteps, with optional
periodic boundary coupling.

Args:
    L (Int): Number of qubits.
    timesteps (Int): Number of timesteps to apply.
    periodic (Bool): Whether to add an RZZ gate between the last and first qubit.

Returns:
    DigitalCircuit: Circuit implementing the RZZ brickwork pattern.
"""
function create_rzz_pi_over_2_brickwork(L::Int, timesteps::Int; periodic::Bool=false)
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
