# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Library of useful quantum circuits.

This module provides functions for creating quantum circuits that simulate
the dynamics of the Ising and Heisenberg models. The functions create_ising_circuit
and create_Heisenberg_circuit construct Qiskit QuantumCircuit objects based on specified
parameters such as the number of qubits, interaction strengths, time steps, and total simulation time.
These circuits are used to simulate the evolution of quantum many-body systems under the
respective Hamiltonians.
"""

from __future__ import annotations

# ignore non-lowercase argument names for physics notation
# ruff: noqa: N803
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister

from .circuit_library_utils import add_random_single_qubit_rotation


def create_ising_circuit(
    L: int, J: float, g: float, dt: float, timesteps: int, *, periodic: bool = False
) -> QuantumCircuit:
    """Create a 1D Ising Trotter circuit.

    This builds a layered circuit with RX rotations and RZZ interactions, optionally including a
    periodic boundary interaction between the last and first qubits.

    Args:
        L (int): Number of qubits in the circuit.
        J (float): Coupling constant for the ZZ interaction.
        g (float): Transverse field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.
        periodic (bool): Whether to add a periodic boundary interaction.

    Returns:
        QuantumCircuit: Circuit representing the Ising evolution.
    """
    # Angle on X rotation
    alpha = -2 * dt * g
    # Angle on ZZ rotation
    beta = -2 * dt * J

    circ = QuantumCircuit(L)
    for _ in range(timesteps):
        # Apply RX rotations on all qubits.
        for site in range(L):
            circ.rx(theta=alpha, qubit=site)
            circ.barrier()
        # Even-odd nearest-neighbor interactions.
        for site in range(L // 2):
            circ.rzz(beta, qubit1=2 * site, qubit2=2 * site + 1)
            circ.barrier()

        # Odd-even nearest-neighbor interactions.
        for site in range(1, L // 2):
            circ.rzz(beta, qubit1=2 * site - 1, qubit2=2 * site)
            circ.barrier()

        # For odd L > 1, handle the last pair.
        if L % 2 != 0 and L != 1:
            circ.rzz(beta, qubit1=L - 2, qubit2=L - 1)
            circ.barrier()

        # If periodic, add an additional long-range gate between qubit L-1 and qubit 0.
        if periodic and L > 1:
            circ.rzz(beta, qubit1=0, qubit2=L - 1)
            circ.barrier()

    return circ


def create_2d_ising_circuit(
    num_rows: int, num_cols: int, J: float, g: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """Create a 2D Ising Trotter circuit with snaking ordering.

    This maps a rectangular grid to a 1D snaking order, applies RX rotations, and inserts RZZ
    interactions along horizontal and vertical bonds for each Trotter step.

    Args:
        num_rows (int): Number of rows in the qubit grid.
        num_cols (int): Number of columns in the qubit grid.
        J (float): Coupling constant for the ZZ interaction.
        g (float): Transverse field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of Trotter steps.

    Returns:
        QuantumCircuit: Circuit representing the 2D Ising evolution.
    """
    total_qubits = num_rows * num_cols
    circ = QuantumCircuit(total_qubits)

    # Define a helper function to compute the snaking index.
    def site_index(row: int, col: int) -> int:
        """Map 2D grid coordinates to a snaking 1D index.

        This maps rows alternately left-to-right and right-to-left to produce an MPS-friendly
        ordering for a 2D grid.

        Args:
            row (int): Row index (0-based).
            col (int): Column index (0-based).

        Returns:
            int: Linear index in snaking order.
        """
        # For even rows, map left-to-right; for odd rows, map right-to-left.
        if row % 2 == 0:
            return row * num_cols + col
        return row * num_cols + (num_cols - 1 - col)

    # Single-qubit rotation and ZZ interaction angles.
    alpha = -2 * dt * g
    beta = -2 * dt * J

    for _ in range(timesteps):
        # Apply RX rotations to all qubits according to the snaking order.
        for row in range(num_rows):
            for col in range(num_cols):
                q = site_index(row, col)
                circ.rx(alpha, q)

        # Horizontal interactions: within each row, apply rzz gates between adjacent qubits.
        for row in range(num_rows):
            # Even bonds in the row.
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(beta, q1, q2)
                circ.barrier()
            # Odd bonds in the row.
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(beta, q1, q2)
                circ.barrier()

        # Vertical interactions: between adjacent rows.
        for col in range(num_cols):
            # Even bonds vertically.
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(beta, q1, q2)
                circ.barrier()

            # Odd bonds vertically.
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(beta, q1, q2)
                circ.barrier()

    return circ


def create_heisenberg_circuit(
    L: int, Jx: float, Jy: float, Jz: float, h: float, dt: float, timesteps: int, *, periodic: bool = False
) -> QuantumCircuit:
    """Create a 1D Heisenberg Trotter circuit.

    This builds a layered circuit with RZ field rotations and alternating RZZ, RXX, and RYY
    interactions, optionally including periodic boundary couplings.

    Args:
        L (int): Number of qubits (sites) in the circuit.
        Jx (float): Coupling constant for the XX interaction.
        Jy (float): Coupling constant for the YY interaction.
        Jz (float): Coupling constant for the ZZ interaction.
        h (float): Magnetic field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.
        periodic (bool): Whether to apply periodic boundary conditions.

    Returns:
        QuantumCircuit: Circuit representing the Heisenberg evolution.
    """
    theta_xx = -2 * dt * Jx
    theta_yy = -2 * dt * Jy
    theta_zz = -2 * dt * Jz
    theta_z = -2 * dt * h

    circ = QuantumCircuit(L)
    for _ in range(timesteps):
        # Z application
        for site in range(L):
            circ.rz(phi=theta_z, qubit=site)

        # ZZ application
        for site in range(L // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.rzz(theta=theta_zz, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.rzz(theta=theta_zz, qubit1=L - 2, qubit2=L - 1)

        # If periodic, add an additional long-range gate between qubit L-1 and qubit 0.
        if periodic and L > 1:
            circ.rzz(theta=theta_zz, qubit1=0, qubit2=L - 1)
            circ.barrier()

        # XX application
        for site in range(L // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.rxx(theta=theta_xx, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.rxx(theta=theta_xx, qubit1=L - 2, qubit2=L - 1)
            circ.barrier()

        if periodic and L > 1:
            circ.rxx(theta=theta_xx, qubit1=0, qubit2=L - 1)

        # YY application
        for site in range(L // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site, qubit2=2 * site + 1)

        for site in range(1, L // 2):
            circ.ryy(theta=theta_yy, qubit1=2 * site - 1, qubit2=2 * site)

        if L % 2 != 0 and L != 1:
            circ.ryy(theta=theta_yy, qubit1=L - 2, qubit2=L - 1)

        if periodic and L > 1:
            circ.ryy(theta=theta_yy, qubit1=0, qubit2=L - 1)

    return circ


def create_2d_heisenberg_circuit(
    num_rows: int, num_cols: int, Jx: float, Jy: float, Jz: float, h: float, dt: float, timesteps: int
) -> QuantumCircuit:
    """Create a 2D Heisenberg Trotter circuit with snaking ordering.

    This maps a rectangular grid to a 1D snaking order, applies RZ field rotations, and inserts
    RZZ, RXX, and RYY interactions along horizontal and vertical bonds.

    Args:
        num_rows (int): Number of rows in the qubit grid.
        num_cols (int): Number of columns in the qubit grid.
        Jx (float): Coupling constant for the XX interaction.
        Jy (float): Coupling constant for the YY interaction.
        Jz (float): Coupling constant for the ZZ interaction.
        h (float): Single-qubit Z-field strength.
        dt (float): Time step for the simulation.
        timesteps (int): Number of Trotter steps.

    Returns:
        QuantumCircuit: Circuit representing the 2D Heisenberg evolution.
    """
    total_qubits = num_rows * num_cols
    circ = QuantumCircuit(total_qubits)

    # Define a helper function to compute the snaking index.
    def site_index(row: int, col: int) -> int:
        # For even rows, map left-to-right; for odd rows, map right-to-left.
        if row % 2 == 0:
            return row * num_cols + col
        return row * num_cols + (num_cols - 1 - col)

    # Define the Trotter angles
    theta_xx = -2.0 * dt * Jx
    theta_yy = -2.0 * dt * Jy
    theta_zz = -2.0 * dt * Jz
    theta_z = -2.0 * dt * h

    for _ in range(timesteps):
        # (1) Apply single-qubit Z rotations to all qubits
        for row in range(num_rows):
            for col in range(num_cols):
                q = site_index(row, col)
                circ.rz(theta_z, q)

        # (2) ZZ interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(theta_zz, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rzz(theta_zz, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(theta_zz, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rzz(theta_zz, q1, q2)

        # (3) XX interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rxx(theta_xx, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.rxx(theta_xx, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rxx(theta_xx, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.rxx(theta_xx, q1, q2)

        # (4) YY interactions
        # Horizontal even bonds
        for row in range(num_rows):
            for col in range(0, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.ryy(theta_yy, q1, q2)
        # Horizontal odd bonds
        for row in range(num_rows):
            for col in range(1, num_cols - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row, col + 1)
                circ.ryy(theta_yy, q1, q2)
        # Vertical even bonds
        for col in range(num_cols):
            for row in range(0, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.ryy(theta_yy, q1, q2)
        # Vertical odd bonds
        for col in range(num_cols):
            for row in range(1, num_rows - 1, 2):
                q1 = site_index(row, col)
                q2 = site_index(row + 1, col)
                circ.ryy(theta_yy, q1, q2)

    return circ


def create_1d_fermi_hubbard_circuit(
    L: int, u: float, t: float, mu: float, num_trotter_steps: int, dt: float, timesteps: int
) -> QuantumCircuit:
    """Create a 1D Fermi-Hubbard Trotter circuit.

    This builds a circuit for the Fermi-Hubbard Hamiltonian using interleaved spin registers and
    applies chemical potential, onsite interaction, and kinetic hopping terms per Trotter step.

    Args:
        L (int): Number of sites in the model.
        u (float): On-site interaction parameter.
        t (float): Transfer energy parameter.
        mu (float): Chemical potential parameter.
        num_trotter_steps (int): Number of Trotter steps.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.

    Returns:
        QuantumCircuit: Circuit representing the 1D Fermi-Hubbard evolution.
    """
    n = num_trotter_steps

    spin_up = QuantumRegister(L, "↑")
    spin_down = QuantumRegister(L, "↓")
    circ = QuantumCircuit(spin_up, spin_down)

    def chemical_potential_term() -> None:
        """Append the chemical potential term for one sub-step.

        This applies phase gates to all spin-up and spin-down qubits using the current Trotter angle.

        Args:
            None

        Returns:
            None: The circuit is modified in-place.
        """
        theta = mu * dt / (2 * n)
        for j in range(L):
            circ.p(theta=theta, qubit=spin_up[j])
            circ.p(theta=theta, qubit=spin_down[j])

    def onsite_interaction_term() -> None:
        """Append the onsite interaction term for one sub-step.

        This applies controlled-phase gates between corresponding spin-up and spin-down sites.

        Args:
            None

        Returns:
            None: The circuit is modified in-place.
        """
        theta = -u * dt / (2 * n)
        for j in range(L):
            circ.cp(theta=theta, control_qubit=spin_up[j], target_qubit=spin_down[j])

    def kinetic_hopping_term() -> None:
        """Append the kinetic hopping term for one sub-step.

        This applies RXX and RYY gates on alternating bonds for both spin registers.

        Args:
            None

        Returns:
            None: The circuit is modified in-place.
        """
        theta = -dt * t / n
        for j in range(L - 1):
            if j % 2 == 0:
                circ.rxx(theta=theta, qubit1=spin_up[j + 1], qubit2=spin_up[j])
                circ.ryy(theta=theta, qubit1=spin_up[j + 1], qubit2=spin_up[j])
                circ.rxx(theta=theta, qubit1=spin_down[j + 1], qubit2=spin_down[j])
                circ.ryy(theta=theta, qubit1=spin_down[j + 1], qubit2=spin_down[j])
        for j in range(L - 1):
            if j % 2 != 0:
                circ.rxx(theta=theta, qubit1=spin_up[j + 1], qubit2=spin_up[j])
                circ.ryy(theta=theta, qubit1=spin_up[j + 1], qubit2=spin_up[j])
                circ.rxx(theta=theta, qubit1=spin_down[j + 1], qubit2=spin_down[j])
                circ.ryy(theta=theta, qubit1=spin_down[j + 1], qubit2=spin_down[j])

    for _ in range(n * timesteps):
        chemical_potential_term()
        onsite_interaction_term()
        kinetic_hopping_term()
        onsite_interaction_term()
        chemical_potential_term()

    return circ


def lookup_qiskit_ordering(particle: int, spin: str) -> int:
    """Map a particle index and spin to Qiskit qubit ordering.

    This implements the interleaved ordering used for Fermi-Hubbard circuits, mapping spin-up and
    spin-down particles to consecutive qubit indices.

    Args:
        particle (int): The index of the particle in the physical lattice.
        spin (str): '↑' for spin up, '↓' for spin down.

    Returns:
        int: The index in the 1D qubit-line.

    Raises:
        ValueError: If spin is neither '↑' nor '↓'.
    """
    if spin == "↑":
        spin_val = 0
    elif spin == "↓":
        spin_val = 1
    else:
        msg = "Spin must be '↑' or '↓."
        raise ValueError(msg)

    return 2 * particle + spin_val


def add_long_range_interaction(circ: QuantumCircuit, i: int, j: int, outer_op: str, alpha: float) -> None:
    """Add a decomposed long-range interaction between two qubits.

    This inserts basis changes and CNOT ladders to implement a long-range XX or YY interaction
    between qubits `i` and `j`.

    Args:
        circ (QuantumCircuit): Circuit to modify.
        i (int): Index of the first qubit.
        j (int): Index of the second qubit.
        outer_op (str): Outer operator, 'X' or 'Y'.
        alpha (float): Phase of the exponent.

    Returns:
        None: The circuit is modified in-place.

    Raises:
        IndexError: If `i` is greater than or equal to `j`.
        ValueError: If `outer_op` is not 'X' or 'Y'.
    """
    if i >= j:
        msg = "Assumption i < j violated."
        raise IndexError(msg)
    if outer_op not in {"x", "X", "y", "Y"}:
        msg = "Outer_op must be either 'X' or 'Y'."
        raise ValueError(msg)

    phi = 1 * alpha
    circ.rz(phi=phi, qubit=j)

    for k in range(i, j):
        # prepend the CNOT gate
        aux_circ = QuantumCircuit(circ.num_qubits)
        aux_circ.cx(control_qubit=k, target_qubit=j)
        circ.compose(aux_circ, front=True, inplace=True)
        # append the CNOT gate
        circ.cx(control_qubit=k, target_qubit=j)
    if outer_op in {"x", "X"}:
        theta = np.pi / 2
        # prepend the Ry gates
        aux_circ = QuantumCircuit(circ.num_qubits)
        aux_circ.ry(theta=theta, qubit=i)
        aux_circ.ry(theta=theta, qubit=j)
        circ.compose(aux_circ, front=True, inplace=True)
        # append the same Ry gates with negative phase
        circ.ry(theta=-theta, qubit=i)
        circ.ry(theta=-theta, qubit=j)
    elif outer_op in {"y", "Y"}:
        theta = np.pi / 2
        # prepend the Rx gates
        aux_circ = QuantumCircuit(circ.num_qubits)
        aux_circ.rx(theta=theta, qubit=i)
        aux_circ.rx(theta=theta, qubit=j)
        circ.compose(aux_circ, front=True, inplace=True)
        # append the same Rx gates with negative phase
        circ.rx(theta=-theta, qubit=i)
        circ.rx(theta=-theta, qubit=j)


def add_hopping_term(circ: QuantumCircuit, i: int, j: int, alpha: float) -> None:
    """Add a hopping term between two sites.

    This composes long-range XX and YY interactions to implement a hopping operator between qubits
    `i` and `j`.

    Args:
        circ (QuantumCircuit): The quantum circuit to modify.
        i (int): Index of the first qubit.
        j (int): Index of the second qubit.
        alpha (float): Phase of the exponent.

    Returns:
        None: The circuit is modified in-place.
    """
    circ_xx = QuantumCircuit(circ.num_qubits)
    circ_yy = QuantumCircuit(circ.num_qubits)
    add_long_range_interaction(circ_xx, i, j, "X", alpha)
    add_long_range_interaction(circ_yy, i, j, "Y", alpha)
    circ.compose(circ_xx, inplace=True)
    circ.compose(circ_yy, inplace=True)


def create_2d_fermi_hubbard_circuit(
    Lx: int, Ly: int, u: float, t: float, mu: float, num_trotter_steps: int, dt: float, timesteps: int
) -> QuantumCircuit:
    """Create a 2D Fermi-Hubbard Trotter circuit with interleaved ordering.

    This builds a circuit for a 2D lattice using interleaved spin ordering and applies chemical
    potential, onsite interaction, and kinetic hopping terms per Trotter step.

    Args:
        Lx (int): Number of columns in the grid lattice.
        Ly (int): Number of rows in the grid lattice.
        u (float): On-site interaction parameter.
        t (float): Transfer energy parameter.
        mu (float): Chemical potential parameter.
        num_trotter_steps (int): Number of Trotter steps.
        dt (float): Time step for the simulation.
        timesteps (int): Number of time steps to simulate.

    Returns:
        QuantumCircuit: Circuit representing the 2D Fermi-Hubbard evolution.
    """
    n = num_trotter_steps
    num_sites = Lx * Ly
    num_qubits = 2 * num_sites

    circ = QuantumCircuit(num_qubits)

    def chemical_potential_term() -> None:
        """Append the chemical potential term for one sub-step.

        This applies phase gates to all sites using the interleaved ordering.

        Args:
            None

        Returns:
            None: The circuit is modified in-place.
        """
        theta = -mu * dt / (2 * n)
        for j in range(num_sites):
            q_up = lookup_qiskit_ordering(j, "↑")
            q_down = lookup_qiskit_ordering(j, "↓")
            circ.p(theta=theta, qubit=q_up)
            circ.p(theta=theta, qubit=q_down)

    def onsite_interaction_term() -> None:
        """Append the onsite interaction term for one sub-step.

        This applies controlled-phase gates between up and down spins at each site.

        Args:
            None

        Returns:
            None: The circuit is modified in-place.
        """
        theta = -u * dt / (2 * n)
        for j in range(num_sites):
            q_up = lookup_qiskit_ordering(j, "↑")
            q_down = lookup_qiskit_ordering(j, "↓")
            circ.cp(theta=theta, control_qubit=q_up, target_qubit=q_down)

    def kinetic_hopping_term() -> None:
        """Append the kinetic hopping term for one sub-step.

        This applies hopping interactions along horizontal and vertical bonds for both spin sectors.

        Args:
            None

        Returns:
            None: The circuit is modified in-place.
        """
        alpha = t * dt / n

        def horizontal_odd() -> None:
            for y in range(Ly):
                for x in range(Lx - 1):
                    if x % 2 == 0:
                        p1 = y * Lx + x
                        p2 = p1 + 1
                        q1_up = lookup_qiskit_ordering(p1, "↑")
                        q2_up = lookup_qiskit_ordering(p2, "↑")
                        q1_down = lookup_qiskit_ordering(p1, "↓")
                        q2_down = lookup_qiskit_ordering(p2, "↓")
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)

        def horizontal_even() -> None:
            for y in range(Ly):
                for x in range(Lx - 1):
                    if x % 2 != 0:
                        p1 = y * Lx + x
                        p2 = p1 + 1
                        q1_up = lookup_qiskit_ordering(p1, "↑")
                        q2_up = lookup_qiskit_ordering(p2, "↑")
                        q1_down = lookup_qiskit_ordering(p1, "↓")
                        q2_down = lookup_qiskit_ordering(p2, "↓")
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)

        def vertical_odd() -> None:
            for y in range(Ly - 1):
                if y % 2 == 0:
                    for x in range(Lx):
                        p1 = y * Lx + x
                        p2 = p1 + Lx
                        q1_up = lookup_qiskit_ordering(p1, "↑")
                        q2_up = lookup_qiskit_ordering(p2, "↑")
                        q1_down = lookup_qiskit_ordering(p1, "↓")
                        q2_down = lookup_qiskit_ordering(p2, "↓")
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)

        def vertical_even() -> None:
            for y in range(Ly - 1):
                if y % 2 != 0:
                    for x in range(Lx):
                        p1 = y * Lx + x
                        p2 = p1 + Lx
                        q1_up = lookup_qiskit_ordering(p1, "↑")
                        q2_up = lookup_qiskit_ordering(p2, "↑")
                        q1_down = lookup_qiskit_ordering(p1, "↓")
                        q2_down = lookup_qiskit_ordering(p2, "↓")
                        add_hopping_term(circ, q1_up, q2_up, alpha)
                        add_hopping_term(circ, q1_down, q2_down, alpha)

        horizontal_odd()
        horizontal_even()
        vertical_odd()
        vertical_even()

    for _ in range(timesteps):
        for _ in range(n):
            chemical_potential_term()
            onsite_interaction_term()
            kinetic_hopping_term()
            onsite_interaction_term()
            chemical_potential_term()

    return circ


def nearest_neighbour_random_circuit(
    n_qubits: int,
    layers: int,
    seed: int = 42,
) -> QuantumCircuit:
    """Create a random nearest-neighbor circuit with alternating entanglers.

    This samples single-qubit rotations and nearest-neighbor CZ/CX gates layer by layer following
    the prescription in https://arxiv.org/abs/2002.07730.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        layers (int): Number of layers to apply.
        seed (int): RNG seed for reproducibility.

    Returns:
        QuantumCircuit: Random nearest-neighbor circuit.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)

    for layer in range(layers):
        # Single-qubit random rotations
        for qubit in range(n_qubits):
            add_random_single_qubit_rotation(qc, qubit, rng)

        # Two-qubit entangling gates
        if layer % 2 == 0:
            # Even layer → pair (1,2), (3,4), ...
            pairs = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
        else:
            # Odd layer → pair (0,1), (2,3), ...
            pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]

        for q1, q2 in pairs:
            if rng.random() < 0.5:
                qc.cz(q1, q2)
            else:
                qc.cx(q1, q2)

        qc.barrier()
    return qc


def qaoa_ising_layer(n_qubits: int) -> QuantumCircuit:
    """
    One QAOA layer for a 1D Ising cost:
        - Apply RX(β) on all qubits
        - Apply RZZ(γ) on nearest neighbors with an even-odd then odd-even sweep

    Angle conventions:
    You defined RX(β) = exp(-i β X) and RZZ(γ) = exp(-i γ Z⊗Z).
    Qiskit uses RX(θ) = exp(-i θ/2 X) and RZZ(θ) = exp(-i θ/2 Z⊗Z),
    so we pass θ_rx = 2β and θ_rzz = 2γ to QuantumCircuit.rx / .rzz.

    Args:
        n_qubits: number of qubits (≥1)

    Returns:
        QuantumCircuit with one QAOA layer.
    """
    rng = np.random.default_rng()
    beta = rng.uniform(0.0, 2.0*np.pi)
    gamma = rng.uniform(0.0, 2.0*np.pi)

    qc = QuantumCircuit(n_qubits, name="QAOA_layer")

    # RX(β) on all qubits
    for q in range(n_qubits):
        qc.rx(2.0 * beta, q)  # Qiskit θ = 2β

    # Cost unitary with brickwork execution:
    # even edges: (0,1), (2,3), ...
    for i in range(0, n_qubits - 1, 2):
        qc.rzz(2.0 * gamma, i, i + 1)  # Qiskit θ = 2γ

    # odd edges: (1,2), (3,4), ...
    for i in range(1, n_qubits - 1, 2):
        qc.rzz(2.0 * gamma, i, i + 1)

    qc.barrier()
    return qc


def hea_layer(n_qubits: int) -> QuantumCircuit:
    """
    One hardware-efficient ansatz (HEA) layer:
        - Arbitrary single-qubit rotations via U3 decomposition Rz(φ) Ry(θ) Rz(λ)
        - Brickwork CZ entanglers on neighbors, using either even-odd or odd-even pattern
          (chosen randomly for a single layer)

    Sampling:
        φ, λ ~ Uniform[0, 2π), θ ~ Uniform[0, π] (covers the Bloch sphere without redundancy).

    Args:
        n_qubits: number of qubits (≥1)

    Returns:
        QuantumCircuit with one HEA layer.
    """
    rng = np.random.default_rng()
    qc = QuantumCircuit(n_qubits, name="HEA_layer")

    # Single-qubit U3 = Rz(φ) Ry(θ) Rz(λ)
    for q in range(n_qubits):
        phi   = rng.uniform(0.0, 2.0*np.pi)
        theta = rng.uniform(0.0, np.pi)
        lam   = rng.uniform(0.0, 2.0*np.pi)
        qc.rz(phi, q)
        qc.ry(theta, q)
        qc.rz(lam, q)

    # Brickwork CZ pattern: pick parity at random for this layer
    start = int(rng.integers(0, 2))  # 0 => even edges; 1 => odd edges
    for i in range(start, n_qubits - 1, 2):
        qc.cz(i, i + 1)

    qc.barrier()
    return qc


def xy_trotter_layer(N, tau, order="YX") -> "QuantumCircuit":
    """Create one Trotter step for the XY Hamiltonian."""
    # Local import to avoid heavy imports at module load
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N)
    even = [(i, i+1) for i in range(0, N-1, 2)]
    odd  = [(i, i+1) for i in range(1, N-1, 2)]

    def apply_pairwise(gate_name):
        for a, b in even: 
            getattr(qc, gate_name)(2*tau, a, b)
        for a, b in odd:  
            getattr(qc, gate_name)(2*tau, a, b)

    if order == "YX":
        apply_pairwise("ryy")
        apply_pairwise("rxx")
    else:
        apply_pairwise("rxx")
        apply_pairwise("ryy")
    
    return qc


def xy_trotter_layer_longrange(
    N: int,
    tau: float,
    order: str = "YX",
) -> "QuantumCircuit":
    """Create one XY Trotter step with a single periodic boundary link.

    Starts from the nearest-neighbor XY layer (open chain) and adds the
    boundary coupling (N-1, 0) to make the quench effectively periodic
    with only one long-range link.
    """
    qc = xy_trotter_layer(N, tau, order=order)

    a, b = N - 1, 0
    if order == "YX":
        qc.ryy(2 * tau, a, b)
        qc.rxx(2 * tau, a, b)
    else:
        qc.rxx(2 * tau, a, b)
        qc.ryy(2 * tau, a, b)

    return qc


def longrange_test_circuit(N: int, theta: float) -> "QuantumCircuit":
    """Create a test circuit designed to isolate the effect of long-range noise.
    
    The circuit has:
    - Single-qubit H gates on all qubits (creates superposition)
    - Exactly ONE two-qubit gate: a long-range RXX gate between qubits N-1 and 0 (periodic boundary)
    - This makes the noise effect on the long-range gate very clear and measurable.
    
    Args:
        N: Number of qubits
        theta: Rotation angle for the RXX gate (typically π/4 or similar)
    
    Returns:
        QuantumCircuit with the test structure
    """
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(N)
    
    # 1. Apply H gates to all qubits to create superposition
    for q in range(N):
        qc.h(q)
    
    # 2. Apply exactly ONE long-range two-qubit gate: RXX between qubits N-1 and 0
    # This is the periodic boundary (N-1, 0) in 0-based indexing
    qc.rxx(theta, N - 1, 0)
    
    return qc

def create_clifford_cz_frame_circuit(L: int, timesteps: int) -> QuantumCircuit:
    """H-all then CZ brickwork (even/odd). Barrier after each CZ layer."""
    qc = QuantumCircuit(L, name="H-CZ-frame")
    for _ in range(timesteps):
        # Frame: H on all qubits
        for q in range(L):
            qc.h(q)
        qc.barrier()
        # CZ brickwork, even edges
        for i in range(0, L - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()
        # CZ brickwork, odd edges
        for i in range(1, L - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()
    return qc



def create_echoed_xx_pi_over_2(L: int, timesteps: int) -> QuantumCircuit:
    """H — RXX(pi/2) brickwork (even/odd) — H echo; barriers delimit windows."""
    qc = QuantumCircuit(L, name="Echoed-XX-pi/2")
    theta = np.pi/2  # Clifford angle
    for _ in range(timesteps):
        for q in range(L):
            qc.h(q)
        qc.barrier()
        for i in range(0, L - 1, 2):
            qc.rxx(theta, i, i + 1)
        qc.barrier()
        for i in range(1, L - 1, 2):
            qc.rxx(theta, i, i + 1)
        qc.barrier()
        for q in range(L):
            qc.h(q)
        qc.barrier()
    return qc


def create_sy_cz_parity_frame(L: int, timesteps: int) -> QuantumCircuit:
    """Sdg∘H frame → CZ brickwork (even/odd). Keeps readouts Pauli in the window."""
    qc = QuantumCircuit(L, name="Sy-CZ-parity")
    for _ in range(timesteps):
        for q in range(L):
            qc.h(q)      # H
            qc.sdg(q)    # S^\dagger (so Z -> Y)
        qc.barrier()
        for i in range(0, L - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()
        for i in range(1, L - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()
    return qc



def create_cz_brickwork_circuit(L: int, timesteps: int, *, periodic: bool = False) -> QuantumCircuit:
    """CZ brickwork (even/odd) with optional periodic ring link.

    Purpose: pure CZ brickwork with no single-qubit rotations, so Z-observables stay Pauli
    within each window. This creates absorbing windows under depolarizing noise and favors
    projector unraveling.

    Args:
        L: number of qubits (≥1)
        timesteps: number of brickwork cycles
        periodic: if True, also apply CZ(L-1, 0) in the odd sublayer (ring)

    Returns:
        QuantumCircuit with repeated CZ brickwork and barriers delimiting sublayers.
    """
    qc = QuantumCircuit(L, name="CZ-brickwork")
    for _ in range(timesteps):
        # Even edges: (0,1), (2,3), ...
        for i in range(0, L - 1, 2):
            qc.cz(i, i + 1)
        qc.barrier()

        # Odd edges: (1,2), (3,4), ...
        for i in range(1, L - 1, 2):
            qc.cz(i, i + 1)
        if periodic and L > 1:
            qc.cz(L - 1, 0)
        qc.barrier()
    return qc


def create_rzz_pi_over_2_brickwork(L: int, timesteps: int, *, periodic: bool = False) -> QuantumCircuit:
    """RZZ(pi/2) brickwork (even/odd) with optional periodic ring link.

    Purpose: Cliffordized Ising layer (RZZ at pi/2) that preserves Z-strings in the
    circuit frame, yielding absorbing windows under depolarizing noise. Projector
    unraveling should outperform standard/analog at moderate/high noise.

    Args:
        L: number of qubits (≥1)
        timesteps: number of brickwork cycles
        periodic: if True, also apply RZZ(pi/2) on (L-1, 0) in the odd sublayer (ring)

    Returns:
        QuantumCircuit with repeated RZZ(pi/2) brickwork and barriers per sublayer.
    """
    qc = QuantumCircuit(L, name="RZZ(pi/2)-brickwork")
    theta = np.pi / 2
    for _ in range(timesteps):
        # Even edges: (0,1), (2,3), ...
        for i in range(0, L - 1, 2):
            qc.rzz(theta, i, i + 1)
        qc.barrier()

        # Odd edges: (1,2), (3,4), ...
        for i in range(1, L - 1, 2):
            qc.rzz(theta, i, i + 1)
        if periodic and L > 1:
            qc.rzz(theta, L - 1, 0)
        qc.barrier()
    return qc