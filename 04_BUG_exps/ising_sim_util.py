"""
This module provides everything needed to run an Ising model simulation
using the BUG algorithm.
"""
from __future__ import annotations

from copy import deepcopy
from enum import Enum

import numpy as np

try:
    from qutip import basis, sigmax, sigmaz, tensor, Options, mesolve, qeye
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    print("Warning: qutip not available. Install with: pip install qutip")

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.methods.bug import (bug,
                                       bug_second_order,
                                       fixed_bug,
                                       fixed_bug_second_order,
                                       hybrid_bug_second_order)
from mqt.yaqs.core.methods.tdvp import single_site_tdvp, two_site_tdvp

ISING_J_DEFAULT = 1.0
ISING_G_DEFAULT = 0.5
QUTIP_RESULTS_FILE_NAME = "qutip_ising_results.h5"
TIMES_KEY = "times"

class BUGKind(Enum):
    """
    The simulation method to use (BUG variants + TDVP baselines).
    """
    FIXED = 1
    ADAPTIVE = 2
    DOUBLEFIXED = 3
    DOUBLEADAPTIVE = 4
    HYBRID = 5
    SINGLE_SITE_TDVP = 6
    TWO_SITE_TDVP = 7

    def is_second_order(self) -> bool:
        """
        Check if the BUG kind is second order.
        """
        return self in {BUGKind.DOUBLEFIXED,
                        BUGKind.DOUBLEADAPTIVE,
                        BUGKind.HYBRID}

    def is_fixed(self) -> bool:
        """
        Check if the BUG kind is fixed.
        """
        return self in {BUGKind.FIXED,
                        BUGKind.DOUBLEFIXED}

    def needs_padded_initial_state(self) -> bool:
        """
        Whether to pad the initial MPS bond dimensions before time evolution.

        We pad for "fixed" methods that keep bond dimensions essentially constant,
        notably fixed BUG variants and single-site TDVP.
        """
        return self in {BUGKind.FIXED, BUGKind.DOUBLEFIXED, BUGKind.SINGLE_SITE_TDVP}

    def run_method(self):
        """
        Get the corresponding BUG method function.
        """
        if self == BUGKind.FIXED:
            return fixed_bug
        if self == BUGKind.ADAPTIVE:
            return bug
        if self == BUGKind.DOUBLEFIXED:
            return fixed_bug_second_order
        if self == BUGKind.DOUBLEADAPTIVE:
            return bug_second_order
        if self == BUGKind.HYBRID:
            return hybrid_bug_second_order
        if self == BUGKind.SINGLE_SITE_TDVP:
            return single_site_tdvp
        if self == BUGKind.TWO_SITE_TDVP:
            return two_site_tdvp
        raise ValueError(f"Unknown BUG kind: {self}")

    def file_key(self) -> str:
        """
        Get the file key for storing results.
        """
        return self.name.lower()

    def plot_kwargs(self,
                    ignore: set | None = None) -> dict:
        """
        Get plotting keyword arguments for this BUG kind.

        Args:
            ignore: Set of keys to ignore (for legend).

        Returns:
            Dictionary of plotting keyword arguments.
        """
        if ignore is None:
            ignore = set()
        if self == BUGKind.ADAPTIVE:
            out =  {"color": "tab:blue", "marker": "o", "label": "Adaptive", "ls": "dotted"}
        elif self == BUGKind.FIXED:
            out = {"color": "tab:orange", "marker": "s", "label": "Fixed", "ls": "dashed"}
        elif self == BUGKind.DOUBLEADAPTIVE:
            out = {"color": "tab:green", "marker": "^", "label": "2nd Order Adaptive", "ls": (0, (1,5))}
        elif self == BUGKind.DOUBLEFIXED:
            out = {"color": "tab:red", "marker": "v", "label": "2nd Order Fixed", "ls": (0, (5,10))}
        elif self == BUGKind.HYBRID:
            out = {"color": "tab:purple", "marker": "D", "label": "Hybrid 2nd Order", "ls": "dashdot"}
        elif self == BUGKind.SINGLE_SITE_TDVP:
            out = {"color": "tab:brown", "marker": "P", "label": "1-site TDVP", "ls": "solid"}
        elif self == BUGKind.TWO_SITE_TDVP:
            out = {"color": "tab:cyan", "marker": "X", "label": "2-site TDVP", "ls": "solid"}
        else:
            return {}
        for key in ignore:
            out.pop(key, None)
        return out

def create_initial_state(num_sites: int, state_type: str = "zeros") -> MPS:
    """Create initial MPS state.

    Args:
        num_sites: Number of sites in the chain.
        state_type: Type of initial state ('zeros', 'ones', 'plus').

    Returns:
        Initial MPS state.
    """
    if state_type == "zeros":
        return MPS(num_sites, state="zeros")
    elif state_type == "ones":
        return MPS(num_sites, state="ones")
    elif state_type == "plus":
        # Create |+⟩^⊗n state
        mps = MPS(num_sites, state="zeros")
        # For simplicity, start with |0⟩ state
        return mps
    else:
        raise ValueError(f"Unknown state type: {state_type}")


def create_ising_hamiltonian(num_sites: int, J: float = 1.0, g: float = 0.5) -> MPO:
    """Create Ising Hamiltonian as MPO.

    H = -J Σ Z_i Z_{i+1} - g Σ X_i

    Args:
        num_sites: Number of sites.
        J: Coupling strength.
        g: Transverse field strength.

    Returns:
        Hamiltonian as MPO.
    """
    mpo = MPO()
    mpo.init_ising(num_sites, J, g)
    return mpo


def qutip_time_evolution(
    num_sites: int,
    J: float,
    g: float,
    times: np.ndarray,
    initial_state: str = "zeros"
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Exact time evolution using qutip.

    Args:
        num_sites: Number of sites.
        J: Coupling strength.
        g: Transverse field strength.
        times: Array of time points.
        initial_state: Initial state type.

    Returns:
        Tuple of (times, expectation_values_per_site) where expectation_values_per_site
        is a list of arrays, one per site.
    """
    if not QUTIP_AVAILABLE:
        raise RuntimeError("qutip is required for this function")

    # Create Pauli operators for each site
    sx = sigmax()
    sz = sigmaz()
    identity = qeye(2)

    # Build operators for the full system
    sx_list = []
    sz_list = []
    for i in range(num_sites):
        # X on site i
        op_list_x = [identity] * num_sites
        op_list_x[i] = sx
        sx_list.append(tensor(op_list_x))
        # Z on site i
        op_list_z = [identity] * num_sites
        op_list_z[i] = sz
        sz_list.append(tensor(op_list_z))

    # Build Hamiltonian H = -J Σ Z_i Z_{i+1} - g Σ X_i
    H = 0
    # ZZ terms
    for i in range(num_sites - 1):
        H += -J * sz_list[i] * sz_list[i + 1]
    # X terms
    for i in range(num_sites):
        H += -g * sx_list[i]

    # Initial state
    if initial_state == "zeros":
        psi0 = tensor([basis(2, 0) for _ in range(num_sites)])
    elif initial_state == "ones":
        psi0 = tensor([basis(2, 1) for _ in range(num_sites)])
    else:
        psi0 = tensor([basis(2, 0) for _ in range(num_sites)])

    # Time evolution
    result = mesolve(H, psi0, times, [], sz_list,
                     options=Options(store_states=True))

    # Extract expectation values
    expect_values = []
    for i in range(num_sites):
        expect_values.append(np.real(result.expect[i]))

    return times, expect_values


def run_bug_time_evolution(
    num_sites: int,
    J: float,
    g: float,
    elapsed_time: float,
    dt: float,
    initial_state: str = "zeros",
    max_bond_dim: int = 32,
    threshold: float = 1e-10,
    num_steps: int | None = None,
    bug_kind: BUGKind = BUGKind.ADAPTIVE,
    verbose: bool = True
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run BUG algorithm for time evolution.

    Args:
        num_sites: Number of sites.
        J: Coupling strength.
        g: Transverse field strength.
        elapsed_time: Total evolution time.
        dt: Time step.
        initial_state: Initial state type.
        max_bond_dim: Maximum bond dimension.
        threshold: Truncation threshold.
        num_steps: Number of time steps (if None, computed from elapsed_time/dt).
        bug_kind: Kind of BUG algorithm to use.
        verbose: Whether to print progress.

    Returns:
        Tuple of (times, expectation_values_per_site).
    """
    # Create initial state
    mps = create_initial_state(num_sites, initial_state)
    if bug_kind.needs_padded_initial_state():
        mps.pad_bond_dimension(max_bond_dim)

    # Create Hamiltonian
    mpo = create_ising_hamiltonian(num_sites, J, g)

    # Compute number of steps
    if num_steps is None:
        num_steps = int(np.round(elapsed_time / dt))

    # Time array
    times = np.linspace(0, elapsed_time, num_steps + 1)

    # Storage for expectation values
    expect_values = [[] for _ in range(num_sites)]
    obs: dict[int, Observable] = {}

    # Compute initial expectation values
    for site in reversed(range(num_sites)):
        ob = Observable(Z(), site)
        obs[site] = ob
        if site == num_sites - 1:
            mps.set_canonical_form(site)
        else:
            mps.shift_orthogonality_center_left(site + 1)
        val = mps.expect(ob)
        expect_values[site].append(np.real(val))

    # Time evolution
    for step_idx in range(num_steps):
        # Set up simulation parameters
        sim_params = AnalogSimParams(
            observables=[Observable(Z(), 0)], # Dummy value to avoid assert
            elapsed_time=dt,
            dt=dt,
            threshold=threshold,
            max_bond_dim=max_bond_dim,
            show_progress=False
        )

        # Perform one BUG step (modifies mps in place)
        bug_method = bug_kind.run_method()
        bug_method(mps, mpo, sim_params, numiter_lanczos=25)

        # Set canonical form for accurate measurement
        mps.set_canonical_form(0)

        # Measure observables
        for site in range(num_sites):
            ob = obs[site]
            mps_copy = deepcopy(mps)
            # Shift to appropriate site for measurement
            if site > 0:
                for s in range(site):
                    mps_copy.shift_orthogonality_center_right(s)
            val = mps_copy.expect(ob)
            expect_values[site].append(np.real(val))

        if verbose:
            print(f"  Step {step_idx + 1}/{num_steps}, time={times[step_idx + 1]:.3f}")

    # Convert to numpy arrays
    expect_values = [np.array(vals) for vals in expect_values]

    return times, expect_values
