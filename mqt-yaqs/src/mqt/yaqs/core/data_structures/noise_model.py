# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Noise Models.

This module defines the NoiseModel class, which represents a noise model in a quantum system.
It stores a list of noise processes and their corresponding strengths, and automatically retrieves
the associated jump operator matrices from the NoiseLibrary. These jump operators are used to simulate
the effects of noise in quantum simulations.
"""




from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from scipy.linalg import expm
from ..libraries.noise_library import NoiseLibrary

if TYPE_CHECKING:
    from numpy.typing import NDArray

CROSSTALK_PREFIX = "longrange_crosstalk_"
PAULI_MAP = {
    "x": NoiseLibrary.pauli_x().matrix,
    "y": NoiseLibrary.pauli_y().matrix,
    "z": NoiseLibrary.pauli_z().matrix,
}

def get_pauli_string_matrix(p: dict[str, Any]) -> np.ndarray:
    # 1-site Pauli
    if len(p["sites"]) == 1:
        assert str(p["name"]).startswith("pauli_"), \
            "Unraveling supported only for Pauli one-site processes (or provide explicit 'matrix')."
        return NoiseModel.get_operator(str(p["name"]))
    # 2-site adjacent Pauli from crosstalk_ab name or explicit matrix
    ii, jj = p["sites"]
    assert abs(jj - ii) == 1, "Unraveling currently supports only adjacent 2-site processes."
    if str(p["name"]).startswith("crosstalk_"):
        suffix = str(p["name"]).rsplit("_", 1)[-1]
        assert len(suffix) == 2 and all(c in "xyz" for c in suffix), \
            "For 2-site unraveling, use crosstalk_ab with a,b in {x,y,z}, or provide 'matrix'."
        a, b = suffix[0], suffix[1]
        return np.kron(PAULI_MAP[a], PAULI_MAP[b])
    assert "matrix" in p, "For 2-site unraveling without crosstalk_ab, provide explicit 'matrix'."
    return p["matrix"]

def add_projector_expansion(processes_out: list[dict[str, Any]], proc: dict[str, Any], P: np.ndarray, gamma: float) -> None:
    dim = P.shape[0]
    I = np.eye(dim, dtype=complex)
    for comp, sign in (("plus", +1.0), ("minus", -1.0)):
        processes_out.append({
            "name": f"projector_{comp}_" + str(proc["name"]),
            "sites": list(proc["sites"]),
            "strength": gamma / 2.0,          # L = sqrt(γ/2) (I ± P)
            "matrix": (I + sign * P),         # sum L†L = 2γ I as desired
        })

def add_unitary_2pt_expansion(processes_out: list[dict[str, Any]], proc: dict[str, Any], P: np.ndarray, gamma: float, theta0: float) -> None:
    s_val = float(np.sin(theta0) ** 2)
    assert s_val > 0.0, "theta0 too small; sin^2(theta0) must be > 0."
    lam = gamma / s_val
    for comp, sign in (("plus", +1.0), ("minus", -1.0)):
        U = expm(1j * sign * theta0 * P)
        processes_out.append({
            "name": f"unitary2pt_{comp}_" + str(proc["name"]),
            "sites": list(proc["sites"]),
            "strength": lam / 2.0,            # two unitary collapses splitting λ
            "matrix": U,
        })

def add_unitary_gauss_expansion(
    processes_out: list[dict[str, Any]],
    proc: dict[str, Any],
    P: np.ndarray,
    gamma: float,
    sigma: float,
    gauss_M: int,
    gauss_k: float,
) -> None:
    M = int(proc.get("M", gauss_M))
    theta_max = float(proc.get("theta_max", gauss_k * sigma))
    thetas_pos = np.linspace(0.0, theta_max, (M + 1) // 2)
    thetas = np.concatenate([-thetas_pos[:0:-1], thetas_pos])
    w = np.exp(-0.5 * (thetas / sigma) ** 2)
    w /= w.sum()
    w = 0.5 * (w + w[::-1])   # symmetrize exactly
    s_weight = float(np.sum(w * (np.sin(thetas) ** 2)))
    assert s_weight > 1e-12, "E[sin^2 θ] too small; increase sigma or theta_max/M."
    lam = gamma / s_weight
    for idx, (wk, th) in enumerate(zip(w, thetas)):
        if wk <= 0.0:
            continue
        U = expm(1j * th * P)
        processes_out.append({
            "name": f"unitary_gauss_{idx}_" + str(proc["name"]),
            "sites": list(proc["sites"]),
            "strength": lam * float(wk),      # mixture weights times λ
            "matrix": U,
        })


def _parse_longrange_factors(proc: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (sigma, tau) as 2x2 matrices for a long-range two-site Pauli process.
    Prefers explicit 'factors' if present; otherwise parses the crosstalk suffix.
    """
    if "factors" in proc:
        a, b = proc["factors"]
        return np.array(a, dtype=complex), np.array(b, dtype=complex)

    name = str(proc["name"])
    if name.startswith("crosstalk_") or name.startswith("longrange_crosstalk_"):
        suffix = name.rsplit("_", 1)[-1]
        assert len(suffix) == 2 and all(c in "xyz" for c in suffix), \
            f"Invalid long-range label suffix: {suffix!r}"
        a, b = suffix[0], suffix[1]
        return np.array(PAULI_MAP[a], dtype=complex), np.array(PAULI_MAP[b], dtype=complex)

    raise ValueError("Cannot infer long-range Pauli factors. Provide proc['factors'] = (sigma, tau).")


def _build_I_plus_bP_mpo_lr(
    L: int,
    i: int,
    j: int,
    sigma: np.ndarray,
    tau: np.ndarray,
    a: complex,
    b: complex,
) -> list[np.ndarray]:
    """
    Bond-2 MPO (left-right order) for O = a*I + b*(sigma_i ⊗ tau_j) on a length-L chain (qubits).
    Tensors are returned as a list with shapes (left, right, 2, 2) at each site.
    """
    assert 0 <= i < j < L, "Require 0 <= i < j < L."
    Id2 = np.eye(2, dtype=complex)

    tensors: list[np.ndarray] = []

    # Sites < i : 1x1 identity blocks
    for _ in range(i):
        tensors.append(Id2.reshape(1, 1, 2, 2))

    # Site i : row block [ I , sigma ]  -> shape (1,2,2,2)
    Wi = np.zeros((1, 2, 2, 2), dtype=complex)
    Wi[0, 0] = Id2
    Wi[0, 1] = sigma
    tensors.append(Wi)

    # i < l < j : diagonal blocks diag(I, I) -> shape (2,2,2,2)
    for _ in range(i + 1, j):
        Wmid = np.zeros((2, 2, 2, 2), dtype=complex)
        Wmid[0, 0] = Id2
        Wmid[1, 1] = Id2
        tensors.append(Wmid)

    # Site j : column block [ a*I ; b*tau ] -> shape (2,1,2,2)
    Wj = np.zeros((2, 1, 2, 2), dtype=complex)
    Wj[0, 0] = a * Id2
    Wj[1, 0] = b * tau
    tensors.append(Wj)

    # Sites > j : 1x1 identity blocks
    for _ in range(j + 1, L):
        tensors.append(Id2.reshape(1, 1, 2, 2))

    return tensors


def _build_aI_plus_bP_mpo_phys(
    L: int,
    i: int,
    j: int,
    sigma: np.ndarray,
    tau: np.ndarray,
    a: complex,
    b: complex,
) -> list[np.ndarray]:
    """
    Bond-2 MPO in INTERNAL order (σ_out, σ_in, left, right) for O = a*I + b*(sigma_i ⊗ tau_j),
    exact for non-adjacent two-site Pauli strings. (Row block at i, diag(I,I) in (i,j), column block at j.)
    
    Args:
        L: Total number of qubits in the system
        i: Left site index (i < j)
        j: Right site index (i < j)
        sigma: 2x2 Pauli matrix acting on site i
        tau: 2x2 Pauli matrix acting on site j
        a: Coefficient of identity operator
        b: Coefficient of Pauli string operator
        
    Returns:
        List of MPO tensors in internal order (σ_out, σ_in, left, right)
    """
    assert 0 <= i < j < L, "Require 0 <= i < j < L."
    Id2 = np.eye(2, dtype=complex)
    tensors: list[np.ndarray] = []

    # sites < i: identity 1x1
    for _ in range(i):
        tensors.append(Id2.reshape(2, 2, 1, 1))

    # site i: row [ I , sigma ]  (left,right)=(1,2) then transpose to (σ_out,σ_in,left,right)
    Wi_lr = np.zeros((1, 2, 2, 2), dtype=complex)
    Wi_lr[0, 0] = Id2
    Wi_lr[0, 1] = sigma
    tensors.append(np.transpose(Wi_lr, (2, 3, 0, 1)))

    # i < l < j: diag(I, I)
    for _ in range(i + 1, j):
        Wmid_lr = np.zeros((2, 2, 2, 2), dtype=complex)
        Wmid_lr[0, 0] = Id2
        Wmid_lr[1, 1] = Id2
        tensors.append(np.transpose(Wmid_lr, (2, 3, 0, 1)))

    # site j: column [ a*I ; b*tau ]
    Wj_lr = np.zeros((2, 1, 2, 2), dtype=complex)
    Wj_lr[0, 0] = a * Id2
    Wj_lr[1, 0] = b * tau
    tensors.append(np.transpose(Wj_lr, (2, 3, 0, 1)))

    # sites > j: identity 1x1
    for _ in range(j + 1, L):
        tensors.append(Id2.reshape(2, 2, 1, 1))

    return tensors


def add_projector_expansion_longrange_mpo(
    processes_out: list[dict[str, Any]],
    proc: dict[str, Any],
    *,
    L: int,
    gamma: float,
) -> None:
    """
    Long-range projector unraveling for a two-site Pauli string P=σ_i τ_j (i<j, non-adjacent).
    Appends two processes with MPOs for (I ± P) and rate gamma/2 each.
    The MPO tensors are stored directly in your internal order (σ_out, σ_in, left, right),
    so you can call MPO.init_custom(mpo, transpose=False) and contract into the MPS.
    """
    sites = sorted(proc["sites"])
    assert len(sites) == 2 and abs(sites[1] - sites[0]) > 1, \
        "Use dense 2×2/4×4 for single-site/adjacent; this is for non-adjacent two-site."

    i, j = sites

    # --- get single-site factors σ, τ for P = σ_i τ_j
    sigma, tau = _parse_longrange_factors(proc)  # returns 2×2 matrices

    Id2 = np.eye(2, dtype=complex)

    def build_mpo_phys(a: complex, b: complex) -> list[np.ndarray]:
        """
        Bond-2 MPO in internal order (σ_out, σ_in, left, right) for O = a*I + b*(σ_i τ_j).
        """
        tensors: list[np.ndarray] = []

        # sites < i: 1×1 identity blocks
        for _ in range(i):
            tensors.append(Id2.reshape(2, 2, 1, 1))

        # site i: row [ I  σ ]  -> (left,right)=(1,2), then transpose to (phys,phys,left,right)
        Wi_lr = np.zeros((1, 2, 2, 2), dtype=complex)
        Wi_lr[0, 0] = Id2
        Wi_lr[0, 1] = sigma
        Wi = np.transpose(Wi_lr, (2, 3, 0, 1))
        tensors.append(Wi)

        # i < l < j: diag(I, I)
        for _ in range(i + 1, j):
            Wmid_lr = np.zeros((2, 2, 2, 2), dtype=complex)
            Wmid_lr[0, 0] = Id2
            Wmid_lr[1, 1] = Id2
            Wmid = np.transpose(Wmid_lr, (2, 3, 0, 1))
            tensors.append(Wmid)

        # site j: column [ a*I ; b*τ ]
        Wj_lr = np.zeros((2, 1, 2, 2), dtype=complex)
        Wj_lr[0, 0] = a * Id2
        Wj_lr[1, 0] = b * tau
        Wj = np.transpose(Wj_lr, (2, 3, 0, 1))
        tensors.append(Wj)

        # sites > j: 1×1 identity blocks
        for _ in range(j + 1, L):
            tensors.append(Id2.reshape(2, 2, 1, 1))

        return tensors

    # Two projector branches: (I ± P) with rates γ/2
    for comp, bsign in (("plus", +1.0), ("minus", -1.0)):
        mpo = build_mpo_phys(a=1.0, b=bsign)
        processes_out.append({
            "name": f"projector_{comp}_" + str(proc["name"]),
            "sites": [i, j],
            "strength": gamma / 2.0,   # L = sqrt(γ/2) * (I ± P)
            "mpo": mpo,                # IN INTERNAL ORDER (σ_out, σ_in, left, right)
            "mpo_bond_dim": 2,
        })


def add_unitary_2pt_expansion_longrange_mpo(
    processes_out: list[dict[str, Any]],
    proc: dict[str, Any],
    *,
    L: int,
    gamma: float,
    theta0: float,
) -> None:
    """
    Long-range analog (two-point) unraveling: U_{±} = exp(± i θ0 P) with P = σ_i τ_j (i<j).
    Bond-2 MPO with a = cos θ0, b = ± i sin θ0. Two components, each at rate λ/2 with λ = γ / sin^2 θ0.
    
    Args:
        processes_out: List to append expanded processes to
        proc: Original process dictionary with 'name', 'sites', 'strength'
        L: Total number of qubits in the system
        gamma: Original Lindblad rate
        theta0: Rotation angle for two-point unraveling
    """
    i, j = sorted(proc["sites"])
    sigma, tau = _parse_longrange_factors(proc)
    s_val = float(np.sin(theta0) ** 2)
    assert s_val > 0.0, "theta0 too small; sin^2(theta0) must be > 0."
    lam = gamma / s_val
    for comp, sgn in (("plus", +1.0), ("minus", -1.0)):
        a = np.cos(theta0)
        b = 1j * sgn * np.sin(theta0)
        mpo = _build_aI_plus_bP_mpo_phys(L, i, j, sigma, tau, a, b)
        processes_out.append({
            "name": f"unitary2pt_{comp}_" + str(proc["name"]),
            "sites": [i, j],
            "strength": lam / 2.0,  # two equal components sum to λ
            "mpo": mpo,
            "mpo_bond_dim": 2,
        })


def add_unitary_gauss_expansion_longrange_mpo(
    processes_out: list[dict[str, Any]],
    proc: dict[str, Any],
    *,
    L: int,
    gamma: float,
    sigma: float,
    gauss_M: int,
    gauss_k: float,
) -> None:
    """
    Long-range analog (Gaussian) unraveling: discrete symmetric quadrature θ_k with weights w_k.
    Each component U(θ_k) has bond-2 MPO with a = cos θ_k, b = i sin θ_k, and rate λ w_k,
    where λ = γ / E_w[sin^2 θ].
    
    Args:
        processes_out: List to append expanded processes to
        proc: Original process dictionary with 'name', 'sites', 'strength'
        L: Total number of qubits in the system
        gamma: Original Lindblad rate
        sigma: Standard deviation for Gaussian distribution
        gauss_M: Number of discretization points
        gauss_k: Factor for theta_max = gauss_k * sigma
    """
    i, j = sorted(proc["sites"])
    S, T = _parse_longrange_factors(proc)

    M = int(proc.get("M", gauss_M))
    theta_max = float(proc.get("theta_max", gauss_k * sigma))

    thetas_pos = np.linspace(0.0, theta_max, (M + 1) // 2)
    thetas = np.concatenate([-thetas_pos[:0:-1], thetas_pos])
    w = np.exp(-0.5 * (thetas / sigma) ** 2)
    w /= w.sum()
    w = 0.5 * (w + w[::-1])  # exact symmetry

    s_weight = float(np.sum(w * (np.sin(thetas) ** 2)))
    assert s_weight > 1e-12, "E[sin^2 θ] too small; increase sigma or theta_max/M."
    lam = gamma / s_weight

    for idx, (wk, th) in enumerate(zip(w, thetas)):
        if wk <= 0.0:
            continue
        a, b = np.cos(th), 1j * np.sin(th)
        mpo = _build_aI_plus_bP_mpo_phys(L, i, j, S, T, a, b)
        processes_out.append({
            "name": f"unitary_gauss_{idx}_" + str(proc["name"]),
            "sites": [i, j],
            "strength": lam * float(wk),
            "mpo": mpo,
            "mpo_bond_dim": 2,
        })


class NoiseModel:
    """
    NoiseModel with automatic variance-aware analog initialization.

    Pass processes with keys:
      - 'name'   : str  (e.g. 'pauli_x', 'pauli_z', 'crosstalk_zz', ...)
      - 'sites'  : list[int]  (len 1 or 2; 2-site must be adjacent unless factors/matrix provided)
      - 'strength': float  (Lindblad rate gamma)
      - 'unraveling': one of {"standard", "projector", "unitary_2pt", "unitary_gauss", "analog_auto"}
    Optional per-process overrides (rarely needed): 'theta0', 'sigma', 'M', 'theta_max'.
    """

    def __init__(
        self,
        processes: list[dict[str, Any]] | None = None,
        *,
        num_qubits: int = 0,
        # Hazard policy: relative boost with a cap (good defaults)
        hazard_gain: float = 1.0,       # Λ_target ≈ hazard_gain * sum(γ) for analog group(s)
        hazard_cap: float = 0.0,       # but not above this absolute cap per layer
        # Gaussian discretization settings
        gauss_M: int = 11,
        gauss_k: float = 4.0,           # theta_max = gauss_k * sigma
    ) -> None:
        self.processes: list[dict[str, Any]] = []
        if processes is None:
            return

        # --- group processes & compute default angles per group --------------
        # We allow three analog groups: explicit 'unitary_2pt', explicit 'unitary_gauss',
        # and 'analog_auto' which we map to either 2pt or gauss based on s*.
        groups = {"unitary_2pt": [], "unitary_gauss": [], "analog_auto": []}
        for idx, p in enumerate(processes):
            unr = str(p.get("unraveling", "standard")).lower()
            if unr in groups:
                groups[unr].append(idx)

        # Compute per-group default s*, angles from a hazard policy:
        # Λ_target = min(hazard_gain * Γ, hazard_cap), s* = Γ / Λ_target.
        group_defaults: dict[str, dict[str, float | str]] = {}
        for unr, idxs in groups.items():
            if not idxs:
                continue
            Gamma = float(sum(float(processes[i]["strength"]) for i in idxs))
            if Gamma <= 0.0:
                continue
            # choose target Λ; ensure gain >= 1
            gain = max(1.0, float(hazard_gain))
            print(f"gain: {gain}")
            Lambda_target = min(gain * Gamma, float(hazard_cap)) if hazard_cap > 0 else gain * Gamma
            print(f"Gamma: {Gamma}, Lambda_target: {Lambda_target}")
            # if Gamma << cap, Λ_target ≈ gain*Gamma; otherwise cap dominates
            # avoid s*>1 (would *reduce* hazard below physical): clamp at 1-eps
            s_star = min(Gamma / max(Lambda_target, 1e-16), 1.0 - 1e-9)
            print(f"s_star: {s_star}")
            # pick scheme for analog_auto; for explicit gauss/2pt we keep that scheme
            scheme = unr
            if unr == "analog_auto":
                scheme = "unitary_gauss" if s_star <= 0.5 else "unitary_2pt"
            # derive angles for defaults
            theta0 = None
            sigma = None
            if scheme == "unitary_gauss":
                # Gaussian needs s < 1/2
                if s_star >= 0.5:
                    s_star = 0.5 - 1e-6
                sigma = float(np.sqrt(-0.5 * np.log(1.0 - 2.0 * s_star)))
            else:
                # two-point allows s in (0,1)
                theta0 = float(np.arcsin(np.sqrt(s_star)))
            group_defaults[unr] = {"scheme": scheme, "s_star": s_star}
            if theta0 is not None:
                group_defaults[unr]["theta0"] = theta0
            if sigma is not None:
                group_defaults[unr]["sigma"] = sigma

        # --- build filled_processes ------------------------------------------
        # We expand unravelings into concrete jump operators; others are passed through.
        for original in processes:
            assert "name" in original, "Each process must have a 'name' key"
            assert "sites" in original, "Each process must have a 'sites' key"
            assert "strength" in original, "Each process must have a 'strength' key"
            assert len(original["sites"]) <= 2, "Each noise process must have at most 2 sites"

            proc = dict(original)
            name = proc["name"]
            sites = proc["sites"]
            unravel = str(proc.get("unraveling", "standard")).lower()
            gamma = float(proc["strength"])

            # normalize site ordering for 2-site
            if isinstance(sites, list) and len(sites) == 2:
                sorted_sites = sorted(sites)
                if sorted_sites != sites:
                    proc["sites"] = sorted_sites
                sites = proc["sites"]

            # Unraveling expansion
            if unravel in ("projector", "unitary_2pt", "unitary_gauss", "analog_auto"):
                if unravel == "projector":
                    # 1-site or adjacent 2-site
                    if len(sites) == 1 or abs(sites[1] - sites[0]) == 1:
                        P = get_pauli_string_matrix(proc)         # 2x2 or 4x4
                        add_projector_expansion(self.processes, proc, P, gamma)
                    else:
                        # long-range 2-site
                        add_projector_expansion_longrange_mpo(self.processes, proc, L = num_qubits, gamma=gamma)
                    continue


                # choose scheme/angles: use per-process overrides if present, else group default
                if unravel == "unitary_2pt" or (unravel == "analog_auto" and group_defaults.get("analog_auto", {}).get("scheme") == "unitary_2pt"):
                    # Check if long-range 2-site
                    if len(sites) == 2 and abs(sites[1] - sites[0]) > 1:
                        # Long-range unitary_2pt
                        theta0 = float(proc.get("theta0", group_defaults.get(unravel, group_defaults.get("analog_auto", {})).get("theta0", 0.0)))
                        if theta0 <= 0.0:
                            # fall back to group s*
                            defaults = group_defaults.get(unravel, group_defaults.get("analog_auto", {}))
                            s_use = float(defaults["s_star"])
                            theta0 = float(np.arcsin(np.sqrt(s_use)))
                        add_unitary_2pt_expansion_longrange_mpo(self.processes, proc, L=num_qubits, gamma=gamma, theta0=theta0)
                        continue
                    # 1-site or adjacent 2-site
                    P = get_pauli_string_matrix(proc)  # 2x2 or 4x4 for 1-site or adjacent 2-site
                    theta0 = float(proc.get("theta0", group_defaults.get(unravel, group_defaults.get("analog_auto", {})).get("theta0", 0.0)))
                    if theta0 <= 0.0:
                        # fall back to group s*
                        defaults = group_defaults.get(unravel, group_defaults.get("analog_auto", {}))
                        s_use = float(defaults["s_star"])
                        theta0 = float(np.arcsin(np.sqrt(s_use)))
                    add_unitary_2pt_expansion(self.processes, proc, P, gamma, theta0)
                    continue

                # Gaussian path
                if unravel == "unitary_gauss" or (unravel == "analog_auto" and group_defaults.get("analog_auto", {}).get("scheme") == "unitary_gauss"):
                    # Check if long-range 2-site
                    if len(sites) == 2 and abs(sites[1] - sites[0]) > 1:
                        # Long-range unitary_gauss
                        sigma = float(proc.get("sigma", group_defaults.get(unravel, group_defaults.get("analog_auto", {})).get("sigma", 0.0)))
                        if sigma <= 0.0:
                            defaults = group_defaults.get(unravel, group_defaults.get("analog_auto", {}))
                            s_use = float(defaults["s_star"])
                            # require s_use < 1/2
                            s_use = min(s_use, 0.5 - 1e-6)
                            sigma = float(np.sqrt(-0.5 * np.log(1.0 - 2.0 * s_use)))
                        add_unitary_gauss_expansion_longrange_mpo(self.processes, proc, L=num_qubits, gamma=gamma, 
                                                                  sigma=sigma, gauss_M=gauss_M, gauss_k=gauss_k)
                        continue
                    # 1-site or adjacent 2-site
                    P = get_pauli_string_matrix(proc)  # 2x2 or 4x4 for 1-site or adjacent 2-site
                    sigma = float(proc.get("sigma", group_defaults.get(unravel, group_defaults.get("analog_auto", {})).get("sigma", 0.0)))
                    if sigma <= 0.0:
                        defaults = group_defaults.get(unravel, group_defaults.get("analog_auto", {}))
                        s_use = float(defaults["s_star"])
                        # require s_use < 1/2
                        s_use = min(s_use, 0.5 - 1e-6)
                        sigma = float(np.sqrt(-0.5 * np.log(1.0 - 2.0 * s_use)))
                    add_unitary_gauss_expansion(self.processes, proc, P, gamma, sigma, gauss_M, gauss_k)
                    continue

                raise ValueError(f"Unhandled unraveling: {unravel!r}")

            # --- pass-through (standard) -------------------------------------
            # 2-site normalization and matrix inference for non-unraveled processes
            if isinstance(sites, list) and len(sites) == 2:
                i, j = sites
                is_adjacent = abs(j - i) == 1
                if is_adjacent:
                    if str(name).startswith("crosstalk_"):
                        suffix = str(name).rsplit("_", 1)[-1]
                        assert len(suffix) == 2 and all(c in "xyz" for c in suffix), \
                            "Invalid crosstalk label. Expected 'crosstalk_ab' with a,b in {x,y,z}."
                        a, b = suffix[0], suffix[1]
                        proc["matrix"] = np.kron(PAULI_MAP[a], PAULI_MAP[b])
                    elif "matrix" not in proc:
                        proc["matrix"] = NoiseModel.get_operator(name)
                    self.processes.append(proc)
                    continue

                # ACCEPT already-expanded long-range 2-site processes that carry an MPO
                # (e.g., projector_plus_* or projector_minus_* built by the projector expander)
                if "mpo" in proc:
                    # Sites were normalized above; just pass through the materialized MPO
                    self.processes.append(proc)
                    continue

                # non-adjacent 2-site handling (legacy labels and factors)
                if str(name).startswith("crosstalk_") or str(name).startswith(CROSSTALK_PREFIX):
                    if "factors" not in proc:
                        suffix = str(name).rsplit("_", 1)[-1]
                        assert len(suffix) == 2 and all(c in "xyz" for c in suffix), \
                            "Invalid long-range label; expected suffix ab with a,b in {x,y,z}."
                        a, b = suffix[0], suffix[1]
                        proc["factors"] = (PAULI_MAP[a], PAULI_MAP[b])
                    self.processes.append(proc)
                    continue

                # other long-range two-site must provide factors
                assert "factors" in proc, \
                    "Non-adjacent 2-site processes must specify 'factors' unless named crosstalk_{ab}."
                self.processes.append(proc)
                continue

            # 1-site: ensure matrix
            if "matrix" not in proc:
                proc["matrix"] = NoiseModel.get_operator(name)
            self.processes.append(proc)

    @staticmethod
    def get_operator(name: str) -> NDArray[Any]:
        operator_class = getattr(NoiseLibrary, name)
        return operator_class().matrix

