# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the NoiseModel class.

This module provides unit tests for the NoiseModel class.
It verifies that a NoiseModel is created correctly when valid processes and strengths are provided,
raises an AssertionError when the lengths of the processes and strengths lists differ,
and handles empty noise models appropriately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.libraries.noise_library import PauliX, PauliY, PauliZ


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(a, b, atol=1e-12)


def test_noise_model_creation() -> None:
    """Test that NoiseModel is created correctly with valid process dicts.

    This test constructs a NoiseModel with two single-site processes
    ("lowering" and "pauli_z") and corresponding strengths.
    It verifies that:
      - Each process is stored as a dictionary with correct fields.
      - The number of processes is correct.
      - Each process contains a jump_operator with the expected shape (2x2).
    """
    processes: list[dict[str, Any]] = [
        {"name": "lowering", "sites": [0], "strength": 0.1},
        {"name": "pauli_z", "sites": [1], "strength": 0.05},
    ]

    model = NoiseModel(processes)

    assert len(model.processes) == 2
    assert model.processes[0]["name"] == "lowering"
    assert model.processes[1]["name"] == "pauli_z"
    assert model.processes[0]["strength"] == 0.1
    assert model.processes[1]["strength"] == 0.05
    assert model.processes[0]["sites"] == [0]
    assert model.processes[1]["sites"] == [1]
    assert model.processes[0]["matrix"].shape == (2, 2)
    assert model.processes[1]["matrix"].shape == (2, 2)


def test_noise_model_assertion() -> None:
    """Test that NoiseModel raises an AssertionError when a process dict is missing required fields.

    This test constructs a process list where one entry is missing the 'strength' field,
    which should cause the NoiseModel initialization to fail.
    """
    # Missing 'strength' in the second dict
    processes: list[dict[str, Any]] = [
        {"name": "lowering", "sites": [0], "strength": 0.1},
        {"name": "pauli_z", "sites": [1]},  # Missing strength
    ]

    with pytest.raises(AssertionError):
        _ = NoiseModel(processes)


def test_noise_model_empty() -> None:
    """Test that NoiseModel handles an empty list of processes without error.

    This test initializes a NoiseModel with an empty list of process dictionaries and verifies that the resulting
    model has empty `processes` and `jump_operators` lists.
    """
    model = NoiseModel()

    assert model.processes == []


def test_noise_model_none() -> None:
    """Test that NoiseModel handles a None input without error.

    This test initializes a NoiseModel with `None` and verifies that the resulting
    model has no processes.
    """
    model = NoiseModel(None)

    assert model.processes == []


def test_one_site_matrix_auto() -> None:
    """Test that one-site processes auto-fill a 2x2 'matrix'.

    This verifies that providing name/sites/strength for a single-site process
    produces a process with a 2x2 operator populated from the library.
    """
    nm = NoiseModel([{"name": "pauli_x", "sites": [1], "strength": 0.1}])
    assert len(nm.processes) == 1
    p = nm.processes[0]
    assert "matrix" in p, "1-site process should have matrix auto-filled"
    assert p["matrix"].shape == (2, 2)
    assert _allclose(p["matrix"], PauliX.matrix)


def test_adjacent_two_site_matrix_auto() -> None:
    """Test that adjacent two-site processes auto-fill a 4x4 'matrix'.

    This checks that nearest-neighbor crosstalk uses the library matrix (kron)
    and requires no explicit operator in the process dict.
    """
    nm = NoiseModel([{"name": "crosstalk_xz", "sites": [1, 2], "strength": 0.2}])
    p = nm.processes[0]
    assert "matrix" in p, "Adjacent 2-site process should have matrix auto-filled"
    assert p["matrix"].shape == (4, 4)
    expected = np.kron(PauliX.matrix, PauliZ.matrix)
    assert _allclose(p["matrix"], expected)


def test_longrange_two_site_factors_auto() -> None:
    """Test that long-range two-site processes auto-fill 'factors' only.

    Using the canonical 'longrange_crosstalk_{ab}' name, the model should attach
    per-site 2x2 factors (A,B) and omit any large Kronecker 'matrix'.
    """
    nm = NoiseModel([{"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.3}])
    p = nm.processes[0]
    assert "factors" in p, "Long-range 2-site process should have factors auto-filled"
    a_op, b_op = p["factors"]
    assert a_op.shape == (2, 2)
    assert b_op.shape == (2, 2)
    assert _allclose(a_op, PauliX.matrix)
    assert _allclose(b_op, PauliY.matrix)
    assert "matrix" not in p, "Long-range processes should not attach a full matrix"


def test_longrange_two_site_factors_explicit() -> None:
    """Test that explicit 'factors' for long-range are accepted and sites normalize.

    Supplying (A,B) and unsorted endpoints should result in stored ascending sites,
    preserving factors and omitting a full 'matrix'.
    """
    nm = NoiseModel([
        {
            "name": "custom_longrange_xy",
            "sites": [3, 1],  # intentionally unsorted
            "strength": 0.3,
            "factors": (PauliX.matrix, PauliY.matrix),
        }
    ])
    p = nm.processes[0]
    # Sites must be normalized to ascending order
    assert p["sites"] == [1, 3]
    assert "factors" in p
    assert len(p["factors"]) == 2
    a_op, b_op = p["factors"]
    assert _allclose(a_op, PauliX.matrix)
    assert _allclose(b_op, PauliY.matrix)
    assert "matrix" not in p


def test_longrange_unknown_label_without_factors_raises() -> None:
    """Test that unknown long-range labels without 'factors' raise.

    If the name is not 'longrange_crosstalk_{ab}' and no factors are provided,
    initialization must fail to avoid guessing operators.

    Raises:
        AssertionError: If the model accepts an unknown long-range label without factors.
    """
    try:
        # Name is not a recognized non-adjacent 'crosstalk_{ab}' and no factors provided
        _ = NoiseModel([{"name": "foo_bar", "sites": [0, 2], "strength": 0.1}])
    except AssertionError:
        return
    msg = "Expected AssertionError for unknown long-range label without factors."
    raise AssertionError(msg)


def test_unraveling_projector_one_site() -> None:
    gamma = 0.2
    nm = NoiseModel([
        {"name": "pauli_z", "sites": [0], "strength": gamma, "unraveling": "projector"}
    ])
    assert len(nm.processes) == 2
    mats = [p["matrix"] for p in nm.processes]
    strs = [p["strength"] for p in nm.processes]
    I = np.eye(2)
    Z = PauliZ.matrix
    # Order-independent check
    assert any(np.allclose(m, I + Z) for m in mats)
    assert any(np.allclose(m, I - Z) for m in mats)
    assert all(np.isclose(s, gamma / 2.0) for s in strs)


def test_unraveling_unitary_2pt_one_site() -> None:
    gamma = 0.3
    theta0 = 0.2
    nm = NoiseModel([
        {"name": "pauli_x", "sites": [1], "strength": gamma, "unraveling": "unitary_2pt", "theta0": theta0}
    ])
    assert len(nm.processes) == 2
    strs = [p["strength"] for p in nm.processes]
    lam = gamma / (np.sin(theta0) ** 2)
    assert all(np.isclose(s, lam / 2.0) for s in strs)
    # unitarity check: U^\dagger U = I
    for p in nm.processes:
        U = p["matrix"]
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-10)


def test_unraveling_unitary_gauss_one_site_strengths_sum() -> None:
    gamma = 0.15
    sigma = 0.25
    M = 21
    nm = NoiseModel([
        {"name": "pauli_y", "sites": [0], "strength": gamma, "unraveling": "unitary_gauss", "sigma": sigma, "M": M}
    ])
    assert len(nm.processes) == M  # symmetric construction includes all grid points
    strengths_sum = sum(p["strength"] for p in nm.processes)
    # expected sum is lambda = gamma / E[sin^2(theta)] computed inside the constructor;
    # since we don't expose it, verify all matrices are unitary and strengths positive
    assert strengths_sum > 0
    for p in nm.processes:
        U = p["matrix"]
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-10)
        assert p["strength"] >= 0


def test_unraveling_projector_two_site_adjacent() -> None:
    gamma = 0.12
    nm = NoiseModel([
        {"name": "crosstalk_xy", "sites": [2, 3], "strength": gamma, "unraveling": "projector"}
    ])
    assert len(nm.processes) == 2
    mats = [p["matrix"] for p in nm.processes]
    I4 = np.eye(4)
    X = PauliX.matrix
    Y = PauliY.matrix
    P = np.kron(X, Y)
    assert any(np.allclose(m, I4 + P) for m in mats)
    assert any(np.allclose(m, I4 - P) for m in mats)
    assert all(np.isclose(p["strength"], gamma / 2.0) for p in nm.processes)


def test_unraveling_projector_two_site_longrange() -> None:
    """Test long-range projector unraveling produces MPO-based processes.

    For non-adjacent two-site processes, projector unraveling should generate
    two processes with MPO representations (I ± P) instead of dense matrices.
    """
    gamma = 0.08
    num_qubits = 5
    nm = NoiseModel([
        {"name": "crosstalk_zz", "sites": [0, 3], "strength": gamma, "unraveling": "projector"}
    ], num_qubits=num_qubits)

    # Should produce two processes: projector_plus and projector_minus
    assert len(nm.processes) == 2
    names = [p["name"] for p in nm.processes]
    assert any("projector_plus" in name for name in names)
    assert any("projector_minus" in name for name in names)

    # Each process should have MPO representation, not dense matrix
    for p in nm.processes:
        assert "mpo" in p, "Long-range projector should have MPO representation"
        assert "mpo_bond_dim" in p
        assert p["mpo_bond_dim"] == 2, "Bond dimension should be 2 for (I ± P) structure"
        assert "matrix" not in p, "Long-range projector should not have dense matrix"
        assert p["sites"] == [0, 3]
        assert np.isclose(p["strength"], gamma / 2.0)


def test_unraveling_unitary_2pt_two_site_longrange() -> None:
    """Test long-range unitary_2pt unraveling produces MPO-based processes.
    
    For non-adjacent two-site processes, unitary_2pt unraveling should generate
    two processes with bond-2 MPO representations for U_± = exp(±i*θ₀*P).
    """
    gamma = 0.05
    theta0 = 0.3
    num_qubits = 6
    nm = NoiseModel([
        {"name": "crosstalk_xy", "sites": [1, 4], "strength": gamma, "unraveling": "unitary_2pt", "theta0": theta0}
    ], num_qubits=num_qubits)
    
    # Should produce two processes: unitary2pt_plus and unitary2pt_minus
    assert len(nm.processes) == 2
    names = [p["name"] for p in nm.processes]
    assert any("unitary2pt_plus" in name for name in names)
    assert any("unitary2pt_minus" in name for name in names)
    
    # Verify strength calculation: λ = γ / sin²(θ₀), each component has λ/2
    s_val = np.sin(theta0) ** 2
    expected_lam = gamma / s_val
    expected_strength = expected_lam / 2.0
    
    # Each process should have MPO representation, not dense matrix
    for p in nm.processes:
        assert "mpo" in p, "Long-range unitary_2pt should have MPO representation"
        assert "mpo_bond_dim" in p
        assert p["mpo_bond_dim"] == 2, "Bond dimension should be 2 for exp(i*θ*P) structure"
        assert "matrix" not in p, "Long-range unitary_2pt should not have dense matrix"
        assert p["sites"] == [1, 4]
        assert np.isclose(p["strength"], expected_strength), f"Expected {expected_strength}, got {p['strength']}"


def test_unraveling_unitary_gauss_two_site_longrange() -> None:
    """Test long-range unitary_gauss unraveling produces MPO-based processes.
    
    For non-adjacent two-site processes, unitary_gauss unraveling should generate
    M processes with bond-2 MPO representations for U(θₖ) with Gaussian weights.
    """
    gamma = 0.06
    sigma = 0.25
    M = 11
    num_qubits = 5
    nm = NoiseModel([
        {"name": "longrange_crosstalk_zx", "sites": [0, 3], "strength": gamma, 
         "unraveling": "unitary_gauss", "sigma": sigma, "M": M}
    ], num_qubits=num_qubits)
    
    # Should produce M processes
    assert len(nm.processes) == M, f"Expected {M} processes, got {len(nm.processes)}"
    
    # Verify all have unitary_gauss_ prefix
    names = [p["name"] for p in nm.processes]
    assert all("unitary_gauss_" in name for name in names)
    
    # Verify strength sum: λ = γ / E[sin²(θ)]
    total_strength = sum(p["strength"] for p in nm.processes)
    
    # Reconstruct the expected λ
    gauss_k = 4.0  # default
    theta_max = gauss_k * sigma
    thetas_pos = np.linspace(0.0, theta_max, (M + 1) // 2)
    thetas = np.concatenate([-thetas_pos[:0:-1], thetas_pos])
    w = np.exp(-0.5 * (thetas / sigma) ** 2)
    w /= w.sum()
    w = 0.5 * (w + w[::-1])
    s_weight = float(np.sum(w * (np.sin(thetas) ** 2)))
    expected_lam = gamma / s_weight
    
    assert np.isclose(total_strength, expected_lam, rtol=1e-6), \
        f"Sum of strengths {total_strength} should equal λ = {expected_lam}"
    
    # Each process should have MPO representation, not dense matrix
    for p in nm.processes:
        assert "mpo" in p, "Long-range unitary_gauss should have MPO representation"
        assert "mpo_bond_dim" in p
        assert p["mpo_bond_dim"] == 2, "Bond dimension should be 2 for exp(i*θ*P) structure"
        assert "matrix" not in p, "Long-range unitary_gauss should not have dense matrix"
        assert p["sites"] == [0, 3]
        assert p["strength"] >= 0, "All weights should be non-negative"
