#!/usr/bin/env python3
"""
Exact (statevector) simulation of the same periodic Heisenberg circuit used by `create_heisenberg_circuit`
and by the other scripts in `03_Nature_review_checks/`.

Writes CSV outputs compatible with the driver:
- <tag>_obs.csv : columns step,Z_site<k>...
- <tag>_chi.csv : columns step,chi_max   (chi_max is NaN for exact sim; bond dims are TN-specific)
"""

# pylint: disable=import-error

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path


def _parse_sites(s: str, L: int) -> list[int]:
    sites = []
    for part in s.split(","):
        t = part.strip()
        if not t:
            continue
        x = int(t)
        if 1 <= x <= L:
            sites.append(x)
    if not sites:
        raise ValueError(f"No valid sites in --sites (1..{L}).")
    return sites


def _z_expectations_from_statevector(state, qubits: list[int]) -> list[float]:
    # Qiskit uses little-endian indexing for basis states: bit 0 corresponds to qubit 0.
    import numpy as np

    psi = np.asarray(state.data)
    probs = (psi.real * psi.real) + (psi.imag * psi.imag)
    idx = np.arange(probs.shape[0], dtype=np.uint64)
    out = []
    for q in qubits:
        bit = (idx >> np.uint64(q)) & np.uint64(1)
        # Z = +1 for bit=0, -1 for bit=1
        z = np.sum(probs * (1.0 - 2.0 * bit.astype(np.float64)))
        out.append(float(z))
    return out


def _build_one_trotter_layer(qc, txx, tyy, tzz, tz, periodic: bool) -> None:
    from qiskit.circuit.library import RXXGate, RYYGate, RZZGate

    L = qc.num_qubits

    # Rz on all (we use the built-in QuantumCircuit.rz)
    if tz != 0.0:
        for i in range(L):
            qc.rz(tz, i)

    def apply_even_odd_pairs(gate_ctor, theta):
        if theta == 0.0:
            return
        # pairs (0,1), (2,3), ...
        for i in range(0, L - 1, 2):
            qc.append(gate_ctor(theta), [i, i + 1])
        # pairs (1,2), (3,4), ...
        for i in range(1, L - 1, 2):
            qc.append(gate_ctor(theta), [i, i + 1])
        if periodic and L > 1:
            qc.append(gate_ctor(theta), [0, L - 1])

    apply_even_odd_pairs(RZZGate, tzz)
    apply_even_odd_pairs(RXXGate, txx)
    apply_even_odd_pairs(RYYGate, tyy)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--Jx", type=float, default=1.0)
    p.add_argument("--Jy", type=float, default=1.0)
    p.add_argument("--Jz", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--periodic", type=str, default="true")
    p.add_argument("--state", type=str, default="neel", choices=["up", "neel"])
    p.add_argument("--sites", type=str, default="1,4,8")
    p.add_argument("--outdir", type=str, default="03_Nature_review_checks/results")
    p.add_argument("--tag", type=str, default="qiskit_exact")
    args = p.parse_args()

    L = int(args.L)
    dt = float(args.dt)
    periodic = str(args.periodic).lower() in ("1", "true", "yes", "y")
    sites_1based = _parse_sites(args.sites, L)
    qubits = [s - 1 for s in sites_1based]  # 0-based

    # Match `create_heisenberg_circuit` angles (note: that circuit uses t = -2*dt*coupling).
    txx = -2.0 * dt * float(args.Jx)
    tyy = -2.0 * dt * float(args.Jy)
    tzz = -2.0 * dt * float(args.Jz)
    tz = -2.0 * dt * float(args.h)

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    # initial circuit (only for initialization)
    init = QuantumCircuit(L)
    if args.state == "neel":
        # Flip even 1-based sites => odd 0-based indices: 1,3,5,...
        for q in range(1, L, 2):
            init.x(q)
    sv = Statevector.from_instruction(init)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    obs_path = outdir / f"{args.tag}_obs.csv"
    chi_path = outdir / f"{args.tag}_chi.csv"
    timing_path = outdir / f"{args.tag}_timing.csv"

    # Buffer results; write CSVs after timing to avoid I/O in the timed section.
    obs_rows: list[list[float]] = [[0.0, *_z_expectations_from_statevector(sv, qubits)]]
    chi_rows: list[list[float]] = [[0.0, math.nan]]

    t0 = time.time()
    for n in range(int(args.steps)):
        layer = QuantumCircuit(L)
        _build_one_trotter_layer(layer, txx, tyy, tzz, tz, periodic=periodic)
        sv = sv.evolve(layer)

        step = n + 1
        obs_rows.append([float(step), *_z_expectations_from_statevector(sv, qubits)])
        chi_rows.append([float(step), math.nan])
    wall = time.time() - t0

    # Write outputs (not timed)
    with obs_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", *[f"Z_site{s}" for s in sites_1based]])
        w.writerows(obs_rows)

    with chi_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "chi_max"])
        w.writerows(chi_rows)

    with timing_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_s_sim"])
        w.writerow([wall])

    print(f"[qiskit_exact] wrote: {obs_path}")
    print(f"[qiskit_exact] wrote: {chi_path}")
    print(f"[qiskit_exact] wall_s_sim={wall:.3f}")
    print(f"[qiskit_exact] wrote: {timing_path}")


if __name__ == "__main__":
    main()

