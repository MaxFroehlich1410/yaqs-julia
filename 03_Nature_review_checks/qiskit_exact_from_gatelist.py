#!/usr/bin/env python3
"""
Exact (statevector) simulation driven by a gate-list exported from Julia `CircuitLibrary.jl`.

Measurement points are defined by barriers with label "SAMPLE_OBSERVABLES".
Outputs are compatible with the Julia comparison driver:
- <tag>_obs.csv : step,Z_site<k>...
- <tag>_chi.csv : step,chi_max   (NaN)
- <tag>_timing.csv : wall_s_sim  (simulation loop only)
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from qiskit_gatelist import build_qiskit_circuit_from_gatelist, is_sample_barrier, read_gatelist_csv


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
        z = np.sum(probs * (1.0 - 2.0 * bit.astype(np.float64)))
        out.append(float(z))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gatelist", type=str, required=True)
    p.add_argument("--state", type=str, default="neel", choices=["up", "neel"])
    p.add_argument("--sites", type=str, default="1")
    p.add_argument("--outdir", type=str, default="03_Nature_review_checks/results")
    p.add_argument("--tag", type=str, default="qiskit_exact")
    args = p.parse_args()

    gatelist = read_gatelist_csv(args.gatelist)
    qc = build_qiskit_circuit_from_gatelist(gatelist)
    L = int(qc.num_qubits)

    sites_1based = _parse_sites(args.sites, L)
    qubits = [s - 1 for s in sites_1based]

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    init = QuantumCircuit(L)
    if args.state == "neel":
        for q in range(1, L, 2):
            init.x(q)
    sv = Statevector.from_instruction(init)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    obs_path = outdir / f"{args.tag}_obs.csv"
    chi_path = outdir / f"{args.tag}_chi.csv"
    timing_path = outdir / f"{args.tag}_timing.csv"

    obs_rows: list[list[float]] = []
    chi_rows: list[list[float]] = []
    step = -1

    t0 = time.time()
    for inst in qc.data:
        op = inst.operation
        if is_sample_barrier(op):
            step += 1
            obs_rows.append([float(step), *_z_expectations_from_statevector(sv, qubits)])
            chi_rows.append([float(step), math.nan])
            continue

        qidx = [qc.find_bit(q).index for q in inst.qubits]
        one = QuantumCircuit(L)
        one.append(op, qidx)
        sv = sv.evolve(one)
    wall = time.time() - t0

    if step < 0:
        step = 0
        obs_rows.append([0.0, *_z_expectations_from_statevector(sv, qubits)])
        chi_rows.append([0.0, math.nan])

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

    print(f"[qiskit_exact] L={L} meas_points={step+1} wall_s_sim={wall:.3f}")
    print(f"[qiskit_exact] wrote: {obs_path}")
    print(f"[qiskit_exact] wrote: {chi_path}")
    print(f"[qiskit_exact] wrote: {timing_path}")


if __name__ == "__main__":
    main()

