#!/usr/bin/env python3
"""
Gate-list loader used by `03_Nature_review_checks/` drivers.

Reads the CSV written by `gatelist_io.jl` and builds a Qiskit `QuantumCircuit`.
This makes Qiskit the "canonical instruction stream" for:
- Qiskit exact evolution
- TenPy MPO construction/application (zip-up or variational)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateRec:
    op: str
    qubits: tuple[int, ...]  # 0-based
    params: tuple[float, ...]
    label: str


def _parse_qubits(field: str) -> tuple[int, ...]:
    t = field.strip()
    if not t:
        return ()
    # CSV uses 1-based indices, separated by ';'
    return tuple(int(x) - 1 for x in t.split(";") if x.strip())


def _parse_params(field: str) -> tuple[float, ...]:
    t = field.strip()
    if not t:
        return ()
    return tuple(float(x) for x in t.split(";") if x.strip())


def read_gatelist_csv(path: str | Path) -> list[GateRec]:
    p = Path(path)
    rows: list[GateRec] = []
    with p.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if header[:4] != ["op", "qubits", "params", "label"]:
            raise ValueError(f"Unexpected gate-list header in {p}: {header}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            op, qubits_s, params_s, label = (line.split(",", maxsplit=3) + ["", "", "", ""])[:4]
            rows.append(
                GateRec(
                    op=op.strip().lower(),
                    qubits=_parse_qubits(qubits_s),
                    params=_parse_params(params_s),
                    label=label.strip(),
                )
            )
    return rows


def infer_num_qubits(gates: list[GateRec]) -> int:
    m = -1
    for g in gates:
        for q in g.qubits:
            m = max(m, q)
    return m + 1


def build_qiskit_circuit_from_gatelist(gates: list[GateRec], *, num_qubits: int | None = None):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RXXGate, RYYGate, RZZGate

    n = infer_num_qubits(gates) if num_qubits is None else int(num_qubits)
    qc = QuantumCircuit(n)

    for g in gates:
        op = g.op
        qs = list(g.qubits)
        ps = list(g.params)

        if op == "barrier":
            # marker barrier: apply on all qubits so it survives scheduling/transpilation
            qc.barrier(*range(n), label=g.label or None)
            continue

        if op in ("id", "x", "y", "z", "h", "s", "t", "sdg", "tdg", "rx", "ry", "rz", "p", "u"):
            if len(qs) != 1:
                raise ValueError(f"{op} expects 1 qubit, got {qs}")
            q = qs[0]
            if op == "id":
                qc.id(q)
            elif op == "x":
                qc.x(q)
            elif op == "y":
                qc.y(q)
            elif op == "z":
                qc.z(q)
            elif op == "h":
                qc.h(q)
            elif op == "s":
                qc.s(q)
            elif op == "t":
                qc.t(q)
            elif op == "sdg":
                qc.sdg(q)
            elif op == "tdg":
                qc.tdg(q)
            elif op == "rx":
                qc.rx(ps[0], q)
            elif op == "ry":
                qc.ry(ps[0], q)
            elif op == "rz":
                qc.rz(ps[0], q)
            elif op == "p":
                qc.p(ps[0], q)
            elif op == "u":
                qc.u(ps[0], ps[1], ps[2], q)
            continue

        if op in ("cx", "cy", "cz", "ch", "cp", "swap", "iswap", "rxx", "ryy", "rzz"):
            if len(qs) != 2:
                raise ValueError(f"{op} expects 2 qubits, got {qs}")
            a, b = qs[0], qs[1]
            if op == "cx":
                qc.cx(a, b)
            elif op == "cy":
                qc.cy(a, b)
            elif op == "cz":
                qc.cz(a, b)
            elif op == "ch":
                qc.ch(a, b)
            elif op == "cp":
                qc.cp(ps[0], a, b)
            elif op == "swap":
                qc.swap(a, b)
            elif op == "iswap":
                qc.iswap(a, b)
            elif op == "rxx":
                qc.append(RXXGate(ps[0]), [a, b])
            elif op == "ryy":
                qc.append(RYYGate(ps[0]), [a, b])
            elif op == "rzz":
                qc.append(RZZGate(ps[0]), [a, b])
            continue

        raise ValueError(f"Unsupported op in gate-list: {op}")

    return qc


def is_sample_barrier(op) -> bool:
    # op is a Qiskit instruction.operation
    try:
        name = str(op.name).lower()
    except Exception:
        return False
    if name != "barrier":
        return False
    label = getattr(op, "label", None)
    if label is None:
        return False
    return str(label).strip().upper() == "SAMPLE_OBSERVABLES"

