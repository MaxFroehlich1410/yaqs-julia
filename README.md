# Yaqs.jl

Yaqs.jl is a Julia toolkit for simulating quantum many-body dynamics, noisy and ideal quantum circuits with tensor networks.
It provides Matrix Product State (MPS) and Matrix Product Operator (MPO) backends, time
evolution with TDVP/TEBD-style updates, and digital-circuit simulations with noise and
sampling workflows. The code targets large-scale, high-performance simulations while
keeping the APIs explicit and composable. It contains the Tensor Jump Method (TJM) for open quantum systems, the local TDVP algorithm for ideal circuits and the circuitTJM for noisy circuits.

## Features

- MPS/MPO core with canonicalization, truncation, and observables
- TDVP-based time evolution for analog Hamiltonians
- Circuit simulator with gate libraries and circuit layering
- Noise + dissipation models and stochastic trajectory sampling
- Optional Qiskit ingestion via PythonCall for circuit import
- Threaded trajectory execution for multi-core runs

## Installation

This is a standard Julia project:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

If you plan to ingest Qiskit circuits, install the Python dependencies as well:

```bash
julia --project -e 'using CondaPkg; CondaPkg.instantiate()'
```

## Quick Start (Digital Circuit)

```julia
using Yaqs
using Yaqs.MPSModule
using Yaqs.GateLibrary
using Yaqs.SimulationConfigs
using Yaqs.CircuitLibrary
using Yaqs.Simulator

L = 6
psi = MPSModule.MPS(L, state="zeros")

obs = SimulationConfigs.Observable("Z1", GateLibrary.ZGate(), 1)
cfg = SimulationConfigs.TimeEvolutionConfig([obs], 1.0; dt=0.1, num_traj=10)

circ = CircuitLibrary.create_ising_circuit(L, 1.0, 1.0, 0.1, 10)
Simulator.run(psi, circ, cfg)

println(obs.results)
```

## Qiskit Circuit Ingestion

```julia
using PythonCall
using Yaqs
using Yaqs.CircuitIngestion

qiskit = pyimport("qiskit")
qc = qiskit.QuantumCircuit(4)
qc.h(0); qc.cx(0, 1)

circ = CircuitIngestion.ingest_qiskit_circuit(qc)
```

## Tests

```bash
julia --project run_tests.jl
```

## Repository Layout

- `src/`: core MPS/MPO algorithms, TDVP, noise, and circuit simulation
- `test/`: unit tests
- `01_PaperExps/`: paper and experiment scripts
- `02_server_exp/`: large-scale and server runs

## Notes

- Most public APIs live under the `Yaqs` module; submodules are re-exported.
- For large digital circuits, see `DigitalTJM.RepeatedDigitalCircuit` to avoid
  materializing repeated gate lists.
