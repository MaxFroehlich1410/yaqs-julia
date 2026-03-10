# Yaqs.jl

**Yaqs.jl** is a Julia toolkit for simulating quantum many-body dynamics, noisy and ideal quantum circuits using tensor networks.
It provides Matrix Product State (MPS) and Matrix Product Operator (MPO) backends, time evolution via TDVP, digital circuit simulation, and stochastic noise sampling.
The package implements the **Tensor Jump Method (TJM)** for open quantum systems, a local TDVP algorithm for ideal circuits, and the **CircuitTJM** for noisy digital circuits.

This repository also contains all experiments from the paper:

> **Noisy quantum circuit simulation with the tensor jump method**

The experiment code and interactive walkthroughs are located in the [`PaperExps/`](PaperExps/) directory.

## Features

- MPS/MPO core with canonicalization, truncation, and local/global observables
- TDVP-based time evolution for analog Hamiltonians
- Digital circuit simulator with a comprehensive gate library and circuit layering
- Noise and dissipation models with stochastic trajectory sampling (Tensor Jump Method)
- Built-in circuit library: Ising, Heisenberg, XY, QAOA, HEA, Fermi-Hubbard, and more
- Optional Qiskit circuit ingestion via PythonCall
- Threaded trajectory execution for multi-core runs

## Installation

Clone the repository and instantiate the Julia environment:

```bash
git clone https://github.com/MaxFroehlich1410/yaqs-julia.git
cd yaqs-julia
julia --project -e 'using Pkg; Pkg.instantiate()'
```

If you plan to use Qiskit circuit ingestion, also set up the Python environment:

```bash
julia --project -e 'using CondaPkg; CondaPkg.instantiate()'
```

## Quick Start

```julia
using Yaqs
using Yaqs.MPSModule
using Yaqs.GateLibrary
using Yaqs.SimulationConfigs
using Yaqs.CircuitLibrary
using Yaqs.Simulator

L = 6
psi = MPS(L, state="zeros")

obs = Observable("Z1", ZGate(), 1)
cfg = TimeEvolutionConfig([obs], 1.0; dt=0.1, num_traj=10)

circ = create_ising_circuit(L, 1.0, 1.0, 0.1, 10)
Simulator.run(psi, circ, cfg)

println(obs.results)
```

## Paper Experiments

The `PaperExps/` directory contains all experiments from the paper, organized into scripts and interactive Pluto notebooks.

### Scripts

| Script | Description |
|--------|-------------|
| `scripts/2sites_variance_exp_circuit_tjm.jl` | 2-qubit variance experiment comparing trajectory variance across different unravelings |
| `scripts/25q_exp.jl` | 25-site XY quench experiment with configurable circuits and noise |
| `scripts/ibm127_exp.jl` | IBM 127-qubit kicked-Ising experiment with long-range crosstalk |

### Pluto Notebooks

Interactive walkthroughs of the paper experiments are provided as [Pluto](https://plutojl.org/) notebooks:

| Notebook | Description |
|----------|-------------|
| `notebooks/2sites_variance_exp_circuit_tjm_pluto.jl` | Interactive 2-qubit variance experiment |
| `notebooks/25q_exp_walkthrough_pluto.jl` | Walkthrough of the 25-site XY quench experiment |
| `notebooks/ibm127_exp_walkthrough_pluto.jl` | Walkthrough of the IBM 127-qubit kicked-Ising experiment |

### Running the Pluto Notebooks

1. Start Julia from the repository root:

```bash
julia --project
```

2. Launch Pluto and open a notebook:

```julia
using Pluto
Pluto.run(notebook="PaperExps/notebooks/25q_exp_walkthrough_pluto.jl")
```

This will open the notebook in your browser. The notebooks activate the repository environment automatically, so all dependencies are available.

## Qiskit Circuit Ingestion

You can convert Qiskit `QuantumCircuit` objects directly into Yaqs circuits:

```julia
using PythonCall
using Yaqs
using Yaqs.CircuitIngestion

qiskit = pyimport("qiskit")
qc = qiskit.QuantumCircuit(4)
qc.h(0); qc.cx(0, 1)

circ = ingest_qiskit_circuit(qc)
```

## Tests

```bash
julia --project run_tests.jl
```

## Repository Layout

| Directory | Contents |
|-----------|----------|
| `src/` | Core MPS/MPO algorithms, TDVP, noise models, gate library, and circuit simulation |
| `PaperExps/scripts/` | Experiment scripts from the paper |
| `PaperExps/notebooks/` | Interactive Pluto notebook walkthroughs |
| `test/` | Unit tests and full algorithm validation checks |

## License

See the repository for license details.
