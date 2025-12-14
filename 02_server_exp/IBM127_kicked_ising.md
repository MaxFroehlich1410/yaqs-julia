# IBM 127q “kicked Ising” circuit in `server_exp.jl`

This repo can generate and run the **untwirled kicked-Ising** benchmark circuit described in:

- Kim *et al.*, “Evidence for the utility of quantum computing before fault tolerance”, *Nature* 618, 500–505 (2023): `https://www.nature.com/articles/s41586-023-06096-3`

## What the circuit is

Per Trotter step (a “kick”):

1. Apply \(R_X(\theta_h)\) on **all** qubits.
2. Apply \(R_{ZZ}(\theta_J)\) on **all hardware-neighbor edges**, scheduled into **edge-disjoint layers** from the backend coupling graph (heavy-hex gives 3 layers).
3. (Optional) Barrier after each step (used as a sampling delimiter in this repo).

## How it is integrated

`02_server_exp/server_exp.jl` supports the circuit name:

- `CIRCUIT_LIST = ["IBM127_kicked_ising"]`

It uses `02_server_exp/ibm_127q_circuit.py` to derive the heavy-hex edge coloring from a Qiskit backend (default `FakeKyiv`).

## Key parameters (set near the top of `server_exp.jl`)

- **Backend**: `IBM_BACKEND_NAME = "FakeKyiv"`
- **Angles**:
  - `IBM_THETA_H` (RX kick angle)
  - `IBM_THETA_J` (RZZ coupling angle; paper uses `-π/2`)
- **Sampling barriers**: `IBM_ADD_BARRIERS = true`
- **Initial state**:
  - If `IBM_INIT_X_EVERY_4 = false`: start from \(|0\rangle^{\otimes N}\)
  - If `IBM_INIT_X_EVERY_4 = true`: apply X on qubits 4, 8, 12, … (Julia 1-based)

## Recommended usage

- **Qiskit Aer (MPS) baseline**: set `RUN_QISKIT_MPS = true`.
- **Julia simulation**: for `N=127` this is generally *not practical* with a 1D MPS backend, because most heavy-hex neighbor couplings become long-range along the MPS line. Use Julia runs mainly for smaller toy systems / different circuit families.


