# Experiment Configuration: Unraveling Efficiency

**Experiment Name:** `unraveling_eff_N25_L20_tau0p1_allnoise_basisXY_obsZ_locTDVP_lrTDVP_ErrX_traj200`

This document outlines the configuration parameters derived from `server_exp.jl` for reproducibility.

## 1. System Parameters
* **System Size (N):** 25 Qubits
* **Depth (L):** 20 Layers
* **Time Step ($\tau$):** 0.1
* **Hamiltonian / Circuit Model:** XY Model (Trotterized)
  * *Note:* Based on `CIRCUIT_LIST = ["XY"]`.

## 2. Simulation & Algorithm Settings
* **Trajectories:** 200
* **Max Bond Dimension:** 128
* **SVD Truncation Threshold:** $1 \times 10^{-16}$
* **Algorithm Methods:**
  * **Local Updates:** TDVP (`local_mode = "TDVP"`)
  * **Long-Range Updates:** TDVP (`longrange_mode = "TDVP"`)

## 3. Noise Model
* **Noise Strengths:** 0.1, 0.01, 0.001
* **Error Types:**
  * **Single Qubit:** Pauli-$X$ errors on all qubits.
  * **Two Qubit Crosstalk:** $XX$ errors on nearest-neighbor pairs ($i, i+1$).
  * *Configuration:* `ENABLE_X_ERROR = true`, `ENABLE_Y_ERROR = false`, `ENABLE_Z_ERROR = false`.

## 4. Initial State
* **Base State:** All-zeros product state $|0\rangle^{\otimes N}$.
* **Preparation:** Pauli-$X$ gate applied to every 4th qubit (indices 3, 7, 11... in 0-indexed notation; 4, 8, 12... in 1-indexed notation).
  * **Pattern:** $|000\mathbf{1}000\mathbf{1}000\mathbf{1}\dots\rangle$

## 5. Observables
* **Basis:** Z
* **Quantities:**
  * Local Expectation Values: $\langle Z_i \rangle$
  * Staggered Magnetization: $M_{stagg} = \frac{1}{N} \sum_{i=1}^N (-1)^{i-1} \langle Z_i \rangle$

## 6. Execution Flags (Current Snapshot)
* `RUN_QISKIT_MPS`: `true`
* `RUN_JULIA`: `false` (Currently disabled in script)
* `RUN_JULIA_ANALOG_2PT`: `false`
* `RUN_JULIA_ANALOG_GAUSS`: `false`
* `RUN_JULIA_PROJECTOR`: `false`

