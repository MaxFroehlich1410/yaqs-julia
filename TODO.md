Based on the analysis of the Python repository and our discussions on efficiency, here is a detailed TODO list to guide the implementation of your high-performance Julia library.

This roadmap prioritizes building the core math primitives first, then the data structures, and finally the complex simulation algorithms, ensuring that the "efficiency fixes" (like type dispatch and in-place operations) are baked in from the start.


IMPORTANT: After you added a new function immediately add a test file to the test folder that checks if the new funciton works as it should. 

### Julia Implementation Roadmap

#### Phase 1: Core Primitives & Gate Architecture

*Goal: Establish the low-level mathematical foundation using `StaticArrays` and Type Dispatch to replace the "GateLibrary" anti-pattern.*

1. **Implement Abstract Operator Hierarchy**

   * Define `abstract type AbstractOperator end`.
   * Implement singleton structs for standard gates: `XGate`, `YGate`, `ZGate`, `HGate`.
   * Implement parameterized structs: `RxGate{T}`, `RyGate{T}`, `RzGate{T}`.
   * **Critical:** Ensure these are immutable bitstypes.
2. **Implement Matrix Dispatch**

   * Create a `matrix(op::AbstractOperator)` function.
   * Use `StaticArrays.SMatrix{2,2,ComplexF64}` for the return types to avoid allocations.
   * *Why:* This replaces the `GateLibrary.py` factory pattern with zero-cost abstractions.
3. **Implement Tensor Primitives (Decompositions)**

   * Implement `right_qr` and `left_qr` using `LinearAlgebra.qr`.
   * Implement `two_site_svd` with truncation logic.
   * **Critical:** Ensure `svd` handles the "fallback" strategies found in the Python repo (regularization for stability) if needed, but standard LAPACK `svd!` is usually robust. Use `LinearAlgebra.BLAS` calls where appropriate.

#### Phase 2: Network Data Structures (MPS/MPO)

*Goal: Create the container classes that hold the state, optimizing for memory layout.*

4. **Implement `MPS` Struct**

   * Define `struct MPS{T} tensors::Vector{Array{T,3}} ... end`.
   * Implement initialization methods: `randomMPS`, `productMPS` (from bitstrings).
   * Implement `norm(psi::MPS)` and canonical form checks.
   * *Optimization:* Implement `move_orthogonality_center!(psi, site)` which modifies the tensors in-place.
5. **Implement `MPO` Struct**

   * Define `struct MPO{T} tensors::Vector{Array{T,4}} end`.
   * Implement MPO generators for Hamiltonians (Ising, Heisenberg) ensuring the correct index order `(phys_out, phys_in, left, right)`.
   * Use `@tensor` macro from `TensorOperations.jl` for all contraction definitions to ensure efficient index permutation.

#### Phase 3: Simulation Parameters & Observables

*Goal: Decouple parameters from logic and implement the efficient Observable pattern.*

6. **Implement `Observable` Struct**

   * Define `struct Observable{O<:AbstractOperator} op::O; site::Int end`.
   * Implement `expect(psi::MPS, obs::Observable)` using local contractions only.
   * *Optimization:* For `local_expect`, ensure it contracts *only* the relevant tensors and doesn't copy the whole MPS.
7. **Implement Simulation Config Structs**

   * Create `struct TimeEvolutionConfig` (replaces `AnalogSimParams`).
   * Fields: `dt`, `total_time`, `max_bond_dim`, `truncation_threshold`.
   * Create `struct MeasurementConfig` (replaces `WeakSimParams`).

#### Phase 4: Time Evolution Algorithms (The "Hot Paths")

*Goal: Implement the TDVP and matrix exponential logic, fixing the allocation issues.*

8. **Implement Matrix-Free Exponentials (Krylov)**

   * Implement a `lanczos!` function that accepts a pre-allocated Krylov subspace buffer.
   * Implement `expm_krylov!` that uses this buffer to compute $e^{-iHt}|v\rangle$.
   * *Optimization:* Use `ExponentialUtilities.jl` if possible, or your own allocation-free implementation.
9. **Implement `TDVP` Sweeps**

   * Implement `single_site_tdvp!` and `two_site_tdvp!`.
   * **Critical:** Ensure these functions modify the `MPS` in-place.
   * Implement the "effective Hamiltonian" contractions (environment updates) using `@tensor`.

#### Phase 5: Noise & Dissipation (The "Efficiency Fix")

*Goal: Rewrite the stochastic process logic to avoid the `deepcopy` bottleneck.*

10. **Implement Noise Models via Dispatch**

    * Define `abstract type AbstractNoiseChannel end`.
    * Implement `PauliChannel <: AbstractNoiseChannel` and `DissipativeChannel`.
11. **Implement Efficient Jump Probability**

    * Create function `calc_jump_prob(psi::MPS, op::AbstractOperator, site::Int)`.
    * **Critical:** This function MUST compute the norm of the jumped state locally (contracting only `L[site] * op * A[site] * R[site]`) without copying the full MPS.
12. **Implement `stochastic_sweep!`**

    * Combine the probability calculation and the jump application into a single pass.
    * Replace the string-based logic (`if "pauli" in name`) with multiple dispatch on the `NoiseChannel` types.

#### Phase 6: Validation & Benchmarking

13. **Equivalence Tests**

    * Write tests that compare the output of your Julia `TDVP` against the Python `yaqs` results for small systems ($L=4, 6$) to ensure correctness.
14. **Performance Benchmarks**

    * Benchmark `expect` and `tdvp_step` vs the Python implementation.

---

### Usage Guide for the Implementing AI

**Prompt for the AI:**
"Please follow the **Julia Implementation Roadmap** above. We have already analyzed the Python codebase and identified specific inefficiencies to avoid (e.g., `GateLibrary` pattern, `deepcopy` in stochastic loops). Start with **Phase 1**, specifically implementing the `AbstractOperator` hierarchy and the immutable gate structs using `StaticArrays`."
