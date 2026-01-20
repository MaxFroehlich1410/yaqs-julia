module DigitalTJM

using LinearAlgebra
using Base.Threads
using TensorOperations
using ..MPSModule
using ..MPOModule
using ..GateLibrary
using ..NoiseModule
using ..DissipationModule
using ..SimulationConfigs
using ..Algorithms
using ..StochasticProcessModule
import ..Timing
using ..Timing: @t

export DigitalGate, DigitalCircuit, RepeatedDigitalCircuit, add_gate!, process_circuit, run_digital_tjm, TJMOptions,
       enable_timing!, set_timing_print_each_call!, reset_timing!, print_timing_summary!

"""
Enable or disable timing collection for DigitalTJM.

This toggles timing instrumentation for DigitalTJM and propagates the setting to TDVP, dissipation,
and stochastic submodules to capture nested timings.

Args:
    flag (Bool): Whether to enable timing collection.

Returns:
    Nothing: Timing settings are updated globally.
"""
enable_timing!(flag::Bool=true) = Timing.enable_timing!(flag)

"""
Configure whether timing data is printed for each call.

This controls automatic printing of timing results after each instrumented call in the Timing module.

Args:
    flag (Bool): Whether to print timing results after each call.

Returns:
    Nothing: Timing print behavior is updated.
"""
set_timing_print_each_call!(flag::Bool=true) = Timing.set_timing_print_each_call!(flag)
"""
Reset all collected timing statistics.

This clears accumulated timing data tracked by the Timing module.

Args:
    None

Returns:
    Nothing: Timing statistics are cleared.
"""
reset_timing!() = Timing.reset_timing!()
"""
Print a summary of collected timing statistics.

This prints the most expensive timing entries with a configurable header and number of rows.

Args:
    header (AbstractString): Header to display above the summary.
    top (Int): Number of top entries to print.

Returns:
    Nothing: Timing summary is printed to stdout.
"""
print_timing_summary!(; header::AbstractString="DigitalTJM timing summary", top::Int=20) =
    Timing.print_timing_summary!(; header=header, top=top)

# --- Data Structures ---

"""
Represent a digital gate applied to one or more sites.

This stores the gate operator, the target site indices, and an optional generator for
Hamiltonian-based evolution of two-qubit gates.

Args:
    op (AbstractOperator): Gate operator instance.
    sites (Vector{Int}): Target qubit indices (1-based).
    generator (Union{Vector{AbstractMatrix{ComplexF64}}, Nothing}): Optional generator matrices.

Returns:
    DigitalGate: Gate description for digital simulation.
"""
struct DigitalGate
    op::AbstractOperator
    sites::Vector{Int}
    generator::Union{Vector{<:AbstractMatrix{ComplexF64}}, Nothing}
end

"""
Store a digital circuit as a sequence of gates and optional layers.

This structure holds the total number of qubits, the flat gate list, and optionally
preprocessed layers of commuting gates.

Args:
    num_qubits (Int): Total number of qubits in the circuit.
    gates (Vector{DigitalGate}): Flat gate list in execution order.
    layers (Vector{Vector{DigitalGate}}): Optional layered gate list.

Returns:
    DigitalCircuit: Circuit container for digital TJM execution.
"""
mutable struct DigitalCircuit
    num_qubits::Int
    gates::Vector{DigitalGate}
    layers::Vector{Vector{DigitalGate}} # Optional: processed layers
end

"""
Represent a circuit repeated multiple times without duplication.

This holds a single `step` circuit and an integer repeat count, avoiding materializing
`repeats` copies of identical gate lists for large circuits.

Args:
    step (DigitalCircuit): Circuit to repeat.
    repeats (Int): Number of repetitions.

Returns:
    RepeatedDigitalCircuit: Compact repeated-circuit representation.
"""
struct RepeatedDigitalCircuit
    step::DigitalCircuit
    repeats::Int
end

"""
Create an empty digital circuit with a given number of qubits.

This initializes a `DigitalCircuit` with no gates and no processed layers.

Args:
    n (Int): Number of qubits.

Returns:
    DigitalCircuit: Empty circuit container.
"""
function DigitalCircuit(n::Int)
    return DigitalCircuit(n, DigitalGate[], Vector{Vector{DigitalGate}}())
end

"""
Append a gate to a digital circuit.

This constructs a `DigitalGate` from the operator and target sites, optionally attaching a
generator, and appends it to the circuit's flat gate list.

Args:
    circ (DigitalCircuit): Circuit to append to.
    op (AbstractOperator): Gate operator.
    sites (Vector{Int}): Target qubit indices (1-based).
    generator: Optional generator matrices for Hamiltonian evolution.

Returns:
    Nothing: The circuit is updated in-place.
"""
function add_gate!(circ::DigitalCircuit, op::AbstractOperator, sites::Vector{Int}; generator=nothing)
    push!(circ.gates, DigitalGate(op, sites, generator))
end

# --- Options ---

"""
Configure algorithm choices for DigitalTJM gate application.

This selects the local and long-range update methods, typically `:TEBD` or `:TDVP`, to control how
gates are applied during simulation.

Args:
    local_method (Symbol): Method for nearest-neighbor gates.
    long_range_method (Symbol): Method for long-range gates.

Returns:
    TJMOptions: Options container for DigitalTJM.
"""
struct TJMOptions
    local_method::Symbol # :TEBD or :TDVP
    long_range_method::Symbol # :TEBD or :TDVP
end

"""
Create TJMOptions with default methods.

This provides keyword defaults for local and long-range methods, both set to `:TDVP` unless
overridden.

Args:
    local_method (Symbol): Method for nearest-neighbor gates.
    long_range_method (Symbol): Method for long-range gates.

Returns:
    TJMOptions: Options container with the specified methods.
"""
TJMOptions(;local_method=:TDVP, long_range_method=:TDVP) = TJMOptions(local_method, long_range_method)

# --- Circuit Processing ---

"""
Group circuit gates into commuting layers and barrier markers.

This performs a greedy layering pass that collects gates into disjoint layers based on qubit
overlap and records barriers by layer index for sampling control.

Args:
    circuit (DigitalCircuit): Circuit whose gate list is processed.

Returns:
    Tuple: `(layers, barrier_map)` where `layers` is a vector of gate layers and `barrier_map` maps
        layer indices to barrier labels.
"""
function process_circuit(circuit::DigitalCircuit)
    # Simple greedy layering
    layers = Vector{Vector{DigitalGate}}()
    barrier_map = Dict{Int, Vector{String}}()
    
    current_layer = DigitalGate[]
    busy_qubits = Set{Int}()
    
    layer_idx = 1
    
    for gate in circuit.gates
        # Check for Barrier
        if gate.op isa GateLibrary.Barrier
            # If barrier, finish current layer
            if !isempty(current_layer)
                push!(layers, current_layer)
                current_layer = DigitalGate[]
                busy_qubits = Set{Int}()
                layer_idx += 1
            end
            
            # Record barrier
            idx = !isempty(layers) ? length(layers) : 0
            if !haskey(barrier_map, idx)
                barrier_map[idx] = String[]
            end
            push!(barrier_map[idx], gate.op.label)
            continue
        end
        
        # Check commutativity / qubit overlap
        overlap = false
        for q in gate.sites
            if q in busy_qubits
                overlap = true
                break
            end
        end
        
        if overlap
            # Push current layer
            push!(layers, current_layer)
            current_layer = DigitalGate[]
            busy_qubits = Set{Int}()
            layer_idx += 1
        end
        
        # Add to current
        push!(current_layer, gate)
        for q in gate.sites
            push!(busy_qubits, q)
        end
    end
    
    if !isempty(current_layer)
        push!(layers, current_layer)
    end
    
    return layers, barrier_map
end

"""
Check whether a barrier label requests sampling at a given layer.

This inspects the barrier map for the provided layer index and looks for a `SAMPLE_OBSERVABLES`
label in a case-insensitive manner.

Args:
    barrier_map (Dict{Int, Vector{String}}): Mapping from layer index to barrier labels.
    idx (Int): Layer index to inspect.

Returns:
    Bool: `true` if a sampling barrier is present, otherwise `false`.
"""
@inline function _has_sample_barrier(barrier_map::Dict{Int, Vector{String}}, idx::Int)
    if !haskey(barrier_map, idx)
        return false
    end
    for label in barrier_map[idx]
        if uppercase(label) == "SAMPLE_OBSERVABLES"
            return true
        end
    end
    return false
end

"""
Create a sampling plan based on barrier positions.

This determines whether to sample at the start and which layer indices should trigger sampling
after their execution.

Args:
    barrier_map (Dict{Int, Vector{String}}): Mapping from layer index to barrier labels.
    num_layers (Int): Total number of circuit layers.

Returns:
    Tuple: `(sample_at_start, sample_after)` where `sample_at_start` is a Bool and `sample_after`
        is a vector of layer indices.
"""
@inline function _sample_plan(barrier_map::Dict{Int, Vector{String}}, num_layers::Int)
    sample_at_start = _has_sample_barrier(barrier_map, 0)
    sample_after = Int[]
    for l in 1:num_layers
        if _has_sample_barrier(barrier_map, l)
            push!(sample_after, l)
        end
    end
    return sample_at_start, sample_after
end

# --- Noise Helpers ---

"""
Extract a local noise model affecting a specific two-site gate.

This filters the global noise processes to those that act on either of the two target sites,
returning a new `NoiseModel` containing only the relevant processes.

Args:
    noise_model (NoiseModel): Global noise model.
    site1 (Int): First site index.
    site2 (Int): Second site index.

Returns:
    NoiseModel: Local noise model for the specified sites.
"""
function create_local_noise_model(noise_model::NoiseModel{T}, site1::Int, site2::Int) where T
    affected_sites = Set([site1, site2])
    local_procs = Vector{AbstractNoiseProcess{T}}()
    for proc in noise_model.processes
        p_sites = proc.sites
        if length(p_sites) == 1
            if p_sites[1] == site1 || p_sites[1] == site2
                push!(local_procs, proc)
            end
        elseif length(p_sites) == 2
            if (p_sites[1] == site1 && p_sites[2] == site2) || (p_sites[1] == site2 && p_sites[2] == site1)
                 push!(local_procs, proc)
            end
        end
    end
    return NoiseModel(local_procs)
end

# --- Helpers ---

"""
Construct a windowed MPO for a two-site gate.

This builds an MPO over the specified window that inserts the two-site gate generator on the
targeted sites and identities elsewhere, matching the gate's generator ordering.

Args:
    gate (DigitalGate): Two-site gate defining the generator.
    window_start (Int): Starting site index of the window.
    window_end (Int): Ending site index of the window.

Returns:
    MPO: Windowed MPO for applying the gate via TDVP.

Raises:
    AssertionError: If the gate sites fall outside the window.
"""
function construct_window_mpo(gate::DigitalGate, window_start::Int, window_end::Int)
    L_window = window_end - window_start + 1
    tensors = Vector{Array{ComplexF64, 4}}(undef, L_window)
    phys_dims = fill(2, L_window)
    
    s1, s2 = sort(gate.sites)
    rel_s1 = s1 - window_start + 1
    rel_s2 = s2 - window_start + 1
    
    # Get generator from gate, or from operator if not set
    gen = gate.generator
    if isnothing(gen)
        gen = GateLibrary.generator(gate.op)
    end
    coeff = GateLibrary.hamiltonian_coeff(gate.op)
    
    @assert rel_s1 >= 1 && rel_s2 <= L_window "Gate sites outside window"

    # Determine which generator element goes to which site
    # gate.sites[1] corresponds to gen[1]
    # gate.sites[2] corresponds to gen[2]
    # s1 is min(sites), s2 is max(sites)
    
    g_s1 = gen[1]
    g_s2 = gen[2]
    
    if gate.sites[1] != s1
        # If sites were sorted/swapped (e.g. [2, 1] -> s1=1, s2=2), 
        # then s1 corresponds to sites[2] -> gen[2]
        g_s1 = gen[2]
        g_s2 = gen[1]
    end
    
    for i in 1:L_window
        T = zeros(ComplexF64, 1, 2, 2, 1)
        T[1, :, :, 1] = Matrix{ComplexF64}(I, 2, 2)
        if i == rel_s1
            T[1, :, :, 1] = coeff * g_s1
        elseif i == rel_s2
            T[1, :, :, 1] = g_s2
        end
        tensors[i] = T
    end
    return MPO(L_window, tensors, phys_dims, 0)
end

"""
Apply a single-qubit gate to an MPS in-place.

This contracts the 1-qubit operator with the physical index of the target site tensor, preserving
the MPS layout `(Left, Phys, Right)`.

Args:
    mps (MPS): State to update in-place.
    gate (DigitalGate): Single-qubit gate with target site.

Returns:
    Nothing: The MPS tensor at the target site is updated.
"""
function apply_single_qubit_gate!(mps::MPS, gate::DigitalGate)
    site = gate.sites[1]
    @t :matrix_1q begin
        op_mat = matrix(gate.op)
        A = mps.tensors[site]
        L, d, R = size(A)
        @t :permutedims_1q A_perm = reshape(permutedims(A, (2, 1, 3)), d, L*R)
        @t :mul_1q New_A_mat = op_mat * A_perm
        @t :permutedims_1q New_A = permutedims(reshape(New_A_mat, d, L, R), (2, 1, 3))
        mps.tensors[site] = New_A
    end
end

"""
Apply a nearest-neighbor two-qubit gate using exact TEBD.

This moves the orthogonality center, contracts the two-site tensor, applies the gate, and splits
with SVD and truncation according to the simulation configuration.

Args:
    mps (MPS): State to update in-place.
    op (AbstractOperator): Two-qubit gate operator.
    s1 (Int): Left site index.
    s2 (Int): Right site index.
    config (AbstractSimConfig): Configuration with truncation settings.

Returns:
    Nothing: The MPS tensors and orthogonality center are updated in-place.
"""
function apply_local_gate_exact!(mps::MPS, op::AbstractOperator, s1::Int, s2::Int, config::AbstractSimConfig)
    # Standard TEBD update for nearest neighbor gate
    # Moves orthogonality center to s1 (assumes mixed/right canonical to right of s1)
    @t :shift_orth_center MPSModule.shift_orthogonality_center!(mps, s1)
    
    A1 = mps.tensors[s1]
    A2 = mps.tensors[s2]
    
    # Contract A1 * A2
    # A1: (l1, p1, b)
    # A2: (b, p2, r2)
    @t :contract_theta @tensor Theta[l1, p1, p2, r2] := A1[l1, p1, k] * A2[k, p2, r2]
    
    # Contract with Gate
    # op_mat: 4x4 (acting on p1, p2)
    # Reshape op_mat to (p1_out, p2_out, p1_in, p2_in)
    @t :matrix_2q op_mat = matrix(op)
    op_tensor = reshape(op_mat, 2, 2, 2, 2)
    
    @t :contract_gate @tensor Theta_prime[l1, p1_out, p2_out, r2] := op_tensor[p1_out, p2_out, p1_in, p2_in] * Theta[l1, p1_in, p2_in, r2]
    
    # SVD and Truncate
    l1_dim, p1_dim, p2_dim, r2_dim = size(Theta_prime)
    @t :reshape_theta Theta_matrix = reshape(Theta_prime, l1_dim * p1_dim, p2_dim * r2_dim)
    
    @t :svd F = svd(Theta_matrix)
    
    # Truncation Logic
    threshold = config.truncation_threshold
    max_bond = config.max_bond_dim
    
    # Truncate based on threshold
    current_sum = 0.0
    keep_count = length(F.S)
    @t :truncation_loop for k in length(F.S):-1:1
        if current_sum + F.S[k]^2 > threshold
            break
        end
        current_sum += F.S[k]^2
        keep_count -= 1
    end
    keep = clamp(keep_count, 1, max_bond)
    
    U = F.U[:, 1:keep]
    S = F.S[1:keep]
    Vt = F.Vt[1:keep, :]
    
    # Update MPS
    # A1 becomes U (Left Canonical)
    @t :update_mps mps.tensors[s1] = reshape(U, l1_dim, p1_dim, keep)
    
    # A2 becomes S * V (Right Canonical? No, Center)
    @t :update_mps mps.tensors[s2] = reshape(Diagonal(S) * Vt, keep, p2_dim, r2_dim)
    mps.orth_center = s2
end

"""
Apply a gate using either local TEBD or windowed TDVP.

This selects the update method based on gate range and algorithm options, using swap networks for
long-range TEBD or a windowed TDVP evolution for more accurate updates.

Args:
    state (MPS): State to update in-place.
    gate (DigitalGate): Gate to apply.
    sim_params (AbstractSimConfig): Simulation configuration.
    alg_options (TJMOptions): Algorithm selection for local and long-range gates.

Returns:
    Nothing: The MPS is updated in-place.
"""
function apply_window!(state::MPS, gate::DigitalGate, sim_params::AbstractSimConfig, alg_options::TJMOptions)
    s1, s2 = sort(gate.sites)
    is_long_range = (s2 > s1 + 1)
    
    method = is_long_range ? alg_options.long_range_method : alg_options.local_method
    
    if method == :TEBD
        if is_long_range
            # Swap Network + Local TEBD
            
            # Swap Path: (s2-1, s2), (s2-2, s2-1), ..., (s1+1, s1+2)
            # s2 moves LEFT to s1+1
            for k in (s2-1):-1:(s1+1)
                @t :apply_swap apply_local_gate_exact!(state, SWAPGate(), k, k+1, sim_params)
            end
            
            # Apply gate on (s1, s1+1)
            @t :apply_local_gate_exact apply_local_gate_exact!(state, gate.op, s1, s1+1, sim_params)
            
            # Unwind Swaps: (s1+1, s1+2), ..., (s2-1, s2)
            # s2 moves RIGHT back to s2
            for k in (s1+1):(s2-1)
                @t :apply_swap apply_local_gate_exact!(state, SWAPGate(), k, k+1, sim_params)
            end
        else
            # Nearest Neighbor: Direct
            @t :apply_local_gate_exact apply_local_gate_exact!(state, gate.op, s1, s2, sim_params)
        end
    else # :TDVP
        padding = 0
        win_start = max(1, s1 - padding)
        win_end = min(state.length, s2 + padding)
        @t :shift_orth_center MPSModule.shift_orthogonality_center!(state, win_start)
        win_len = win_end - win_start + 1
        @t :window_slice begin
            win_tensors = state.tensors[win_start:win_end]
            win_phys = state.phys_dims[win_start:win_end]
            short_state = MPS(win_len, win_tensors, win_phys, 1)
        end
        @t :construct_window_mpo short_mpo = construct_window_mpo(gate, win_start, win_end)
        
        # Use StrongMeasurementConfig to trigger the efficient Directed Circuit Sweep
        # instead of the Symmetric Hamiltonian Sweep triggered by TimeEvolutionConfig.
        gate_config = StrongMeasurementConfig(Observable[]; 
                                            max_bond_dim=sim_params.max_bond_dim, 
                                            truncation_threshold=sim_params.truncation_threshold)
        
        @t :two_site_tdvp two_site_tdvp!(short_state, short_mpo, gate_config)
    
        @t :window_writeback state.tensors[win_start:win_end] .= short_state.tensors
        state.orth_center = win_end
    end
end

# --- Main Runner ---

"""
Run DigitalTJM for a single digital circuit instance.

This processes the circuit into layers, applies gates to a deep-copied state, applies noise if
present, and records observables according to sampling barriers.

Args:
    initial_state (MPS): Initial state to copy and evolve.
    circuit (DigitalCircuit): Circuit to execute.
    noise_model (Union{NoiseModel, Nothing}): Noise model or `nothing` to disable noise.
    sim_params (AbstractSimConfig): Simulation parameters including observables and truncation.
    alg_options (TJMOptions): Algorithm options for local and long-range gates.

Returns:
    Tuple: `(state, results, bond_dims)` containing the final state, observable measurements, and
        maximum bond dimensions per sample.
"""
function run_digital_tjm(initial_state::MPS, circuit::DigitalCircuit, 
                            noise_model::Union{NoiseModel, Nothing}, 
                            sim_params::AbstractSimConfig;
                            alg_options::TJMOptions = TJMOptions(local_method=:TDVP, long_range_method=:TDVP))

    local ts = Timing.begin_scope!()
    try
        state = @t :deepcopy_state deepcopy(initial_state)
        layers, barrier_map = @t :process_circuit process_circuit(circuit)
        num_layers = length(layers)
        num_obs = length(sim_params.observables)

        sample_at_start, sample_after = _sample_plan(barrier_map, num_layers)
        num_steps = (sample_at_start ? 1 : 0) + length(sample_after)
        if num_steps == 0
            num_steps = 1
        end
    
    results = zeros(ComplexF64, num_obs, num_steps)
    bond_dims = zeros(Int, num_steps)
    current_meas_idx = 1
    
    """
    Measure observables and record bond dimension at a sample index.

    This evaluates all configured observables and stores them along with the current maximum bond
    dimension into the results arrays.

    Args:
        idx (Int): Column index to write measurements into.

    Returns:
        Nothing: Results are written into `results` and `bond_dims`.
    """
    function measure!(idx)
        @t :measure begin
            for (i, obs) in enumerate(sim_params.observables)
                results[i, idx] = @t :expect SimulationConfigs.expect(state, obs)
            end
            bond_dims[idx] = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
        end
    end
    
    if sample_at_start
        measure!(current_meas_idx)
        current_meas_idx += 1
    end
    
    for (l_idx, layer) in enumerate(layers)
        for gate in layer
            if length(gate.sites) == 1
                @t :apply_single_qubit_gate apply_single_qubit_gate!(state, gate)
            end
        end
        for gate in layer
            if length(gate.sites) == 2
                @t :apply_window apply_window!(state, gate, sim_params, alg_options)
                if !isnothing(noise_model) && !isempty(noise_model.processes)
                    s1, s2 = sort(gate.sites)
                    local_noise = @t :create_local_noise_model create_local_noise_model(noise_model, s1, s2)
                    if !isempty(local_noise.processes)
                        @t :apply_dissipation apply_dissipation(state, local_noise, 1.0, sim_params)
                        @t :stochastic_process stochastic_process!(state, local_noise, 1.0, sim_params)
                    else
                         @t :normalize MPSModule.normalize!(state)
                    end
                else
                     @t :normalize MPSModule.normalize!(state)
                end
            end
        end

        # Progress logging (overwrite line with \r)
        current_bond = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
        print("\r\tLayer $l_idx/$num_layers | Max Bond: $current_bond")
        flush(stdout)

        if l_idx in sample_after
            measure!(current_meas_idx)
            current_meas_idx += 1
        end
    end
    print("\n") # Newline after finishing all layers of this trajectory
    
    # If the circuit did not request sampling (no SAMPLE_OBSERVABLES barriers),
    # still produce at least one measurement for compatibility.
    if !sample_at_start && isempty(sample_after) && num_obs > 0
        measure!(1)
    end
        # If the circuit did not request sampling (no SAMPLE_OBSERVABLES barriers),
        # still produce at least one measurement for compatibility.
        if !sample_at_start && isempty(sample_after) && num_obs > 0
            measure!(1)
        end
    
    return state, results, bond_dims
    finally
        Timing.end_scope!(ts; header="DigitalTJM per-trajectory timing")
    end
end

"""
Run DigitalTJM for a repeated digital circuit without materializing repeats.

This executes the layers of a step circuit multiple times, applying noise and collecting
measurements based on barrier sampling rules.

Args:
    initial_state (MPS): Initial state to copy and evolve.
    circuit (RepeatedDigitalCircuit): Repeated circuit wrapper.
    noise_model (Union{NoiseModel, Nothing}): Noise model or `nothing` to disable noise.
    sim_params (AbstractSimConfig): Simulation parameters including observables and truncation.
    alg_options (TJMOptions): Algorithm options for local and long-range gates.

Returns:
    Tuple: `(state, results, bond_dims)` containing the final state, observable measurements, and
        maximum bond dimensions per sample.
"""
function run_digital_tjm(initial_state::MPS, circuit::RepeatedDigitalCircuit,
                         noise_model::Union{NoiseModel, Nothing},
                         sim_params::AbstractSimConfig;
                         alg_options::TJMOptions = TJMOptions(local_method=:TDVP, long_range_method=:TDVP))
    local ts = Timing.begin_scope!()
    try
        state = @t :deepcopy_state deepcopy(initial_state)
        layers, barrier_map = @t :process_circuit process_circuit(circuit.step)
        step_layers = length(layers)
        repeats = circuit.repeats

        num_obs = length(sim_params.observables)
        sample_at_start, sample_after = _sample_plan(barrier_map, step_layers)

        num_steps = (sample_at_start ? 1 : 0) + repeats * length(sample_after)
        if num_steps == 0
            num_steps = 1
        end

        results = zeros(ComplexF64, num_obs, num_steps)
        bond_dims = zeros(Int, num_steps)
        current_meas_idx = 1

        """
        Measure observables and record bond dimension at a sample index.

        This evaluates all configured observables and stores them along with the current maximum
        bond dimension into the results arrays.

        Args:
            idx (Int): Column index to write measurements into.

        Returns:
            Nothing: Results are written into `results` and `bond_dims`.
        """
        function measure!(idx)
            @t :measure begin
                for (i, obs) in enumerate(sim_params.observables)
                    results[i, idx] = @t :expect SimulationConfigs.expect(state, obs)
                end
                bond_dims[idx] = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
            end
        end

        if sample_at_start
            measure!(current_meas_idx)
            current_meas_idx += 1
        end

        total_layers = repeats * step_layers
        for rep in 1:repeats
            for (l_idx, layer) in enumerate(layers)
                for gate in layer
                    if length(gate.sites) == 1
                        @t :apply_single_qubit_gate apply_single_qubit_gate!(state, gate)
                    end
                end
                for gate in layer
                    if length(gate.sites) == 2
                        @t :apply_window apply_window!(state, gate, sim_params, alg_options)
                        if !isnothing(noise_model) && !isempty(noise_model.processes)
                            s1, s2 = sort(gate.sites)
                            local_noise = @t :create_local_noise_model create_local_noise_model(noise_model, s1, s2)
                            if !isempty(local_noise.processes)
                                @t :apply_dissipation apply_dissipation(state, local_noise, 1.0, sim_params)
                                @t :stochastic_process stochastic_process!(state, local_noise, 1.0, sim_params)
                            else
                                @t :normalize MPSModule.normalize!(state)
                            end
                        else
                            @t :normalize MPSModule.normalize!(state)
                        end
                    end
                end

                global_layer = (rep - 1) * step_layers + l_idx
                current_bond = @t :write_max_bond_dim MPSModule.write_max_bond_dim(state)
                print("\r\tLayer $global_layer/$total_layers | Max Bond: $current_bond")
                flush(stdout)

                if l_idx in sample_after
                    measure!(current_meas_idx)
                    current_meas_idx += 1
                end
            end
        end
        print("\n")

        if num_steps == 1 && num_obs > 0 && !sample_at_start && isempty(sample_after)
            measure!(1)
        end

        return state, results, bond_dims
    finally
        Timing.end_scope!(ts; header="DigitalTJM per-trajectory timing")
    end
end

end # module
