module Simulator

using Base.Threads
using ..MPSModule
using ..MPOModule
using ..SimulationConfigs
using ..NoiseModule
using ..AnalogTJM
using ..CircuitTJM


export run, available_cpus

function _progress_bar_string(done::Int, total::Int; width::Int=30)
    total <= 0 && return "[ ] 0/0"
    frac = clamp(done / total, 0.0, 1.0)
    filled = Int(floor(frac * width))
    filled = clamp(filled, 0, width)
    bar = string("[", repeat("=", filled), repeat(".", width - filled), "]")
    return string(bar, " (", done, "/", total, ")")
end

function _print_traj_progress(done::Int, total::Int; maxbond_sofar::Int=0, width::Int=30)
    msg = _progress_bar_string(done, total; width=width)
    if maxbond_sofar > 0
        msg = string(msg, "  Max Bond: ", maxbond_sofar)
    end
    # \r overwrites the current line; trailing spaces clear leftovers from longer previous prints
    print("\r", msg, "    ")
    flush(stdout)
    return nothing
end

"""
Return the number of available CPU threads.

This checks for SLURM-provided CPU counts and falls back to Julia's `Sys.CPU_THREADS`.

Args:
    None

Returns:
    Int: Number of available CPU threads.
"""
function available_cpus()
    if haskey(ENV, "SLURM_CPUS_ON_NODE")
        return parse(Int, ENV["SLURM_CPUS_ON_NODE"])
    else
        return Sys.CPU_THREADS
    end
end

"""
Run an analog (Hamiltonian) simulation over multiple trajectories.

This selects a first- or second-order TJM backend, runs trajectories in parallel when requested,
and aggregates observable trajectories into results.

Args:
    initial_state (MPS): Initial state to evolve.
    operator (MPO): Hamiltonian MPO.
    sim_params (TimeEvolutionConfig): Simulation configuration.
    noise_model (Union{NoiseModel, Nothing}): Noise model or `nothing` to disable noise.
    parallel (Bool): Whether to parallelize over trajectories.
    kwargs: Additional keyword arguments forwarded to the backend.

Returns:
    Nothing: Results are stored in `sim_params.observables`.
"""
function _run_analog(initial_state::MPS, operator::MPO, sim_params::TimeEvolutionConfig, noise_model::Union{NoiseModel, Nothing}; parallel::Bool=true, kwargs...)
    # Backend selection
    backend = (sim_params.order == 1) ? analog_tjm_1 : analog_tjm_2

    # Noise check
    if isnothing(noise_model) || all(p -> p.strength == 0, noise_model.processes)
        sim_params.num_traj = 1
    end

    # Initialize observables
    for obs in sim_params.observables
        SimulationConfigs.initialize!(obs, sim_params)
    end

    # Parallel Loop
    if parallel && sim_params.num_traj > 1
        counter = Atomic{Int}(0)
        total = sim_params.num_traj
        progress_lock = ReentrantLock()
        # Update once per finished trajectory (user-facing progress bar).
        stride = 1
        
        Threads.@threads for i in 1:sim_params.num_traj
            args = (i, initial_state, noise_model, sim_params, operator)
            result = backend(args)
            
            for (obs_idx, obs) in enumerate(sim_params.observables)
                obs.trajectories[i, :] = result[obs_idx, :]
            end
            
            c = atomic_add!(counter, 1) + 1
            if c % stride == 0 || c == total
                lock(progress_lock) do
                    _print_traj_progress(c, total)
                    if c == total
                        println()
                    end
                end
            end
        end
    else
        for i in 1:sim_params.num_traj
            args = (i, initial_state, noise_model, sim_params, operator)
            result = backend(args)
            for (obs_idx, obs) in enumerate(sim_params.observables)
                obs.trajectories[i, :] = result[obs_idx, :]
            end
            _print_traj_progress(i, sim_params.num_traj)
        end
        println()
    end

    # Aggregate
    SimulationConfigs.aggregate_trajectories!(sim_params)
end

"""
Run a digital circuit simulation over multiple trajectories.

This executes the digital TJM backend for each trajectory, optionally in parallel, and aggregates
observable trajectories into results while tracking bond dimensions.

Args:
    initial_state (MPS): Initial state to evolve.
    circuit: Digital circuit or repeated circuit wrapper.
    sim_params (TimeEvolutionConfig): Simulation configuration.
    noise_model (Union{NoiseModel, Nothing}): Noise model or `nothing` to disable noise.
    parallel (Bool): Whether to parallelize over trajectories.
    kwargs: Additional keyword arguments forwarded to the backend.

Returns:
    Vector{Vector{Int}}: Per-trajectory bond dimension traces.
"""
function _run_digital(initial_state::MPS, circuit, sim_params::TimeEvolutionConfig, noise_model::Union{NoiseModel, Nothing}; parallel::Bool=true, kwargs...)
    
    # Noise check
    if isnothing(noise_model) || all(p -> p.strength == 0, noise_model.processes)
        sim_params.num_traj = 1
    end
    
    for obs in sim_params.observables
        SimulationConfigs.initialize!(obs, sim_params)
    end

    all_bond_dims = Vector{Vector{Int}}(undef, sim_params.num_traj)

    # Parallel Loop
    if parallel && sim_params.num_traj > 1
        counter = Atomic{Int}(0)
        total = sim_params.num_traj
        progress_lock = ReentrantLock()
        maxbond_sofar = Ref{Int}(0)
        # Update once per finished trajectory (user-facing progress bar).
        stride = 1
        
        Threads.@threads for i in 1:sim_params.num_traj
            # run_circuit_tjm returns (state, results, bond_dims)
            _, result, bond_dims = run_circuit_tjm(initial_state, circuit, noise_model, sim_params; kwargs...)
            
            all_bond_dims[i] = bond_dims
            

            if size(result, 2) != length(sim_params.times)

            end
            
            for (obs_idx, obs) in enumerate(sim_params.observables)
                # Copy what we have
                n_copy = min(size(result, 2), size(obs.trajectories, 2))
                obs.trajectories[i, 1:n_copy] = result[obs_idx, 1:n_copy]
            end
            
            c = atomic_add!(counter, 1) + 1
            # Update progress occasionally to reduce contention
            if c % stride == 0 || c == total
                local_maxbond = isempty(bond_dims) ? 0 : maximum(bond_dims)
                lock(progress_lock) do
                    if local_maxbond > maxbond_sofar[]
                        maxbond_sofar[] = local_maxbond
                    end
                    _print_traj_progress(c, total; maxbond_sofar=maxbond_sofar[])
                    if c == total
                        println()
                    end
                end
            end
        end
    else
        maxbond_sofar = 0
        for i in 1:sim_params.num_traj
            _, result, bond_dims = run_circuit_tjm(initial_state, circuit, noise_model, sim_params; kwargs...)
            all_bond_dims[i] = bond_dims
            for (obs_idx, obs) in enumerate(sim_params.observables)
                n_copy = min(size(result, 2), size(obs.trajectories, 2))
                obs.trajectories[i, 1:n_copy] = result[obs_idx, 1:n_copy]
            end
            local_maxbond = isempty(bond_dims) ? 0 : maximum(bond_dims)
            maxbond_sofar = max(maxbond_sofar, local_maxbond)
            _print_traj_progress(i, sim_params.num_traj; maxbond_sofar=maxbond_sofar)
        end
        println()
    end
    
    SimulationConfigs.aggregate_trajectories!(sim_params)
    return all_bond_dims
end

"""
Run a simulation given an MPO or digital circuit.

This normalizes the initial state and dispatches to the analog or digital backend based on the
operator type, using the provided simulation configuration.

Args:
    initial_state (MPS): Initial state to evolve.
    operator_or_circuit: MPO for analog runs or circuit for digital runs.
    sim_params: Simulation configuration (TimeEvolutionConfig).
    noise_model: Noise model or `nothing` to disable noise.
    parallel (Bool): Whether to parallelize over trajectories.
    kwargs: Additional keyword arguments forwarded to the backend.

Returns:
    Any: Backend-specific result (e.g., bond dimension traces for digital runs).

Raises:
    ErrorException: If the operator type or configuration is unsupported.
"""
function run(initial_state::MPS, operator_or_circuit, sim_params, noise_model=nothing; parallel::Bool=true, kwargs...)
    MPSModule.normalize!(initial_state; form="B") 
    
    if isa(sim_params, TimeEvolutionConfig)
        if isa(operator_or_circuit, MPO)
            return _run_analog(initial_state, operator_or_circuit, sim_params, noise_model; parallel=parallel, kwargs...)
        elseif isa(operator_or_circuit, DigitalCircuit) || isa(operator_or_circuit, CircuitTJM.RepeatedDigitalCircuit)
            return _run_digital(initial_state, operator_or_circuit, sim_params, noise_model; parallel=parallel, kwargs...)
        else
             error("Simulation requires MPO (Analog) or DigitalCircuit (Digital).")
        end
    else
        error("Only TimeEvolutionConfig is supported.")
    end
end

end
