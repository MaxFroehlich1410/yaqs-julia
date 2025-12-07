module Simulator

using Base.Threads
using ..MPSModule
using ..MPOModule
using ..SimulationConfigs
using ..NoiseModule
using ..AnalogTJM
using ..DigitalTJM


export run, available_cpus

function available_cpus()
    if haskey(ENV, "SLURM_CPUS_ON_NODE")
        return parse(Int, ENV["SLURM_CPUS_ON_NODE"])
    else
        return Sys.CPU_THREADS
    end
end

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
        
        Threads.@threads for i in 1:sim_params.num_traj
            args = (i, initial_state, noise_model, sim_params, operator)
            result = backend(args)
            
            for (obs_idx, obs) in enumerate(sim_params.observables)
                obs.trajectories[i, :] = result[obs_idx, :]
            end
            
            c = atomic_add!(counter, 1) + 1
            if c % 10 == 0 || c == total
                print("\rProgress: $c / $total trajectories finished.")
                if c == total
                    println()
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
        end
    end

    # Aggregate
    SimulationConfigs.aggregate_trajectories!(sim_params)
end

function _run_digital(initial_state::MPS, circuit::DigitalCircuit, sim_params::TimeEvolutionConfig, noise_model::Union{NoiseModel, Nothing}; parallel::Bool=true, kwargs...)
    
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
        
        Threads.@threads         for i in 1:sim_params.num_traj
            # run_digital_tjm returns (state, results, bond_dims)
            _, result, bond_dims = run_digital_tjm(initial_state, circuit, noise_model, sim_params; kwargs...)
            
            all_bond_dims[i] = bond_dims
            

            if size(result, 2) != length(sim_params.times)

            end
            
            for (obs_idx, obs) in enumerate(sim_params.observables)
                # Copy what we have
                n_copy = min(size(result, 2), size(obs.trajectories, 2))
                obs.trajectories[i, 1:n_copy] = result[obs_idx, 1:n_copy]
            end
            
            c = atomic_add!(counter, 1) + 1
            if c % 10 == 0 || c == total
                print("\rProgress: $c / $total trajectories finished.")
                if c == total; println(); end
            end
        end
    else
        for i in 1:sim_params.num_traj
            _, result, bond_dims = run_digital_tjm(initial_state, circuit, noise_model, sim_params; kwargs...)
            all_bond_dims[i] = bond_dims
            for (obs_idx, obs) in enumerate(sim_params.observables)
                n_copy = min(size(result, 2), size(obs.trajectories, 2))
                obs.trajectories[i, 1:n_copy] = result[obs_idx, 1:n_copy]
            end
        end
    end
    
    SimulationConfigs.aggregate_trajectories!(sim_params)
    return all_bond_dims
end

function run(initial_state::MPS, operator_or_circuit, sim_params, noise_model=nothing; parallel::Bool=true, kwargs...)
    MPSModule.normalize!(initial_state; form="B") 
    
    if isa(sim_params, TimeEvolutionConfig)
        if isa(operator_or_circuit, MPO)
            return _run_analog(initial_state, operator_or_circuit, sim_params, noise_model; parallel=parallel, kwargs...)
        elseif isa(operator_or_circuit, DigitalCircuit)
            return _run_digital(initial_state, operator_or_circuit, sim_params, noise_model; parallel=parallel, kwargs...)
        else
             error("Simulation requires MPO (Analog) or DigitalCircuit (Digital).")
        end
    else
        error("Only TimeEvolutionConfig is supported.")
    end
end

end
