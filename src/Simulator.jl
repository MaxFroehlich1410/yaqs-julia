module Simulator

using Base.Threads
using ..MPSModule
using ..MPOModule
using ..SimulationConfigs
using ..NoiseModule
using ..AnalogTJM

export run, available_cpus

function available_cpus()
    if haskey(ENV, "SLURM_CPUS_ON_NODE")
        return parse(Int, ENV["SLURM_CPUS_ON_NODE"])
    else
        return Sys.CPU_THREADS
    end
end

function _run_analog(initial_state::MPS, operator::MPO, sim_params::TimeEvolutionConfig, noise_model::Union{NoiseModel, Nothing}; parallel::Bool=true)
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

    # Arguments generator (implicit)
    # We iterate 1:num_traj

    # Parallel Loop
    # Ensure thread safety for trajectories writing.
    # trajectories is (num_traj, time_steps) or (num_traj, 1).
    # Writing to slice [i, :] is safe as long as i is unique per thread.

    if parallel && sim_params.num_traj > 1
        # Progress Counter
        counter = Atomic{Int}(0)
        total = sim_params.num_traj
        
        Threads.@threads for i in 1:sim_params.num_traj
            args = (i, initial_state, noise_model, sim_params, operator)
            # Result is Matrix (num_obs, num_times)
            # Julia backend returns Matrix
            result = backend(args)
            
            for (obs_idx, obs) in enumerate(sim_params.observables)
                # obs.trajectories is (num_traj, num_times)
                # result[obs_idx, :] is (num_times)
                # Assign row
                obs.trajectories[i, :] = result[obs_idx, :]
            end
            
            # Update and Print Progress
            c = atomic_add!(counter, 1) + 1
            if c % 10 == 0 || c == total
                # Print to stdout, use \r to overwrite line if possible or just simple log
                # Printing from threads can be messy, usually done by main thread or careful print
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

function run(initial_state::MPS, operator, sim_params, noise_model=nothing; parallel::Bool=true)
    # State must start in B normalization (Right Canonical sites 2..L)
    # Use explicit string "B" to match Python's simulator.py logic conceptually,
    # though MPS.jl normalize! defaults to right-canonical sweep.
    MPSModule.normalize!(initial_state; form="B") 
    
    if isa(sim_params, TimeEvolutionConfig)
        # Analog
        # operator must be MPO
        if !isa(operator, MPO)
             error("Analog simulation requires an MPO operator.")
        end
        _run_analog(initial_state, operator, sim_params, noise_model; parallel=parallel)
    else
        error("Only Analog Simulation (TimeEvolutionConfig) is supported in this optimized version.")
    end
end

end

