module AnalogTJM

using LinearAlgebra
using ..MPSModule
using ..MPOModule
using ..NoiseModule
using ..SimulationConfigs
using ..Algorithms
using ..DissipationModule
using ..StochasticProcessModule
using ..Decompositions

export initialize, step_through, sample, analog_tjm_1, analog_tjm_2

"""
Prepare the initial sampling MPS for analog TJM trajectories.

This applies a half-step of dissipation followed by a full-step stochastic process to produce
the initial sampling state Phi(0). The input state is updated in-place and returned for chaining.

Args:
    state (MPS): State to initialize in-place.
    noise_model (Union{NoiseModel, Nothing}): Noise model or `nothing` to skip noise.
    sim_params (TimeEvolutionConfig): Simulation parameters including the time step.

Returns:
    MPS: The initialized sampling state.
"""
function initialize(state::MPS{T}, noise_model::Union{NoiseModel{T}, Nothing}, sim_params::TimeEvolutionConfig) where T
    dt = sim_params.dt
    
    # Apply Dissipation (dt/2)
    apply_dissipation(state, noise_model, dt / 2, sim_params)
    
    # Stochastic Process (dt)
    if !isnothing(noise_model)
        stochastic_process!(state, noise_model, dt, sim_params)
    end
    
    return state
end

"""
Advance the sampling state by one analog TJM step.

This performs coherent evolution via TDVP, applies dissipation, and then applies the stochastic
process when a noise model is present. The input state is updated in-place.

Args:
    state (MPS): State to evolve in-place.
    hamiltonian (MPO): Hamiltonian MPO for coherent evolution.
    noise_model (Union{NoiseModel, Nothing}): Noise model or `nothing` to skip noise.
    sim_params (TimeEvolutionConfig): Simulation parameters including time step.

Returns:
    MPS: The evolved state.
"""
function step_through(state::MPS{T}, hamiltonian::MPO{T}, noise_model::Union{NoiseModel{T}, Nothing}, sim_params::TimeEvolutionConfig) where T
    dt = sim_params.dt
    
    # 1. Coherent Evolution (TDVP)
    two_site_tdvp!(state, hamiltonian, sim_params)
    
    # 2. Dissipation
    apply_dissipation(state, noise_model, dt, sim_params)
    
    # 3. Stochastic Process
    if !isnothing(noise_model)
        stochastic_process!(state, noise_model, dt, sim_params)
    end
    
    return state
end

"""
Sample observables from a noisy evolution of the sampling state.

This deep-copies the sampling MPS, performs a TDVP evolution and dissipation/noise step, then
measures the configured observables into the results array.

Args:
    phi (MPS): Sampling state to copy and evolve.
    hamiltonian (MPO): Hamiltonian MPO for coherent evolution.
    noise_model (Union{NoiseModel, Nothing}): Noise model or `nothing` to skip noise.
    sim_params (TimeEvolutionConfig): Simulation parameters including observables.
    results (Matrix{Float64}): Output array to store observable values.
    j (Int): Time index at which to store results.

Returns:
    Nothing: Results are written into `results`.
"""
function sample(phi::MPS{T}, hamiltonian::MPO{T}, noise_model::Union{NoiseModel{T}, Nothing}, 
                sim_params::TimeEvolutionConfig, results::Matrix{Float64}, j::Int) where T
    psi = deepcopy(phi)
    dt = sim_params.dt
    
    # Evolution Step (Dissipation dt/2)
    two_site_tdvp!(psi, hamiltonian, sim_params)
    apply_dissipation(psi, noise_model, dt / 2, sim_params)
    
    if !isnothing(noise_model)
        stochastic_process!(psi, noise_model, dt, sim_params)
    end
    
    # Measure
    if sim_params.sample_timesteps
        evaluate_observables!(psi, sim_params, results, j)
    else
        evaluate_observables!(psi, sim_params, results)
    end
end

"""
Evaluate observables at a specific time index.

This computes the real part of each observable expectation value and stores it into the provided
results matrix at the given time index.

Args:
    psi (MPS): State whose observables are evaluated.
    sim_params (TimeEvolutionConfig): Simulation parameters containing observables.
    results (Matrix{Float64}): Output matrix storing observable values.
    time_idx (Int): Column index to write results into.

Returns:
    Nothing: Results are written into `results`.
"""
function evaluate_observables!(psi::MPS, sim_params::TimeEvolutionConfig, results::Matrix{Float64}, time_idx::Int)
    for (i, obs) in enumerate(sim_params.observables)
        val = real(expect(psi, obs))
        results[i, time_idx] = val
    end
end

"""
Evaluate observables for single-column results.

This computes the real part of each observable expectation value and stores it in the first column
of the results matrix, used when only the final timestep is sampled.

Args:
    psi (MPS): State whose observables are evaluated.
    sim_params (TimeEvolutionConfig): Simulation parameters containing observables.
    results (Matrix{Float64}): Output matrix storing observable values.

Returns:
    Nothing: Results are written into `results`.
"""
function evaluate_observables!(psi::MPS, sim_params::TimeEvolutionConfig, results::Matrix{Float64})
    # For single-column results (no time index)
    for (i, obs) in enumerate(sim_params.observables)
        val = real(expect(psi, obs))
        results[i, 1] = val
    end
end

"""
Run a single analog TJM trajectory using second-order splitting.

This initializes the sampling state, evolves it through time with second-order updates, and records
observables at the requested timesteps. It is designed for use in parallel trajectory sampling.

Args:
    args: Tuple `(traj_idx, initial_state, noise_model, sim_params, hamiltonian)`.

Returns:
    Matrix{Float64}: Observable values for the trajectory.
"""
function analog_tjm_2(args)
    (traj_idx, initial_state, noise_model, sim_params, hamiltonian) = args
    
    # Removed StochasticProcess pre-calculation. 
    # Directly pass noise_model to functions.

    state = deepcopy(initial_state)
    num_obs = length(sim_params.observables)
    num_steps = length(sim_params.times)
    
    if sim_params.sample_timesteps
        results = zeros(Float64, num_obs, num_steps)
        evaluate_observables!(state, sim_params, results, 1)
    else
        results = zeros(Float64, num_obs, 1)
    end
    
    # Initialize Phi(0)
    phi = initialize(state, noise_model, sim_params)
    
    if sim_params.sample_timesteps && num_steps > 1
        sample(phi, hamiltonian, noise_model, sim_params, results, 2)
    end
    
    for j in 3:num_steps
        phi = step_through(phi, hamiltonian, noise_model, sim_params)
        
        if sim_params.sample_timesteps || j == num_steps
            sample(phi, hamiltonian, noise_model, sim_params, results, j)
        end
    end
    
    return results
end

"""
Run a single analog TJM trajectory using first-order splitting.

This evolves the state through time with one-site updates and applies dissipation and stochastic
process steps as configured, recording observables at requested timesteps.

Args:
    args: Tuple `(traj_idx, initial_state, noise_model, sim_params, hamiltonian)`.

Returns:
    Matrix{Float64}: Observable values for the trajectory.
"""
function analog_tjm_1(args)
    (traj_idx, initial_state, noise_model, sim_params, hamiltonian) = args
    
    # Removed StochasticProcess pre-calculation.

    state = deepcopy(initial_state)
    num_obs = length(sim_params.observables)
    num_steps = length(sim_params.times)
    
    if sim_params.sample_timesteps
        results = zeros(Float64, num_obs, num_steps)
        evaluate_observables!(state, sim_params, results, 1)
    else
        results = zeros(Float64, num_obs, 1)
    end
    
    for j in 2:num_steps
        two_site_tdvp!(state, hamiltonian, sim_params)
        
        if !isnothing(noise_model)
            apply_dissipation(state, noise_model, sim_params.dt, sim_params)
            stochastic_process!(state, noise_model, sim_params.dt, sim_params)
        end
        
        if sim_params.sample_timesteps
            evaluate_observables!(state, sim_params, results, j)
        elseif j == num_steps
            evaluate_observables!(state, sim_params, results)
        end
    end
    
    return results
end

end
