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
    initialize(state, noise_model, sim_params)

Prepare the initial sampling MPS Phi(0) by applying a half time step of dissipation
followed by a stochastic process step.
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
    step_through(state, hamiltonian, noise_model, sim_params)

Perform a single time step evolution: TDVP -> Dissipation -> Stochastic Process.
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
    sample(phi, hamiltonian, noise_model, sim_params, results, j)

Evolve a copy of the sampling MPS, apply dissipation/noise, and measure.
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

function evaluate_observables!(psi::MPS, sim_params::TimeEvolutionConfig, results::Matrix{Float64}, time_idx::Int)
    for (i, obs) in enumerate(sim_params.observables)
        val = real(expect(psi, obs))
        results[i, time_idx] = val
    end
end

function evaluate_observables!(psi::MPS, sim_params::TimeEvolutionConfig, results::Matrix{Float64})
    # For single-column results (no time index)
    for (i, obs) in enumerate(sim_params.observables)
        val = real(expect(psi, obs))
        results[i, 1] = val
    end
end

"""
    analog_tjm_2(args)

Run a single trajectory using 2nd order TJM.
args: (traj_idx, initial_state, noise_model, sim_params, hamiltonian)
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
    analog_tjm_1(args)

Run a single trajectory using 1st order TJM.
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
