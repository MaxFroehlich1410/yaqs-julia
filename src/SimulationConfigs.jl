module SimulationConfigs

using ..GateLibrary
using ..MPSModule
using LinearAlgebra

export Observable, TimeEvolutionConfig, MeasurementConfig, StrongMeasurementConfig
export initialize!, aggregate_trajectories!, aggregate_measurements!, expect

# --- Observable ---

"""
    Observable{O<:AbstractOperator}

Represents an observable to be measured on the quantum state.

# Fields
- `op::O`: The operator (gate) to measure.
- `sites::Vector{Int}`: The site(s) to measure on.
- `results::Vector{Float64}`: Aggregated results (e.g. mean).
- `trajectories::Matrix{ComplexF64}`: Per-trajectory data.
"""
mutable struct Observable{O<:AbstractOperator}
    name::String
    op::O
    sites::Vector{Int}
    results::Vector{Float64}
    trajectories::Matrix{ComplexF64}
    
    function Observable(name::String, op::O, sites::Union{Int, Vector{Int}}) where O <: AbstractOperator
        s = isa(sites, Int) ? [sites] : sites
        new{O}(name, op, s, Float64[], Matrix{ComplexF64}(undef, 0, 0))
    end
end

"""
    expect(psi::MPS, obs::Observable)

Compute the expectation value of the observable on the MPS.
Currently supports single-site observables.
"""
function expect(psi::MPS, obs::Observable)
    # Get matrix from GateLibrary
    op_mat = matrix(obs.op)
    
    if length(obs.sites) == 1
        # Single site
        site = obs.sites[1]
        return real(local_expect(psi, op_mat, site))
    else
        error("Multi-site expectation not yet implemented in Julia port.")
    end
end


# --- Simulation Configurations ---

abstract type AbstractSimConfig end

"""
    TimeEvolutionConfig (AnalogSimParams)

Configuration for time evolution simulations (TEBD/TDVP).
"""
mutable struct TimeEvolutionConfig <: AbstractSimConfig
    observables::Vector{<:Observable}
    total_time::Float64
    dt::Float64
    times::Vector{Float64}
    num_traj::Int
    max_bond_dim::Int
    min_bond_dim::Int
    truncation_threshold::Float64
    sample_timesteps::Bool
    
    function TimeEvolutionConfig(observables::Vector{<:Observable}, total_time::Float64;
                                 dt::Float64=0.1,
                                 num_traj::Int=1000,
                                 max_bond_dim::Int=4096,
                                 min_bond_dim::Int=2,
                                 truncation_threshold::Float64=1e-9,
                                 sample_timesteps::Bool=true)
        
        times = collect(0.0:dt:total_time)
        new(observables, total_time, dt, times, num_traj, max_bond_dim, min_bond_dim, truncation_threshold, sample_timesteps)
    end
end

"""
    MeasurementConfig (WeakSimParams)

Configuration for weak measurement simulations (Shots).
"""
mutable struct MeasurementConfig <: AbstractSimConfig
    shots::Int
    max_bond_dim::Int
    min_bond_dim::Int
    truncation_threshold::Float64
    measurements::Vector{Union{Dict{Int, Int}, Nothing}} # List of shot results
    results::Dict{Int, Int} # Aggregated
    
    function MeasurementConfig(shots::Int;
                               max_bond_dim::Int=4096,
                               min_bond_dim::Int=2,
                               truncation_threshold::Float64=1e-9)
        measurements = Vector{Union{Dict{Int, Int}, Nothing}}(nothing, shots)
        new(shots, max_bond_dim, min_bond_dim, truncation_threshold, measurements, Dict{Int, Int}())
    end
end

"""
    StrongMeasurementConfig (StrongSimParams)

Configuration for strong simulation (Trajectories).
"""
mutable struct StrongMeasurementConfig <: AbstractSimConfig
    observables::Vector{<:Observable}
    num_traj::Int
    max_bond_dim::Int
    min_bond_dim::Int
    truncation_threshold::Float64
    
    function StrongMeasurementConfig(observables::Vector{<:Observable};
                                     num_traj::Int=1000,
                                     max_bond_dim::Int=4096,
                                     min_bond_dim::Int=2,
                                     truncation_threshold::Float64=1e-9)
        new(observables, num_traj, max_bond_dim, min_bond_dim, truncation_threshold)
    end
end

# --- Initialization & Aggregation Logic ---

"""
    initialize!(obs::Observable, config::AbstractSimConfig)

Initialize results/trajectories buffers for an observable based on config.
"""
function initialize!(obs::Observable, config::TimeEvolutionConfig)
    if config.sample_timesteps
        obs.trajectories = zeros(ComplexF64, config.num_traj, length(config.times))
        obs.results = zeros(Float64, length(config.times))
    else
        obs.trajectories = zeros(ComplexF64, config.num_traj, 1)
        obs.results = zeros(Float64, 1)
    end
end

function initialize!(obs::Observable, config::StrongMeasurementConfig)
    obs.trajectories = zeros(ComplexF64, config.num_traj, 1)
    obs.results = zeros(Float64, 1)
end

"""
    aggregate_trajectories!(config::TimeEvolutionConfig)

Compute mean of trajectories for all observables.
"""
function aggregate_trajectories!(config::TimeEvolutionConfig)
    for obs in config.observables
        if !isempty(obs.trajectories)
            # Mean over trajectories (dim 1)
            # obs.trajectories is (num_traj, num_times)
            # mean -> (num_times)
            # Julia `mean` requires Statistics, or manual
            # We can use sum / N
            
            # Manual mean:
            sum_traj = sum(real(obs.trajectories), dims=1) # (1, num_times)
            obs.results = vec(sum_traj) ./ config.num_traj
        end
    end
end

function aggregate_trajectories!(config::StrongMeasurementConfig)
    for obs in config.observables
        if !isempty(obs.trajectories)
            sum_traj = sum(real(obs.trajectories), dims=1)
            obs.results = vec(sum_traj) ./ config.num_traj
        end
    end
end

"""
    aggregate_measurements!(config::MeasurementConfig)

Aggregate shot results.
"""
function aggregate_measurements!(config::MeasurementConfig)
    total_counts = Dict{Int, Int}()
    for shot_res in config.measurements
        if !isnothing(shot_res) # Handle potential Nothing if not filled
            for (k, v) in shot_res
                total_counts[k] = get(total_counts, k, 0) + v
            end
        end
    end
    config.results = total_counts
end

end # module

