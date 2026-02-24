module SimulationConfigs

using ..GateLibrary
using ..MPSModule
using LinearAlgebra

export Observable, TimeEvolutionConfig, MeasurementConfig, StrongMeasurementConfig, AbstractSimConfig
export initialize!, aggregate_trajectories!, aggregate_measurements!, expect

# --- Observable ---

"""
Represent a measurable observable on a quantum state.

This stores the operator, target sites, and buffers for per-trajectory data and aggregated results.
It is used by simulation configurations to track measurements across trajectories.

Args:
    name (String): Observable name.
    op (AbstractOperator): Operator to measure.
    sites (Union{Int, Vector{Int}}): Target site indices.

Returns:
    Observable{O}: Observable container with initialized buffers.
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
Compute the expectation value of an observable on an MPS.

This dispatches to single-site or two-site expectation routines based on the observable's site list
and returns the real part of the expectation value.

Args:
    psi (MPS): State to evaluate.
    obs (Observable): Observable to measure.

Returns:
    Float64: Expectation value of the observable.

Raises:
    ErrorException: If the observable acts on more than two sites.
"""
function expect(psi::MPS, obs::Observable)
    # Get matrix from GateLibrary
    op_mat = matrix(obs.op)
    
    if length(obs.sites) == 1
        # Single site
        site = obs.sites[1]
        return real(local_expect(psi, op_mat, site))
    elseif length(obs.sites) == 2
        s1, s2 = sort(obs.sites)
        return real(local_expect_two_site(psi, op_mat, s1, s2))
    else
        error("Multi-site expectation not yet implemented in Julia port.")
    end
end


# --- Simulation Configurations ---

"""
Abstract supertype for simulation configuration objects.

This provides a common parent for time-evolution and measurement configuration structs.

Args:
    None

Returns:
    AbstractSimConfig: Abstract configuration type.
"""
abstract type AbstractSimConfig end

"""
Configuration for time evolution simulations.

This stores observables, time stepping parameters, and truncation settings for TEBD/TDVP runs,
including whether to sample at intermediate timesteps.

Args:
    observables (Vector{Observable}): Observables to record.
    total_time (Float64): Total evolution time.
    dt (Float64): Time step size.
    num_traj (Int): Number of trajectories to simulate.
    max_bond_dim (Int): Maximum bond dimension.
    min_bond_dim (Int): Minimum bond dimension.
    truncation_threshold (Float64): Truncation threshold.
    sample_timesteps (Bool): Whether to sample at each timestep.
    order (Int): Trotter order.

Returns:
    TimeEvolutionConfig: Time-evolution configuration object.
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
    order::Int
    
    function TimeEvolutionConfig(observables::Vector{<:Observable}, total_time::Float64;
                                 dt::Float64=0.1,
                                 num_traj::Int=1000,
                                 max_bond_dim::Int=4096,
                                 min_bond_dim::Int=2,
                                 truncation_threshold::Float64=1e-9,
                                 sample_timesteps::Bool=true,
                                 order::Int=2)
        
        times = collect(0.0:dt:total_time)
        new(observables, total_time, dt, times, num_traj, max_bond_dim, min_bond_dim, truncation_threshold, sample_timesteps, order)
    end
end

"""
Configuration for weak measurement simulations.

This stores shot counts and truncation settings for sampling-based measurements, along with
per-shot and aggregated measurement results.

Args:
    shots (Int): Number of measurement shots.
    max_bond_dim (Int): Maximum bond dimension.
    min_bond_dim (Int): Minimum bond dimension.
    truncation_threshold (Float64): Truncation threshold.

Returns:
    MeasurementConfig: Measurement configuration object.
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
Configuration for strong measurement simulations.

This stores observables and truncation settings for trajectory-based simulations, optionally
sampling after each circuit layer.

Args:
    observables (Vector{Observable}): Observables to record.
    num_traj (Int): Number of trajectories to simulate.
    max_bond_dim (Int): Maximum bond dimension.
    min_bond_dim (Int): Minimum bond dimension.
    truncation_threshold (Float64): Truncation threshold.
    sample_layers (Bool): Whether to sample after each layer.

Returns:
    StrongMeasurementConfig: Strong measurement configuration object.
"""
mutable struct StrongMeasurementConfig <: AbstractSimConfig
    observables::Vector{<:Observable}
    num_traj::Int
    max_bond_dim::Int
    min_bond_dim::Int
    truncation_threshold::Float64
    sample_layers::Bool
    
    function StrongMeasurementConfig(observables::Vector{<:Observable};
                                     num_traj::Int=1000,
                                     max_bond_dim::Int=4096,
                                     min_bond_dim::Int=2,
                                     truncation_threshold::Float64=1e-9,
                                     sample_layers::Bool=false)
        new(observables, num_traj, max_bond_dim, min_bond_dim, truncation_threshold, sample_layers)
    end
end

# --- Initialization & Aggregation Logic ---

"""
Initialize observable buffers for time-evolution simulations.

This allocates trajectory and result buffers sized to the number of timesteps and trajectories,
respecting whether intermediate sampling is enabled.

Args:
    obs (Observable): Observable to initialize.
    config (TimeEvolutionConfig): Time-evolution configuration.

Returns:
    Nothing: Observable buffers are updated in-place.
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

"""
Initialize observable buffers for strong measurement simulations.

This allocates trajectory and result buffers sized to the number of layers (if sampling per layer)
or a single measurement otherwise.

Args:
    obs (Observable): Observable to initialize.
    config (StrongMeasurementConfig): Strong measurement configuration.
    num_layers (Int): Number of circuit layers for sizing when sampling per layer.

Returns:
    Nothing: Observable buffers are updated in-place.
"""
function initialize!(obs::Observable, config::StrongMeasurementConfig; num_layers::Int=0)
    # If sample_layers is true, we need space for each layer (+ initial state)
    steps = config.sample_layers ? (num_layers + 1) : 1
    obs.trajectories = zeros(ComplexF64, config.num_traj, steps)
    obs.results = zeros(Float64, steps)
end

"""
Aggregate trajectory results for time-evolution simulations.

This computes the mean of observable trajectories across all trajectories and stores the results
in each observable's `results` field.

Args:
    config (TimeEvolutionConfig): Configuration containing observables to aggregate.

Returns:
    Nothing: Observable results are updated in-place.
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

"""
Aggregate trajectory results for strong measurement simulations.

This computes the mean of observable trajectories across all trajectories and stores the results
in each observable's `results` field.

Args:
    config (StrongMeasurementConfig): Configuration containing observables to aggregate.

Returns:
    Nothing: Observable results are updated in-place.
"""
function aggregate_trajectories!(config::StrongMeasurementConfig)
    for obs in config.observables
        if !isempty(obs.trajectories)
            sum_traj = sum(real(obs.trajectories), dims=1)
            obs.results = vec(sum_traj) ./ config.num_traj
        end
    end
end

"""
Aggregate shot results for weak measurement simulations.

This sums per-shot count dictionaries into a single aggregated results dictionary.

Args:
    config (MeasurementConfig): Configuration containing per-shot measurements.

Returns:
    Nothing: Aggregated results are stored in `config.results`.
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
