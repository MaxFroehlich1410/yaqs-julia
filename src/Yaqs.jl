module Yaqs

module Timing
using Base.Threads
using Printf

export enable_timing!, set_timing_print_each_call!, reset_timing!, print_timing_summary!, begin_scope!, end_scope!, @t

const _ENABLE_TIMING = Ref(false)
const _TIMING_TOP = Ref(20)
const _TIMING_PRINT_EACH_CALL = Ref(true)

mutable struct TimingStats
    times_ns::Dict{Symbol, UInt64}
    counts::Dict{Symbol, Int}
end

TimingStats() = TimingStats(Dict{Symbol, UInt64}(), Dict{Symbol, Int}())

@inline function _timing_add!(ts::TimingStats, key::Symbol, dt_ns::UInt64)
    ts.times_ns[key] = get(ts.times_ns, key, UInt64(0)) + dt_ns
    ts.counts[key] = get(ts.counts, key, 0) + 1
    return nothing
end

@inline function _timing_merge!(dst::TimingStats, src::TimingStats)
    for (k, v) in src.times_ns
        dst.times_ns[k] = get(dst.times_ns, k, UInt64(0)) + v
    end
    for (k, c) in src.counts
        dst.counts[k] = get(dst.counts, k, 0) + c
    end
    return nothing
end

const _ACTIVE_STATS = Ref{Vector{Union{Nothing, TimingStats}}}(Vector{Union{Nothing, TimingStats}}(undef, 0))

function __init__()
    v = Vector{Union{Nothing, TimingStats}}(undef, Base.Threads.maxthreadid())
    fill!(v, nothing)
    _ACTIVE_STATS[] = v
    return nothing
end

@inline function _active_stats()
    v = _ACTIVE_STATS[]
    tid = threadid()
    if tid > length(v)
        # In case module was precompiled with fewer threads.
        v2 = Vector{Union{Nothing, TimingStats}}(undef, Base.Threads.maxthreadid())
        fill!(v2, nothing)
        _ACTIVE_STATS[] = v2
        v = v2
    end
    return v[tid]
end

const _GLOBAL_TIMING = TimingStats()

"""
    enable_timing!(flag::Bool=true)

Enable timing collection (thread-local, aggregated into a global summary).
"""
enable_timing!(flag::Bool=true) = (_ENABLE_TIMING[] = flag)

"""
    set_timing_print_each_call!(flag::Bool=true)

If `true`, print a timing summary for every completed scope (e.g. each `run_digital_tjm` call).
If `false`, only the global summary is kept (use `print_timing_summary!`).
"""
set_timing_print_each_call!(flag::Bool=true) = (_TIMING_PRINT_EACH_CALL[] = flag)

"""
    reset_timing!()

Clear accumulated (global) timing statistics.
"""
function reset_timing!()
    empty!(_GLOBAL_TIMING.times_ns)
    empty!(_GLOBAL_TIMING.counts)
    return nothing
end

function _print_timing_summary(ts::TimingStats; header::AbstractString="Timing summary", top::Int=_TIMING_TOP[])
    total_ns = UInt64(0)
    @inbounds for v in values(ts.times_ns)
        total_ns += v
    end

    pairs = collect(ts.times_ns)
    sort!(pairs; by = p -> p[2], rev = true)

    @printf "\n\t%s (total %.6f s)\n" header (total_ns / 1e9)
    nshow = min(top, length(pairs))
    for i in 1:nshow
        key, tns = pairs[i]
        c = get(ts.counts, key, 0)
        ms_per = c > 0 ? (tns / 1e6) / c : 0.0
        frac = total_ns > 0 ? 100.0 * (tns / float(total_ns)) : 0.0
        @printf "\t  %-36s %10.4f s  (%5.1f%%)  %8d calls  %9.3f ms/call\n" String(key) (tns / 1e9) frac c ms_per
    end
    return nothing
end

"""
    print_timing_summary!(; header="Timing summary", top=_TIMING_TOP[])

Print (and keep) the currently accumulated global timing summary.
"""
function print_timing_summary!(; header::AbstractString="Timing summary", top::Int=_TIMING_TOP[])
    _print_timing_summary(_GLOBAL_TIMING; header=header, top=top)
    return nothing
end

"""
    begin_scope!() -> Union{Nothing,TimingStats}

Start a timing scope on the current thread (used to collect nested `@t` timings).
"""
function begin_scope!()
    if !_ENABLE_TIMING[]
        return nothing
    end
    ts = TimingStats()
    _ACTIVE_STATS[][threadid()] = ts
    return ts
end

"""
    end_scope!(ts; header="Timing scope")

End a timing scope, merging into the global summary and optionally printing.
"""
function end_scope!(ts::Union{Nothing, TimingStats}; header::AbstractString="Timing scope")
    _ACTIVE_STATS[][threadid()] = nothing
    if ts === nothing
        return nothing
    end
    _timing_merge!(_GLOBAL_TIMING, ts)
    if _TIMING_PRINT_EACH_CALL[]
        _print_timing_summary(ts; header=header)
    end
    return nothing
end

macro t(key, ex)
    return quote
        local _ts = Timing._active_stats()
        if _ts === nothing
            $(esc(ex))
        else
            local _t0 = time_ns()
            local _val = $(esc(ex))
            Timing._timing_add!(_ts, $(esc(key)), UInt64(time_ns() - _t0))
            _val
        end
    end
end

end # module Timing

include("GateLibrary.jl")
include("Decompositions.jl")
include("MPS.jl")
include("MPO.jl")
include("SimulationConfigs.jl")
include("Algorithms.jl")
include("BUG.jl")
include("Noise.jl")
include("StochasticProcess.jl")
include("Dissipation.jl")
include("AnalogTJM.jl")
include("DigitalTJM.jl")
include("Simulator.jl")
include("CircuitIngestion.jl")
include("CircuitLibrary.jl")

using .GateLibrary
using .Decompositions
using .MPSModule
using .MPOModule
using .SimulationConfigs
using .Algorithms
using .BUGModule
using .NoiseModule
using .StochasticProcessModule
using .DissipationModule
using .AnalogTJM
using .DigitalTJM
using .Simulator
using .CircuitIngestion
using .CircuitLibrary

export GateLibrary, Decompositions, MPSModule, MPOModule, SimulationConfigs, Algorithms, BUGModule, NoiseModule, StochasticProcessModule, DissipationModule, AnalogTJM, Simulator, DigitalTJM, CircuitIngestion, CircuitLibrary

end
