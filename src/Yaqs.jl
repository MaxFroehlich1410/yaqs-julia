module Yaqs

module Timing
using Base.Threads
using Printf

export enable_timing!, set_timing_print_each_call!, reset_timing!, print_timing_summary!, begin_scope!, end_scope!, @t

const _ENABLE_TIMING = Ref(false)
const _TIMING_TOP = Ref(20)
const _TIMING_PRINT_EACH_CALL = Ref(true)

"""
Store timing statistics for named code regions.

This tracks total elapsed time and call counts per timing key and is used to aggregate results
across nested timing scopes.

Args:
    times_ns (Dict{Symbol, UInt64}): Accumulated time in nanoseconds per key.
    counts (Dict{Symbol, Int}): Call counts per key.

Returns:
    TimingStats: Timing statistics container.
"""
mutable struct TimingStats
    times_ns::Dict{Symbol, UInt64}
    counts::Dict{Symbol, Int}
end

"""
Create an empty TimingStats container.

This initializes empty dictionaries for time and count accumulation.

Args:
    None

Returns:
    TimingStats: Empty timing statistics container.
"""
TimingStats() = TimingStats(Dict{Symbol, UInt64}(), Dict{Symbol, Int}())

"""
Accumulate timing data for a key.

This adds the elapsed time and increments the call count for the specified timing key.

Args:
    ts (TimingStats): Timing statistics container to update.
    key (Symbol): Timing key identifier.
    dt_ns (UInt64): Elapsed time in nanoseconds.

Returns:
    Nothing: The statistics are updated in-place.
"""
@inline function _timing_add!(ts::TimingStats, key::Symbol, dt_ns::UInt64)
    ts.times_ns[key] = get(ts.times_ns, key, UInt64(0)) + dt_ns
    ts.counts[key] = get(ts.counts, key, 0) + 1
    return nothing
end

"""
Merge timing statistics from one container into another.

This adds per-key times and counts from `src` into `dst`.

Args:
    dst (TimingStats): Destination statistics to update.
    src (TimingStats): Source statistics to merge from.

Returns:
    Nothing: The destination statistics are updated in-place.
"""
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

"""
Initialize thread-local timing statistics storage.

This sizes the per-thread active statistics vector to the current thread count and clears it.

Args:
    None

Returns:
    Nothing: Thread-local storage is updated in-place.
"""
function __init__()
    v = Vector{Union{Nothing, TimingStats}}(undef, Base.Threads.maxthreadid())
    fill!(v, nothing)
    _ACTIVE_STATS[] = v
    return nothing
end

"""
Get the active timing stats for the current thread.

This retrieves the thread-local TimingStats object if one is active, resizing storage if the
thread count has changed.

Args:
    None

Returns:
    Union{Nothing, TimingStats}: Active stats for the current thread, or `nothing`.
"""
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
Enable or disable timing collection.

This toggles timing collection for the current process and controls whether `@t` records timings.

Args:
    flag (Bool): Whether to enable timing collection.

Returns:
    Bool: The updated timing enable flag.
"""
enable_timing!(flag::Bool=true) = (_ENABLE_TIMING[] = flag)

"""
Configure whether timing summaries print after each scope.

This controls automatic printing of timing summaries when a scope ends, keeping global summaries
available regardless of the print setting.

Args:
    flag (Bool): Whether to print a summary after each scope.

Returns:
    Bool: The updated print-each-call flag.
"""
set_timing_print_each_call!(flag::Bool=true) = (_TIMING_PRINT_EACH_CALL[] = flag)

"""
Clear accumulated global timing statistics.

This resets the global timing dictionaries used to summarize performance.

Args:
    None

Returns:
    Nothing: Global timing statistics are cleared.
"""
function reset_timing!()
    empty!(_GLOBAL_TIMING.times_ns)
    empty!(_GLOBAL_TIMING.counts)
    return nothing
end

"""
Print a timing summary for a given TimingStats object.

This computes total time, sorts entries by elapsed time, and prints a formatted summary for the
top keys.

Args:
    ts (TimingStats): Timing statistics to print.
    header (AbstractString): Header text for the summary.
    top (Int): Maximum number of entries to print.

Returns:
    Nothing: Summary is printed to stdout.
"""
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
Print the global timing summary.

This prints the accumulated global timing statistics without clearing them.

Args:
    header (AbstractString): Header text for the summary.
    top (Int): Maximum number of entries to print.

Returns:
    Nothing: Summary is printed to stdout.
"""
function print_timing_summary!(; header::AbstractString="Timing summary", top::Int=_TIMING_TOP[])
    _print_timing_summary(_GLOBAL_TIMING; header=header, top=top)
    return nothing
end

"""
Begin a timing scope on the current thread.

This creates a thread-local TimingStats object used to accumulate `@t` timings within the scope.

Args:
    None

Returns:
    Union{Nothing, TimingStats}: Timing stats object if timing is enabled, otherwise `nothing`.
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
End a timing scope and merge results into the global summary.

This clears the thread-local active stats, merges them into the global summary, and optionally
prints a per-scope summary.

Args:
    ts (Union{Nothing, TimingStats}): Timing stats from `begin_scope!`.
    header (AbstractString): Header text for the per-scope summary.

Returns:
    Nothing: Timing data is merged and optionally printed.
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

"""
Time a code block and record the duration under a key.

This macro wraps an expression, measuring elapsed time when timing is enabled and accumulating
the result in the current scope's TimingStats.

Args:
    key (Symbol): Timing key identifier.
    ex: Expression to execute and time.

Returns:
    Any: Result of the wrapped expression.
"""
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
include("Noise.jl")
include("StochasticProcess.jl")
include("Dissipation.jl")
include("AnalogTJM.jl")
include("CircuitTJM.jl")
include("Simulator.jl")
include("CircuitIngestion.jl")
include("CircuitLibrary.jl")

using .GateLibrary
using .Decompositions
using .MPSModule
using .MPOModule
using .SimulationConfigs
using .Algorithms
using .NoiseModule
using .StochasticProcessModule
using .DissipationModule
using .AnalogTJM
using .CircuitTJM
using .Simulator
using .CircuitIngestion
using .CircuitLibrary

export GateLibrary, Decompositions, MPSModule, MPOModule, SimulationConfigs, Algorithms, NoiseModule, StochasticProcessModule, DissipationModule, AnalogTJM, Simulator, CircuitTJM, CircuitIngestion, CircuitLibrary

end
