"""
    BenchmarkIO

I/O utilities for the benchmark suite:
- CSV manifest (one row per run)
- Timeseries data (JLD2-like via Serialization)
- JSON metadata per run
"""
module BenchmarkIO

using Printf
using Dates
using Serialization

export ManifestRow, write_manifest_header!, write_manifest_row!,
       save_timeseries, save_metadata, TimeseriesData

# ── Manifest ─────────────────────────────────────────────────────────

struct ManifestRow
    exp_id::String
    model::String
    N::Int
    T::Float64
    dt::Float64
    method::String
    pairing::String
    trunc_mode::String
    chi_fixed::Int           # 0 if not applicable
    svd_threshold::Float64   # 0.0 if not applicable
    runtime_seconds::Float64
    err_infidelity_T::Float64
    err_maxZ_T::Float64
    err_energy_T::Float64
    norm_drift_T::Float64
    chi_max_over_time::Int
    is_pareto::Int           # -1 = not applicable, 0 = false, 1 = true
    seed::Int
    timestamp::String
    git_commit::String
end

const MANIFEST_HEADER = join([
    "exp_id", "model", "N", "T", "dt", "method", "pairing", "trunc_mode",
    "chi_fixed", "svd_threshold",
    "runtime_seconds",
    "err_infidelity_T", "err_maxZ_T", "err_energy_T", "norm_drift_T",
    "chi_max_over_time",
    "is_pareto",
    "seed", "timestamp", "git_commit"
], ",")

function write_manifest_header!(io::IO)
    println(io, MANIFEST_HEADER)
    flush(io)
end

function write_manifest_row!(io::IO, r::ManifestRow)
    @printf(io, "%s,%s,%d,%.6g,%.6g,%s,%s,%s,%d,%.6g,%.6f,%.10e,%.10e,%.10e,%.10e,%d,%d,%d,%s,%s\n",
            r.exp_id, r.model, r.N, r.T, r.dt, r.method, r.pairing, r.trunc_mode,
            r.chi_fixed, r.svd_threshold,
            r.runtime_seconds,
            r.err_infidelity_T, r.err_maxZ_T, r.err_energy_T, r.norm_drift_T,
            r.chi_max_over_time,
            r.is_pareto,
            r.seed, r.timestamp, r.git_commit)
    flush(io)
end

function write_manifest_row!(filepath::String, r::ManifestRow; append::Bool=true)
    need_header = !isfile(filepath) || filesize(filepath) == 0
    open(filepath, append ? "a" : "w") do io
        if need_header
            write_manifest_header!(io)
        end
        write_manifest_row!(io, r)
    end
end

# ── Timeseries ───────────────────────────────────────────────────────

struct TimeseriesData
    t_grid::Vector{Float64}
    z_expect::Matrix{Float64}     # (N, n_times)
    energy::Vector{Float64}       # (n_times,)
    norm_vals::Vector{Float64}    # (n_times,)
    chi_max::Vector{Int}          # (n_times,)
end

function save_timeseries(filepath::String, ts::TimeseriesData)
    serialize(filepath, ts)
end

# ── Metadata ─────────────────────────────────────────────────────────

function save_metadata(filepath::String, meta::Dict{String,Any})
    open(filepath, "w") do io
        println(io, "{")
        keys_sorted = sort(collect(keys(meta)))
        for (idx, k) in enumerate(keys_sorted)
            v = meta[k]
            vstr = if v isa AbstractString
                "\"$(escape_string(v))\""
            elseif v isa Number
                string(v)
            elseif v isa AbstractVector
                "[" * join(string.(v), ", ") * "]"
            else
                "\"$(escape_string(string(v)))\""
            end
            sep = idx < length(keys_sorted) ? "," : ""
            println(io, "  \"$k\": $vstr$sep")
        end
        println(io, "}")
    end
end

"""
    machine_info() -> Dict{String,Any}

Collect basic machine info for metadata.
"""
function machine_info()
    return Dict{String,Any}(
        "hostname" => gethostname(),
        "nthreads" => Threads.nthreads(),
        "julia_version" => string(VERSION),
        "os" => string(Sys.KERNEL),
        "cpu" => Sys.CPU_NAME,
        "word_size" => Sys.WORD_SIZE,
    )
end

end # module BenchmarkIO
