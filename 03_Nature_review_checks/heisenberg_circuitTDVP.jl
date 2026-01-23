using LinearAlgebra
using Printf

# Minimal Yaqs subset without PythonCall/CircuitIngestion (avoids CondaPkg init)
module YaqsLite
module Timing
export enable_timing!, set_timing_print_each_call!, reset_timing!, print_timing_summary!, begin_scope!, end_scope!, @t
enable_timing!(::Bool=true) = nothing
set_timing_print_each_call!(::Bool=true) = nothing
reset_timing!() = nothing
print_timing_summary!(; header::AbstractString="Timing summary", top::Int=20) = nothing
begin_scope!() = nothing
end_scope!(::Any; header::AbstractString="Timing scope") = nothing
macro t(_key, ex)
    return esc(ex)
end
end # Timing

include("../src/GateLibrary.jl")
include("../src/Decompositions.jl")
include("../src/MPS.jl")
include("../src/MPO.jl")
include("../src/SimulationConfigs.jl")
include("../src/Algorithms.jl")
include("../src/Noise.jl")
include("../src/StochasticProcess.jl")
include("../src/Dissipation.jl")
include("../src/AnalogTJM.jl")
include("../src/DigitalTJM.jl")
include("../src/CircuitLibrary.jl")

using .GateLibrary
using .MPSModule
using .SimulationConfigs
using .DigitalTJM
using .CircuitLibrary
end # YaqsLite

using .YaqsLite.CircuitLibrary
using .YaqsLite.GateLibrary
using .YaqsLite.MPSModule
using .YaqsLite.SimulationConfigs
using .YaqsLite.DigitalTJM: TJMOptions, run_digital_tjm

using DelimitedFiles

"""
Run noise-free CircuitTDVP (DigitalTJM with TDVP windows) for a periodic Heisenberg circuit
and write CSV outputs compatible with the TenPy scripts.

Outputs (in --outdir):
- <tag>_obs.csv : columns step,Z_site<k>...
- <tag>_chi.csv : columns step,chi_max
"""

function _parse_kv_args(args::Vector{String})
    d = Dict{String,String}()
    for a in args
        if startswith(a, "--")
            if occursin("=", a)
                k, v = split(a[3:end], "=", limit=2)
                d[k] = v
            else
                d[a[3:end]] = "true"
            end
        end
    end
    return d
end

function _parse_sites(s::AbstractString, L::Int)
    parts = split(s, ",")
    sites = Int[]
    for p in parts
        t = strip(p)
        isempty(t) && continue
        x = parse(Int, t)
        if 1 <= x <= L
            push!(sites, x)
        end
    end
    isempty(sites) && error("No valid sites in --sites (1..$L).")
    return sites
end

function run_heisenberg_circuitTDVP!(;
    L::Int=8,
    steps::Int=20,
    dt::Float64=0.05,
    Jx::Float64=1.0,
    Jy::Float64=1.0,
    Jz::Float64=1.0,
    h_field::Float64=0.0,
    periodic::Bool=true,
    max_bond_dim::Int=256,
    initial_state::String="Neel",
    sites::Vector{Int}=[1, 4, 8],
    outdir::String="03_Nature_review_checks/results",
    tag::String="circuitTDVP",
    local_mode::Symbol=:TDVP,
    longrange_mode::Symbol=:TDVP,
    warmup::Bool=true,
)
    mkpath(outdir)

    circ = create_heisenberg_circuit(L, Jx, Jy, Jz, h_field, dt, steps; periodic=periodic)
    obs_list = [Observable("Z_$s", ZGate(), s) for s in sites]

    # Note: dt is irrelevant for DigitalTJM gate-application; we keep dt=1.0 so "times" length is stable,
    # but sampling is determined by SAMPLE_OBSERVABLES barriers in the circuit.
    sim_params = TimeEvolutionConfig(obs_list, 1.0; dt=1.0, num_traj=1, sample_timesteps=true, max_bond_dim=max_bond_dim)
    alg_options = TJMOptions(local_method=local_mode, long_range_method=longrange_mode)

    psi = MPS(L; state=initial_state)
    pad_bond_dimension!(psi, 2; noise_scale=0.0)

    @printf("[circuitTDVP] L=%d steps=%d dt=%g periodic=%s init=%s max_bond=%d\n", L, steps, dt, string(periodic), initial_state, max_bond_dim)

    # Warm-up run to exclude compilation from the timed measurement.
    if warmup
        steps_w = min(1, steps)
        circ_w = create_heisenberg_circuit(L, Jx, Jy, Jz, h_field, dt, steps_w; periodic=periodic)
        psi_w = MPS(L; state=initial_state)
        pad_bond_dimension!(psi_w, 2; noise_scale=0.0)
        run_digital_tjm(psi_w, circ_w, nothing, sim_params; alg_options=alg_options)
    end

    t0 = time()
    _, results, bond_dims = run_digital_tjm(psi, circ, nothing, sim_params; alg_options=alg_options)
    wall_s_sim = time() - t0

    # results: (num_obs, num_meas), bond_dims: (num_meas,)
    num_meas = size(results, 2)
    obs_path = joinpath(outdir, "$(tag)_obs.csv")
    chi_path = joinpath(outdir, "$(tag)_chi.csv")
    timing_path = joinpath(outdir, "$(tag)_timing.csv")

    open(obs_path, "w") do io
        header = ["step"; ["Z_site$(s)" for s in sites]...]
        println(io, join(header, ","))
        for t in 1:num_meas
            step = t - 1
            vals = [real(results[i, t]) for i in 1:length(sites)]
            println(io, join([string(step); string.(vals)...], ","))
        end
    end

    open(chi_path, "w") do io
        println(io, "step,chi_max")
        for t in 1:length(bond_dims)
            step = t - 1
            println(io, string(step), ",", string(bond_dims[t]))
        end
    end

    open(timing_path, "w") do io
        println(io, "wall_s_sim")
        println(io, wall_s_sim)
    end

    @printf("[circuitTDVP] wrote: %s\n", obs_path)
    @printf("[circuitTDVP] wrote: %s\n", chi_path)
    @printf("[circuitTDVP] wall_s_sim=%.3f\n", wall_s_sim)
    @printf("[circuitTDVP] wrote: %s\n", timing_path)
    return obs_path, chi_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    kv = _parse_kv_args(ARGS)
    L = parse(Int, get(kv, "L", "8"))
    steps = parse(Int, get(kv, "steps", "20"))
    dt = parse(Float64, get(kv, "dt", "0.05"))
    max_bond_dim = parse(Int, get(kv, "chi_max", get(kv, "max_bond_dim", "256")))
    outdir = get(kv, "outdir", "03_Nature_review_checks/results")
    tag = get(kv, "tag", "circuitTDVP")
    state = get(kv, "state", "Neel")
    periodic = lowercase(get(kv, "periodic", "true")) in ("1", "true", "yes", "y")
    sites = _parse_sites(get(kv, "sites", "1,4,8"), L)
    local_mode = Symbol(get(kv, "local_mode", "TDVP"))
    longrange_mode = Symbol(get(kv, "longrange_mode", "TDVP"))
    warmup = lowercase(get(kv, "warmup", "true")) in ("1", "true", "yes", "y")

    run_heisenberg_circuitTDVP!(;
        L=L,
        steps=steps,
        dt=dt,
        periodic=periodic,
        max_bond_dim=max_bond_dim,
        initial_state=state,
        sites=sites,
        outdir=outdir,
        tag=tag,
        local_mode=local_mode,
        longrange_mode=longrange_mode,
        warmup=warmup,
    )
end

