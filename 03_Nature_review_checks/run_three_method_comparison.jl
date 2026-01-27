using Printf
using DelimitedFiles

"""
Driver script: run Julia + TenPy methods + 1 exact reference on the same circuit and plot
(1) bond dimension growth and (2) <Z> expectation values at selected sites.

Methods:
- CircuitTDVP (Julia): DigitalTJM on a `CircuitLibrary.jl` circuit
- CircuitTEBD (Julia): DigitalTJM forced to TEBD for local + long-range evolution
- CircuitSRC  (Julia): DigitalTJM forced to SRC (randomized MPO×MPS application) for 2q / kq gates
- TenPy MPO zip-up / variational (Python): driven by gate-list exported from Julia circuit
- Qiskit exact (Python): driven by the same gate-list

All parameters can be adjusted below (or via ARGS using --key=value).

--------------------------------------------------------------------------------
ALL SUPPORTED FLAGS (with defaults)
--------------------------------------------------------------------------------

Core / output:
  --outdir=03_Nature_review_checks/results
  --tag_jl=circuitTDVP
  --tag_jl_tebd=circuitTEBD
  --tag_jl_src=circuitSRC
  --tag_zipup=tenpy_zipup
  --tag_var=tenpy_variational
  --tag_exact=qiskit_exact
    Note: the exact (Qiskit) reference is automatically skipped for L > 12.

Circuit selection:
  --circuit=Heisenberg
    Supported: Heisenberg | Ising | CZBrickwork | RZZPiOver2Brickwork | SingleLongRangeGate
  --L=8
  --steps=20
  --dt=0.05
  --periodic=true

Initial states:
  --state_jl=Neel       (Julia: zeros | ones | x+ | x- | y+ | y- | Neel | wall | basis)
  --state_py=neel       (TenPy/Qiskit: up | neel)

Measurement:
  --sites=1,4,8         (1-based site indices)

Truncation / bond cap:
  --chi_max=256
  --trunc=1e-12
  --trunc_mode=relative (relative | absolute)
    Note: TenPy uses relative discarded weight. Use `absolute` only for Julia-side legacy comparisons.

Julia circuit evolution backend:
  --jl_local_mode=TDVP       (TDVP | TEBD | BUG | SRC)
  --jl_longrange_mode=TDVP   (TDVP | TEBD | BUG | SRC)
  --jl_run_tebd=true         (true | false)
  --jl_run_src=false         (true | false)
  --jl_tdvp_truncation=during (during | after_window)
    Applies to TDVP window evolution; BUG reuses the same setting.
  --jl_bug_truncation=after_sweep (after_sweep | after_site)
    BUG-only: controls truncation granularity when truncating "during" evolution.
    - after_sweep: truncate once per BUG half-step sweep (default, cheaper)
    - after_site : truncate after each local BUG site update (tighter bond control)
  --warmup=true              (true | false)

Optional: run BOTH Julia methods (TDVP + BUG) and plot both:
  --jl_compare_bug=false     (true | false)
  --tag_jl_bug=circuitBUG

TenPy variational options:
  --min_sweeps=2
  --max_sweeps=10

Model parameters (depend on --circuit):
  Heisenberg:
    --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.0
  Ising:
    --J=1.0 --g=0.5
  SingleLongRangeGate:
    (see implementation below for exact parameters)

--------------------------------------------------------------------------------
COPY/PASTE COMMANDS (each run writes into results/experimentX/)
--------------------------------------------------------------------------------

Shell note (zsh/bash): use a single backslash `\` for line continuation and make sure it is the
last character on the line (no trailing spaces). Using `\\` will *not* continue the command.

Base (Heisenberg):
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl \
    --circuit=Heisenberg --L=8 --steps=20 --dt=0.05 --periodic=true \
    --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.0 \
    --sites=1,4,8 --state_jl=Neel --state_py=neel \
    --chi_max=256 --trunc=1e-16 --trunc_mode=relative \
    --jl_local_mode=TDVP --jl_longrange_mode=TDVP --warmup=true \
    --jl_run_tebd=true --tag_jl_tebd=circuitTEBD \
    --jl_tdvp_truncation=after_window \
    --min_sweeps=2 --max_sweeps=10 \
    --tag_jl=circuitTDVP --tag_zipup=tenpy_zipup --tag_var=tenpy_variational --tag_exact=qiskit_exact \
    --outdir=03_Nature_review_checks/results

Heisenberg: compare Julia TDVP vs Julia BUG2nd in the SAME run (both plotted)
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl \
    --circuit=Heisenberg --L=8 --steps=20 --dt=0.05 --periodic=true \
    --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.0 \
    --sites=1,4,8 --state_jl=Neel --state_py=neel \
    --chi_max=256 --trunc=1e-16 --trunc_mode=relative \
    --jl_local_mode=TDVP --jl_longrange_mode=TDVP \
    --jl_run_tebd=true --tag_jl_tebd=circuitTEBD \
    --jl_compare_bug=true --tag_jl=circuitTDVP --tag_jl_bug=circuitBUG \
    --outdir=03_Nature_review_checks/results

Heisenberg: TEBD-only (Julia) against TenPy/Qiskit (no additional Julia TDVP run)
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl \
    --circuit=Heisenberg --L=8 --steps=20 --dt=0.05 --periodic=true \
    --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.0 \
    --sites=1,4,8 --state_jl=Neel --state_py=neel \
    --chi_max=256 --trunc=1e-12 --trunc_mode=relative \
    --jl_run_tebd=false --jl_local_mode=TEBD --jl_longrange_mode=TEBD --tag_jl=circuitTEBD \
    --outdir=03_Nature_review_checks/results

Heisenberg: include SRC + BUG as additional Julia curves (plotted alongside TDVP/TEBD)
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl \
    --circuit=Heisenberg --L=8 --steps=20 --dt=0.05 --periodic=true \
    --Jx=1.0 --Jy=1.0 --Jz=1.0 --h=0.5 \
    --sites=1,4,8 --state_jl=Neel --state_py=neel \
    --chi_max=256 --trunc=1e-16 --trunc_mode=relative \
    --jl_tdvp_truncation=after_window \
    --jl_local_mode=TDVP --jl_longrange_mode=TDVP --jl_run_tebd=true --jl_run_src=true --jl_compare_bug=true \
    --tag_jl=circuitTDVP --tag_jl_tebd=circuitTEBD --tag_jl_src=circuitSRC --tag_jl_bug=circuitBUG \
    --outdir=03_Nature_review_checks/results

Legacy Julia truncation mode (NOTE: TenPy stays relative; use only for Julia-side comparisons):
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl \
    --circuit=Heisenberg --L=6 --steps=1 --dt=0.05 --sites=1,3,6 \
    --chi_max=256 --trunc=1e-12 --trunc_mode=absolute

Ising:
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl \
    --circuit=Ising --L=8 --steps=20 --dt=0.05 --periodic=true \
    --J=1.0 --g=0.5 \
    --sites=1,4,8 --chi_max=256 --trunc=1e-12 \
    --jl_local_mode=TDVP --jl_longrange_mode=TDVP \
    --jl_run_tebd=true --tag_jl_tebd=circuitTEBD

Ising: BUG2nd-only (Julia) against TenPy/Qiskit
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl \
    --circuit=Ising --L=8 --steps=20 --dt=0.05 --periodic=true \
    --J=1.0 --g=0.5 \
    --sites=1,4,8 --chi_max=256 --trunc=1e-12 \
    --jl_local_mode=BUG --jl_longrange_mode=BUG --tag_jl=circuitBUG

Brickwork examples:
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl --circuit=CZBrickwork --L=12 --steps=30 --sites=1,6,12 --chi_max=256 --trunc=1e-12
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl --circuit=RZZPiOver2Brickwork --L=12 --steps=30 --sites=1,6,12 --chi_max=256 --trunc=1e-12

Brickwork: compare TDVP vs BUG2nd (Julia) and include both in plots
  julia --project=. 03_Nature_review_checks/run_three_method_comparison.jl \
    --circuit=CZBrickwork --L=12 --steps=30 --sites=1,6,12 --chi_max=256 --trunc=1e-12 \
    --jl_local_mode=TDVP --jl_longrange_mode=TDVP --jl_compare_bug=true \
    --jl_run_tebd=true --tag_jl_tebd=circuitTEBD

Tips:
- `--trunc=0` disables discarded-weight truncation (still capped by `--chi_max`).
- `--warmup=false` includes compilation cost in CircuitTDVP wallclock (usually not desired).
- `--min_sweeps/--max_sweeps` affect TenPy variational MPO application convergence.
- `--jl_tdvp_truncation=after_window` can reduce per-gate truncation artifacts (at higher peak χ).
- `--jl_compare_bug=true` runs TDVP (tag `--tag_jl`) and BUG2nd (tag `--tag_jl_bug`) back-to-back and plots both.
- `--jl_run_src=true` runs an additional Julia SRC curve (tag `--tag_jl_src`) and includes it in plots.
- For large systems (L > 12), the script skips Qiskit exact automatically (runtime scales ~O(2^L) per gate).
"""

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
include("../src/BUG.jl")
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
using .BUGModule
end # YaqsLite

using .YaqsLite.GateLibrary
using .YaqsLite.MPSModule
using .YaqsLite.SimulationConfigs
using .YaqsLite.DigitalTJM: DigitalCircuit, TJMOptions, run_digital_tjm, add_gate!
using .YaqsLite.CircuitLibrary

include("gatelist_io.jl")
using .GateListIO: write_gatelist_csv

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

function _read_csv_matrix(path::AbstractString)
    header = split(strip(readline(path)), ",")
    data = readdlm(path, ',', Float64; skipstart=1)
    return header, data
end

function _read_obs(path::AbstractString)
    header, data = _read_csv_matrix(path)
    steps = Int.(data[:, 1])
    vals = data[:, 2:end]  # columns match header[2:end]
    return steps, header[2:end], vals
end

function _read_chi(path::AbstractString)
    header, data = _read_csv_matrix(path)
    steps = Int.(data[:, 1])
    chi = data[:, 2:end]
    return steps, header[2:end], chi
end

function _next_experiment_outdir(base_outdir::AbstractString)
    mkpath(base_outdir)
    max_id = 0
    for name in readdir(base_outdir)
        full = joinpath(base_outdir, name)
        isdir(full) || continue
        m = match(r"^experiment(\d+)$", name)
        m === nothing && continue
        n = parse(Int, m.captures[1])
        max_id = max(max_id, n)
    end
    new_dir = joinpath(base_outdir, "experiment$(max_id + 1)")
    mkpath(new_dir)
    return new_dir
end

function _has_sample_barrier(circ::DigitalCircuit)
    for g in circ.gates
        if g.op isa GateLibrary.Barrier && uppercase(g.op.label) == "SAMPLE_OBSERVABLES"
            return true
        end
    end
    return false
end

function _ensure_sample_barriers!(circ::DigitalCircuit)
    has = _has_sample_barrier(circ)
    if has
        return circ
    end
    # Minimal fallback: add a barrier at start and end so the pipeline still produces outputs.
    # For meaningful time-series, circuits should insert SAMPLE_OBSERVABLES markers at desired points.
    pushfirst!(circ.gates, YaqsLite.DigitalTJM.DigitalGate(GateLibrary.Barrier("SAMPLE_OBSERVABLES"), Int[], nothing))
    push!(circ.gates, YaqsLite.DigitalTJM.DigitalGate(GateLibrary.Barrier("SAMPLE_OBSERVABLES"), Int[], nothing))
    return circ
end

function _build_circuit_from_library(kv::Dict{String,String})
    name = get(kv, "circuit", "Heisenberg")

    if name == "Heisenberg"
        L = parse(Int, get(kv, "L", "8"))
        steps = parse(Int, get(kv, "steps", "20"))
        dt = parse(Float64, get(kv, "dt", "0.05"))
        periodic = lowercase(get(kv, "periodic", "true")) in ("1", "true", "yes", "y")
        Jx = parse(Float64, get(kv, "Jx", "1.0"))
        Jy = parse(Float64, get(kv, "Jy", "1.0"))
        Jz = parse(Float64, get(kv, "Jz", "1.0"))
        h = parse(Float64, get(kv, "h", "0.0"))
        circ = create_heisenberg_circuit(L, Jx, Jy, Jz, h, dt, steps; periodic=periodic)
        return circ
    elseif name == "Ising"
        L = parse(Int, get(kv, "L", "8"))
        steps = parse(Int, get(kv, "steps", "20"))
        dt = parse(Float64, get(kv, "dt", "0.05"))
        periodic = lowercase(get(kv, "periodic", "true")) in ("1", "true", "yes", "y")
        J = parse(Float64, get(kv, "J", "1.0"))
        g = parse(Float64, get(kv, "g", "0.5"))
        circ = create_ising_circuit(L, J, g, dt, steps; periodic=periodic)
        return circ
    elseif name == "CZBrickwork"
        L = parse(Int, get(kv, "L", "8"))
        steps = parse(Int, get(kv, "steps", "20"))
        periodic = lowercase(get(kv, "periodic", "false")) in ("1", "true", "yes", "y")
        circ = create_cz_brickwork_circuit(L, steps; periodic=periodic)
        return circ
    elseif name == "RZZPiOver2Brickwork"
        L = parse(Int, get(kv, "L", "8"))
        steps = parse(Int, get(kv, "steps", "20"))
        periodic = lowercase(get(kv, "periodic", "false")) in ("1", "true", "yes", "y")
        circ = create_rzz_pi_over_2_brickwork(L, steps; periodic=periodic)
        return circ
    elseif name == "SingleLongRangeGate"
        # Debug circuit: a single long-range 2-qubit gate (no other gates).
        L = parse(Int, get(kv, "L", "16"))
        gate = lowercase(get(kv, "gate", "rzz"))  # rzz|rxx|ryy|cx|cz|swap|cp
        theta = parse(Float64, get(kv, "theta", string(π / 4)))
        i = parse(Int, get(kv, "i", "1"))
        j = parse(Int, get(kv, "j", string(L)))
        (1 <= i <= L && 1 <= j <= L && i != j) || error("SingleLongRangeGate requires distinct i,j in 1..L")

        circ = DigitalCircuit(L)
        add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
        if gate == "rzz"
            add_gate!(circ, RzzGate(theta), [i, j])
        elseif gate == "rxx"
            add_gate!(circ, RxxGate(theta), [i, j])
        elseif gate == "ryy"
            add_gate!(circ, RyyGate(theta), [i, j])
        elseif gate == "cx"
            add_gate!(circ, CXGate(), [i, j])
        elseif gate == "cz"
            add_gate!(circ, CZGate(), [i, j])
        elseif gate == "swap"
            add_gate!(circ, SWAPGate(), [i, j])
        elseif gate == "cp"
            add_gate!(circ, CPhaseGate(theta), [i, j])
        else
            error("Unsupported gate=$gate for SingleLongRangeGate. Use rzz|rxx|ryy|cx|cz|swap|cp.")
        end
        add_gate!(circ, Barrier("SAMPLE_OBSERVABLES"), Int[])
        return circ
    else
        error("Unsupported --circuit=$name. Supported: Heisenberg, Ising, CZBrickwork, RZZPiOver2Brickwork, SingleLongRangeGate")
    end
end

function _run_circuit_tdvp!(; circ::DigitalCircuit, sites::Vector{Int}, chi_max::Int, trunc::Float64, outdir::String, tag::String,
                            state_jl::String, local_mode::Symbol, longrange_mode::Symbol,
                            tdvp_truncation_timing::Symbol=:during,
                            bug_truncation_granularity::Symbol=:after_sweep,
                            warmup::Bool=true)
    mkpath(outdir)

    circ = _ensure_sample_barriers!(circ)
    L = circ.num_qubits
    obs_list = [Observable("Z_$s", ZGate(), s) for s in sites]

    sim_params = TimeEvolutionConfig(obs_list, 1.0; dt=1.0, num_traj=1, sample_timesteps=true, max_bond_dim=chi_max, truncation_threshold=trunc)
    alg_options = TJMOptions(local_method=local_mode, long_range_method=longrange_mode,
                             tdvp_truncation_timing=tdvp_truncation_timing,
                             bug_truncation_granularity=bug_truncation_granularity)

    # Warm-up run to exclude compilation from timing.
    if warmup
        @printf("[circuitTDVP] warmup enabled: running short prefix to compile\n")
        # Use a short prefix of the circuit to avoid a full duplicate run being printed.
        # This still compiles the hot paths (gate application + measurement).
        n_pref = min(length(circ.gates), 50)
        circ_w = DigitalCircuit(L)
        circ_w.gates = circ.gates[1:n_pref]
        circ_w = _ensure_sample_barriers!(circ_w)

        # Use a smaller bond cap for warmup to keep it fast.
        warmup_chi = min(chi_max, 8)
        sim_params_w = TimeEvolutionConfig(obs_list, 1.0; dt=1.0, num_traj=1, sample_timesteps=true,
                                           max_bond_dim=warmup_chi, truncation_threshold=trunc)
        psi_w = MPS(L; state=state_jl)
        pad_bond_dimension!(psi_w, 2; noise_scale=0.0)
        run_digital_tjm(psi_w, circ_w, nothing, sim_params_w; alg_options=alg_options)
    end

    psi = MPS(L; state=state_jl)
    pad_bond_dimension!(psi, 2; noise_scale=0.0)

    t0 = time()
    _, results, bond_dims = run_digital_tjm(psi, circ, nothing, sim_params; alg_options=alg_options)
    wall_s_sim = time() - t0

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
    @printf("[circuitTDVP] wrote: %s\n", timing_path)
    @printf("[circuitTDVP] wall_s_sim=%.3f\n", wall_s_sim)

    return obs_path, chi_path, timing_path
end

function _python_plot!(; outdir::String, jl_tag::String, zip_tag::String, var_tag::String, exact_tag::String,
                       jl_tag_bug::Union{Nothing,String}=nothing,
                       jl_tag_tebd::Union{Nothing,String}=nothing,
                       jl_tag_src::Union{Nothing,String}=nothing)
    # Plot with system python to avoid PythonCall/CondaPkg initialization issues.
    code = """
import os, sys, csv

outdir = sys.argv[1]
jl_tag, zip_tag, var_tag, exact_tag = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
jl_bug = sys.argv[6] if len(sys.argv) > 6 else "none"
jl_tebd = sys.argv[7] if len(sys.argv) > 7 else "none"
jl_src = sys.argv[8] if len(sys.argv) > 8 else "none"
jl_bug = None if (jl_bug is None or jl_bug.lower() in ("none","null","")) else jl_bug
jl_tebd = None if (jl_tebd is None or jl_tebd.lower() in ("none","null","")) else jl_tebd
jl_src = None if (jl_src is None or jl_src.lower() in ("none","null","")) else jl_src
exact_tag = None if (exact_tag is None or str(exact_tag).lower() in ("none","null","")) else exact_tag

def read_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [[float(x) for x in row] for row in r]
    return header, rows

def read_timing(tag):
    path = os.path.join(outdir, f"{tag}_timing.csv")
    if not os.path.exists(path):
        return None
    h, rows = read_csv(path)
    if len(rows) < 1 or len(rows[0]) < 1:
        return None
    return float(rows[0][0])

def read_obs(tag):
    h, rows = read_csv(os.path.join(outdir, f"{tag}_obs.csv"))
    step = [int(r[0]) for r in rows]
    vals = [r[1:] for r in rows]
    return step, h[1:], vals

def read_chi(tag):
    h, rows = read_csv(os.path.join(outdir, f"{tag}_chi.csv"))
    step = [int(r[0]) for r in rows]
    vals = [r[1:] for r in rows]
    return step, h[1:], vals

def fmt_s(x):
    if x is None:
        return "n/a"
    if x >= 60.0:
        return f"{x/60.0:.2f} min"
    return f"{x:.2f} s"

step_jl, obs_cols, obs_jl = read_obs(jl_tag)
step_zip, _, obs_zip = read_obs(zip_tag)
step_var, _, obs_var = read_obs(var_tag)
step_ex, obs_ex = None, None
if exact_tag is not None:
    step_ex, _, obs_ex = read_obs(exact_tag)

_, chi_cols_jl, chi_jl = read_chi(jl_tag)
_, chi_cols_zip, chi_zip = read_chi(zip_tag)
_, chi_cols_var, chi_var = read_chi(var_tag)
chi_cols_ex, chi_ex = None, None
if exact_tag is not None:
    _, chi_cols_ex, chi_ex = read_chi(exact_tag)

t_jl = read_timing(jl_tag)
t_zip = read_timing(zip_tag)
t_var = read_timing(var_tag)
t_ex = read_timing(exact_tag) if exact_tag is not None else None

# Optional second Julia curve (BUG2nd)
obs_bug = None
chi_bug = None
t_bug = None
if jl_bug is not None:
    obs_path = os.path.join(outdir, f"{jl_bug}_obs.csv")
    chi_path = os.path.join(outdir, f"{jl_bug}_chi.csv")
    if os.path.exists(obs_path) and os.path.exists(chi_path):
        _, _, obs_bug = read_obs(jl_bug)
        _, _, chi_bug = read_chi(jl_bug)
        t_bug = read_timing(jl_bug)
    else:
        jl_bug = None


# Optional TEBD curve (Julia)
step_tebd = None
obs_tebd = None
chi_tebd = None
t_tebd = None
if jl_tebd is not None:
    obs_path = os.path.join(outdir, f"{jl_tebd}_obs.csv")
    chi_path = os.path.join(outdir, f"{jl_tebd}_chi.csv")
    if os.path.exists(obs_path) and os.path.exists(chi_path):
        step_tebd, _, obs_tebd = read_obs(jl_tebd)
        _, _, chi_tebd = read_chi(jl_tebd)
        t_tebd = read_timing(jl_tebd)
    else:
        jl_tebd = None

# Optional SRC curve (Julia)
step_src = None
obs_src = None
chi_src = None
t_src = None
if jl_src is not None:
    obs_path = os.path.join(outdir, f"{jl_src}_obs.csv")
    chi_path = os.path.join(outdir, f"{jl_src}_chi.csv")
    if os.path.exists(obs_path) and os.path.exists(chi_path):
        step_src, _, obs_src = read_obs(jl_src)
        _, _, chi_src = read_chi(jl_src)
        t_src = read_timing(jl_src)
    else:
        jl_src = None
def col_index(cols, name):
    try:
        return cols.index(name)
    except ValueError:
        raise RuntimeError(f"Missing column {name} in {cols}")

chi_max_jl = [row[0] for row in chi_jl]  # only chi_max
chi_max_tebd = None
if chi_tebd is not None:
    chi_max_tebd = [row[0] for row in chi_tebd]
chi_max_src = None
if chi_src is not None:
    chi_max_src = [row[0] for row in chi_src]
chi_max_zip = [row[col_index(chi_cols_zip, "chi_max")] for row in chi_zip]
chi_max_var = [row[col_index(chi_cols_var, "chi_max")] for row in chi_var]
chi_max_ex = None
if chi_ex is not None:
    chi_max_ex = [row[col_index(chi_cols_ex, "chi_max")] for row in chi_ex]
chi_max_bug = None
if jl_bug is not None:
    chi_max_bug = [row[0] for row in chi_bug]

lengths = [len(step_jl), len(step_zip), len(step_var)]
if step_ex is not None:
    lengths.append(len(step_ex))
if step_tebd is not None:
    lengths.append(len(step_tebd))
if step_src is not None:
    lengths.append(len(step_src))
nmin = min(lengths)
x = list(range(nmin))
chi_max_jl = chi_max_jl[:nmin]
if chi_max_tebd is not None:
    chi_max_tebd = chi_max_tebd[:nmin]
if chi_max_src is not None:
    chi_max_src = chi_max_src[:nmin]
chi_max_zip = chi_max_zip[:nmin]
chi_max_var = chi_max_var[:nmin]
if chi_max_ex is not None:
    chi_max_ex = chi_max_ex[:nmin]
if chi_max_bug is not None:
    chi_max_bug = chi_max_bug[:nmin]
obs_jl = obs_jl[:nmin]
if obs_tebd is not None:
    obs_tebd = obs_tebd[:nmin]
if obs_src is not None:
    obs_src = obs_src[:nmin]
obs_zip = obs_zip[:nmin]
obs_var = obs_var[:nmin]
if obs_ex is not None:
    obs_ex = obs_ex[:nmin]
if obs_bug is not None:
    obs_bug = obs_bug[:nmin]

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("matplotlib not available in system python") from e

# Bond dims
plt.figure(figsize=(10,4))
plt.plot(x, chi_max_jl, label=f"{jl_tag} (Julia) [{fmt_s(t_jl)}]", linewidth=2)
if chi_max_tebd is not None:
    plt.plot(x, chi_max_tebd, label=f"{jl_tebd} (Julia) [{fmt_s(t_tebd)}]", linewidth=2, linestyle="-.")
if chi_max_src is not None:
    plt.plot(x, chi_max_src, label=f"{jl_src} (Julia) [{fmt_s(t_src)}]", linewidth=2, linestyle=":")
if chi_max_bug is not None:
    plt.plot(x, chi_max_bug, label=f"BUG2nd (Julia) [{fmt_s(t_bug)}]", linewidth=2, linestyle="--")
plt.plot(x, chi_max_zip, label=f"TenPy MPO zip-up [{fmt_s(t_zip)}]", linewidth=2, linestyle="--")
plt.plot(x, chi_max_var, label=f"TenPy MPO variational [{fmt_s(t_var)}]", linewidth=2, linestyle=":")
if chi_max_ex is not None:
    plt.plot(x, chi_max_ex, label=f"Qiskit exact (chi N/A) [{fmt_s(t_ex)}]", linewidth=2, linestyle="-.", color="k", alpha=0.6)
plt.xlabel("Layer / step")
plt.ylabel("chi_max")
plt.title("Bond dimension growth")
plt.grid(True)
plt.legend()
bond_plot = os.path.join(outdir, "bond_dims_comparison.png")
plt.tight_layout()
plt.savefig(bond_plot)
plt.close()

# Observables
nsites = len(obs_cols)
fig, axes = plt.subplots(nsites, 1, figsize=(10, 3*nsites), sharex=True)
if nsites == 1:
    axes = [axes]

for i in range(nsites):
    ax = axes[i]
    y_jl = [r[i] for r in obs_jl]
    y_tebd = [r[i] for r in obs_tebd] if obs_tebd is not None else None
    y_src = [r[i] for r in obs_src] if obs_src is not None else None
    y_zip = [r[i] for r in obs_zip]
    y_var = [r[i] for r in obs_var]
    y_ex = [r[i] for r in obs_ex] if obs_ex is not None else None
    y_bug = [r[i] for r in obs_bug] if obs_bug is not None else None

    ax.plot(x, y_jl, label=f"{jl_tag} (Julia)", linewidth=2)
    if y_tebd is not None:
        ax.plot(x, y_tebd, label=f"{jl_tebd} (Julia)", linewidth=2, linestyle="-.")
    if y_src is not None:
        ax.plot(x, y_src, label=f"{jl_src} (Julia)", linewidth=2, linestyle=":")
    if y_bug is not None:
        ax.plot(x, y_bug, label="BUG2nd (Julia)", linewidth=2, linestyle="--")
    ax.plot(x, y_zip, label="TenPy zip-up", linewidth=2, linestyle="--")
    ax.plot(x, y_var, label="TenPy variational", linewidth=2, linestyle=":")
    if y_ex is not None:
        ax.plot(x, y_ex, label="Qiskit exact", linewidth=2, linestyle="-.", color="k", alpha=0.6)

    # Right y-axis: per-timepoint squared error vs exact (only if exact is available)
    ax2 = None
    if y_ex is not None:
        ax2 = ax.twinx()
        se_jl = [(abs(a - b)) ** 2 for a, b in zip(y_jl, y_ex)]
        se_tebd = [(abs(a - b)) ** 2 for a, b in zip(y_tebd, y_ex)] if y_tebd is not None else None
        se_src = [(abs(a - b)) ** 2 for a, b in zip(y_src, y_ex)] if y_src is not None else None
        se_bug = [(abs(a - b)) ** 2 for a, b in zip(y_bug, y_ex)] if y_bug is not None else None
        se_zip = [(abs(a - b)) ** 2 for a, b in zip(y_zip, y_ex)]
        se_var = [(abs(a - b)) ** 2 for a, b in zip(y_var, y_ex)]
        # Plot SE curves with distinct styles and keep CircuitTDVP on top (in case curves overlap).
        ax2.plot(x, se_zip, label="SE: TenPy zip-up", linewidth=1.3, linestyle="--", alpha=0.8, color="C1", zorder=2)
        ax2.plot(x, se_var, label="SE: TenPy variational", linewidth=1.3, linestyle=":", alpha=0.8, color="C2", zorder=3)
        ax2.plot(x, se_jl, label=f"SE: {jl_tag}", linewidth=1.6, linestyle="-.", alpha=0.9, color="C0", zorder=4)
        if se_tebd is not None:
            ax2.plot(x, se_tebd, label=f"SE: {jl_tebd}", linewidth=1.6, linestyle=":", alpha=0.9, color="C4", zorder=4)
        if se_src is not None:
            ax2.plot(x, se_src, label=f"SE: {jl_src}", linewidth=1.6, linestyle="-", alpha=0.9, color="C5", zorder=4)
        if se_bug is not None:
            ax2.plot(x, se_bug, label="SE: BUG2nd", linewidth=1.6, linestyle="--", alpha=0.9, color="C3", zorder=5)
        ax2.set_ylabel("Squared error vs exact")

        # If SE spans many orders of magnitude, use log-scale.
        all_se = se_jl + se_zip + se_var
        if se_tebd is not None:
            all_se = all_se + se_tebd
        if se_src is not None:
            all_se = all_se + se_src
        if se_bug is not None:
            all_se = all_se + se_bug
        all_se = [v for v in all_se if v > 0.0]
        if all_se:
            mn, mx = min(all_se), max(all_se)
            if mx / mn > 1.0e6:
                ax2.set_yscale("log")

    ax.set_ylabel(obs_cols[i])
    ax.grid(True)
    # Combined legend (left + right axes)
    h1, l1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize="small")
    else:
        ax.legend(h1, l1, loc="upper right", fontsize="small")
axes[-1].set_xlabel("Layer / step")
fig.suptitle("Observable comparison")
fig.tight_layout()
obs_plot = os.path.join(outdir, "observables_comparison.png")
fig.savefig(obs_plot)
plt.close(fig)

print("Saved plots:")
print(bond_plot)
print(obs_plot)
"""
    jl_bug = jl_tag_bug === nothing ? "none" : jl_tag_bug
    jl_tebd = jl_tag_tebd === nothing ? "none" : jl_tag_tebd
    jl_src = jl_tag_src === nothing ? "none" : jl_tag_src
    cmd = `python3 -c $code $outdir $jl_tag $zip_tag $var_tag $exact_tag $jl_bug $jl_tebd $jl_src`
    run(cmd)
end

function main()
    kv = _parse_kv_args(ARGS)

    # --- Adjustable parameters ---
    circuit = get(kv, "circuit", "Heisenberg")
    L = parse(Int, get(kv, "L", "8"))  # some circuit constructors infer L from kv; keep for sites parsing
    chi_max = parse(Int, get(kv, "chi_max", "256"))
    trunc = parse(Float64, get(kv, "trunc", "1e-12"))
    trunc_mode = lowercase(get(kv, "trunc_mode", "relative"))  # relative (default) | absolute (legacy Julia)
    sites = _parse_sites(get(kv, "sites", "1,4,8"), L)
    state_py = get(kv, "state_py", "neel")  # TenPy choices: up|neel
    state_jl = get(kv, "state_jl", "Neel")  # Julia choices: zeros|Neel|...
    jl_local_mode = Symbol(get(kv, "jl_local_mode", "TDVP"))
    jl_longrange_mode = Symbol(get(kv, "jl_longrange_mode", "TDVP"))
    jl_tdvp_truncation_raw = lowercase(get(kv, "jl_tdvp_truncation", "during"))  # during|after_window
    jl_tdvp_truncation = if jl_tdvp_truncation_raw in ("during", "in", "insweep", "in_sweep")
        :during
    elseif jl_tdvp_truncation_raw in ("after", "post", "after_window", "post_window")
        :after_window
    else
        error("Unknown --jl_tdvp_truncation=$jl_tdvp_truncation_raw. Use during|after_window.")
    end
    warmup = lowercase(get(kv, "warmup", "true")) in ("1", "true", "yes", "y")

    jl_bug_trunc_raw = lowercase(get(kv, "jl_bug_truncation", "after_sweep"))  # after_sweep|after_site
    jl_bug_trunc = if jl_bug_trunc_raw in ("after_sweep", "sweep", "after")
        :after_sweep
    elseif jl_bug_trunc_raw in ("after_site", "site", "per_site")
        :after_site
    else
        error("Unknown --jl_bug_truncation=$jl_bug_trunc_raw. Use after_sweep|after_site.")
    end

    base_outdir = get(kv, "outdir", "03_Nature_review_checks/results")
    outdir = _next_experiment_outdir(base_outdir)

    @printf("Running 4-way comparison (CircuitLibrary → gatelist): circuit=%s L=%d chi_max=%d sites=%s\n",
            circuit, L, chi_max, join(string.(sites), ","))
    @printf("CircuitTDVP modes: local=%s longrange=%s (warmup=%s)\n",
            String(jl_local_mode), String(jl_longrange_mode), string(warmup))
    @printf("CircuitTDVP truncation timing (2-site TDVP): %s\n", String(jl_tdvp_truncation))
    @printf("BUG truncation granularity (when truncating during): %s\n", String(jl_bug_trunc))
    if trunc_mode in ("absolute", "abs")
        @printf("Truncation (Julia): absolute_discarded_weight=%.3g  [legacy]\n", trunc)
        @printf("Note: TenPy uses relative discarded weight; for strict matching use --trunc_mode=relative.\n")
        trunc_jl = -abs(trunc)  # negative => legacy absolute mode in Julia internals
    elseif trunc_mode in ("relative", "rel")
        @printf("Truncation: relative_discarded_weight=%.3g\n", trunc)
        trunc_jl = abs(trunc)
    else
        error("Unknown --trunc_mode=$trunc_mode. Use relative|absolute.")
    end
    @printf("Writing outputs to: %s\n", outdir)

    # --- Build circuit in Julia (CircuitLibrary) ---
    circ = _build_circuit_from_library(kv)
    # update L for downstream consistency
    L = circ.num_qubits

    # --- Export gate-list (shared by TenPy + Qiskit exact) ---
    gatelist_path = joinpath(outdir, "circuit_gatelist.csv")
    write_gatelist_csv(_ensure_sample_barriers!(circ), gatelist_path)
    @printf("Wrote gate-list: %s\n", gatelist_path)

    # --- Run CircuitTDVP (Julia) in-process ---
    jl_tag = get(kv, "tag_jl", "circuitTDVP")
    _run_circuit_tdvp!(; circ=circ, sites=sites, chi_max=chi_max, trunc=trunc_jl, outdir=outdir, tag=jl_tag,
                       state_jl=state_jl, local_mode=jl_local_mode, longrange_mode=jl_longrange_mode,
                       tdvp_truncation_timing=jl_tdvp_truncation,
                       bug_truncation_granularity=jl_bug_trunc,
                       warmup=warmup)

    # Additional: run TEBD (Julia) and include in plots.
    jl_run_tebd = lowercase(get(kv, "jl_run_tebd", "true")) in ("1", "true", "yes", "y")
    jl_tag_tebd = nothing
    if jl_run_tebd
        jl_tag_tebd = get(kv, "tag_jl_tebd", "circuitTEBD")
        @printf("[jl_tebd] running additional Julia TEBD (tag=%s)\n", jl_tag_tebd)
        _run_circuit_tdvp!(; circ=circ, sites=sites, chi_max=chi_max, trunc=trunc_jl, outdir=outdir, tag=jl_tag_tebd,
                           state_jl=state_jl, local_mode=:TEBD, longrange_mode=:TEBD,
                           tdvp_truncation_timing=jl_tdvp_truncation,
                           bug_truncation_granularity=jl_bug_trunc,
                           warmup=warmup)
    end

    # Additional: run SRC (Julia) and include in plots.
    jl_run_src = lowercase(get(kv, "jl_run_src", "false")) in ("1", "true", "yes", "y")
    jl_tag_src = nothing
    if jl_run_src
        jl_tag_src = get(kv, "tag_jl_src", "circuitSRC")
        @printf("[jl_src] running additional Julia SRC (tag=%s)\n", jl_tag_src)
        _run_circuit_tdvp!(; circ=circ, sites=sites, chi_max=chi_max, trunc=trunc_jl, outdir=outdir, tag=jl_tag_src,
                           state_jl=state_jl, local_mode=:SRC, longrange_mode=:SRC,
                           tdvp_truncation_timing=jl_tdvp_truncation,
                           bug_truncation_granularity=jl_bug_trunc,
                           warmup=warmup)
    end

    # Optional: also run BUG2nd (Julia) and include in plots.
    jl_compare_bug = lowercase(get(kv, "jl_compare_bug", "false")) in ("1", "true", "yes", "y")
    jl_tag_bug = nothing
    if jl_compare_bug
        jl_tag_bug = get(kv, "tag_jl_bug", "circuitBUG")
        @printf("[jl_compare_bug] running additional Julia BUG2nd (tag=%s)\n", jl_tag_bug)
        _run_circuit_tdvp!(; circ=circ, sites=sites, chi_max=chi_max, trunc=trunc_jl, outdir=outdir, tag=jl_tag_bug,
                           state_jl=state_jl, local_mode=:BUG, longrange_mode=:BUG,
                           tdvp_truncation_timing=jl_tdvp_truncation,
                           bug_truncation_granularity=jl_bug_trunc,
                           warmup=false)
    end

    # --- Run TenPy zip-up from gate-list ---
    py_tenpy = joinpath(@__DIR__, "tenpy_mpo_from_gatelist.py")
    py_zip_tag = get(kv, "tag_zipup", "tenpy_zipup")
    cmd_zip = `python3 $py_tenpy --gatelist=$gatelist_path --method=zip_up --chi-max=$chi_max --trunc=$trunc --sites=$(join(sites, ",")) --state=$state_py --outdir=$outdir --tag=$py_zip_tag`
    @printf("-> %s\n", string(cmd_zip))
    run(cmd_zip)

    # --- Run TenPy variational from gate-list ---
    py_var_tag = get(kv, "tag_var", "tenpy_variational")
    # Variational MPO application needs enough sweeps to converge when truncation is effectively off.
    min_sweeps = parse(Int, get(kv, "min_sweeps", "2"))
    max_sweeps = parse(Int, get(kv, "max_sweeps", "10"))
    cmd_var = `python3 $py_tenpy --gatelist=$gatelist_path --method=variational --chi-max=$chi_max --trunc=$trunc --sites=$(join(sites, ",")) --state=$state_py --outdir=$outdir --tag=$py_var_tag --min-sweeps=$min_sweeps --max-sweeps=$max_sweeps --max-trunc-err=none`
    @printf("-> %s\n", string(cmd_var))
    run(cmd_var)

    # --- Run Qiskit exact (statevector) from gate-list ---
    py_ex = joinpath(@__DIR__, "qiskit_exact_from_gatelist.py")
    py_ex_tag = get(kv, "tag_exact", "qiskit_exact")
    if L > 12
        @printf("[qiskit_exact] skipping exact reference for L=%d (>12)\n", L)
        py_ex_tag = "none"
    else
        cmd_ex = `python3 $py_ex --gatelist=$gatelist_path --sites=$(join(sites, ",")) --state=$state_py --outdir=$outdir --tag=$py_ex_tag`
        @printf("-> %s\n", string(cmd_ex))
        run(cmd_ex)
    end

    # --- Load results ---
    jl_obs = joinpath(outdir, "$(jl_tag)_obs.csv")
    jl_chi = joinpath(outdir, "$(jl_tag)_chi.csv")
    zip_obs = joinpath(outdir, "$(py_zip_tag)_obs.csv")
    zip_chi = joinpath(outdir, "$(py_zip_tag)_chi.csv")
    var_obs = joinpath(outdir, "$(py_var_tag)_obs.csv")
    var_chi = joinpath(outdir, "$(py_var_tag)_chi.csv")
    ex_obs = joinpath(outdir, "$(py_ex_tag)_obs.csv")
    ex_chi = joinpath(outdir, "$(py_ex_tag)_chi.csv")

    _python_plot!(; outdir=outdir, jl_tag=jl_tag, jl_tag_bug=jl_tag_bug, jl_tag_tebd=jl_tag_tebd, jl_tag_src=jl_tag_src,
                  zip_tag=py_zip_tag, var_tag=py_var_tag, exact_tag=py_ex_tag)
end

main()

