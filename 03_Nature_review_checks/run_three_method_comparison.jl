using Printf
using DelimitedFiles

"""
Driver script: run 4 noise-free methods on the same circuit and plot
(1) bond dimension growth and (2) <Z> expectation values at selected sites.

Methods:
- CircuitTDVP (Julia): `heisenberg_circuitTDVP.jl`
- TenPy MPO zip-up (Python): `heisenberg16_mpo_zipup.py`
- TenPy MPO variational (Python): `heisenberg16_mpo_variational.py`
- Qiskit exact (Python): `qiskit_exact_heisenberg.py`

All parameters can be adjusted below (or via ARGS using --key=value).
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

function _python_plot!(; outdir::String, jl_tag::String, zip_tag::String, var_tag::String, exact_tag::String)
    # Plot with system python to avoid PythonCall/CondaPkg initialization issues.
    code = """
import os, sys, csv

outdir = sys.argv[1]
jl_tag, zip_tag, var_tag, exact_tag = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

def read_csv(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [[float(x) for x in row] for row in r]
    return header, rows

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

step_jl, obs_cols, obs_jl = read_obs(jl_tag)
step_zip, _, obs_zip = read_obs(zip_tag)
step_var, _, obs_var = read_obs(var_tag)
step_ex, _, obs_ex = read_obs(exact_tag)

_, chi_cols_jl, chi_jl = read_chi(jl_tag)
_, chi_cols_zip, chi_zip = read_chi(zip_tag)
_, chi_cols_var, chi_var = read_chi(var_tag)
_, chi_cols_ex, chi_ex = read_chi(exact_tag)

def col_index(cols, name):
    try:
        return cols.index(name)
    except ValueError:
        raise RuntimeError(f"Missing column {name} in {cols}")

chi_max_jl = [row[0] for row in chi_jl]  # only chi_max
chi_max_zip = [row[col_index(chi_cols_zip, "chi_max")] for row in chi_zip]
chi_max_var = [row[col_index(chi_cols_var, "chi_max")] for row in chi_var]
chi_max_ex = [row[col_index(chi_cols_ex, "chi_max")] for row in chi_ex]

nmin = min(len(step_jl), len(step_zip), len(step_var), len(step_ex))
x = list(range(nmin))
chi_max_jl = chi_max_jl[:nmin]
chi_max_zip = chi_max_zip[:nmin]
chi_max_var = chi_max_var[:nmin]
chi_max_ex = chi_max_ex[:nmin]
obs_jl = obs_jl[:nmin]
obs_zip = obs_zip[:nmin]
obs_var = obs_var[:nmin]
obs_ex = obs_ex[:nmin]

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("matplotlib not available in system python") from e

# Bond dims
plt.figure(figsize=(10,4))
plt.plot(x, chi_max_jl, label="CircuitTDVP (Julia)", linewidth=2)
plt.plot(x, chi_max_zip, label="TenPy MPO zip-up", linewidth=2, linestyle="--")
plt.plot(x, chi_max_var, label="TenPy MPO variational", linewidth=2, linestyle=":")
plt.plot(x, chi_max_ex, label="Qiskit exact (chi N/A)", linewidth=2, linestyle="-.", color="k", alpha=0.6)
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
    ax.plot(x, [r[i] for r in obs_jl], label="CircuitTDVP (Julia)", linewidth=2)
    ax.plot(x, [r[i] for r in obs_zip], label="TenPy zip-up", linewidth=2, linestyle="--")
    ax.plot(x, [r[i] for r in obs_var], label="TenPy variational", linewidth=2, linestyle=":")
    ax.plot(x, [r[i] for r in obs_ex], label="Qiskit exact", linewidth=2, linestyle="-.", color="k", alpha=0.6)
    ax.set_ylabel(obs_cols[i])
    ax.grid(True)
    ax.legend(loc="upper right", fontsize="small")
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
    cmd = `python3 -c $code $outdir $jl_tag $zip_tag $var_tag $exact_tag`
    run(cmd)
end

function main()
    kv = _parse_kv_args(ARGS)

    # --- Adjustable parameters ---
    circuit = get(kv, "circuit", "Heisenberg")  # currently implemented: Heisenberg
    L = parse(Int, get(kv, "L", "16"))
    steps = parse(Int, get(kv, "steps", "20"))
    dt = parse(Float64, get(kv, "dt", "0.05"))
    chi_max = parse(Int, get(kv, "chi_max", "256"))
    sites = _parse_sites(get(kv, "sites", "1,8,16"), L)
    state_py = get(kv, "state_py", "neel")  # TenPy choices: up|neel
    state_jl = get(kv, "state_jl", "Neel")  # Julia choices: zeros|Neel|...
    periodic = lowercase(get(kv, "periodic", "true")) in ("1", "true", "yes", "y")

    outdir = get(kv, "outdir", "03_Nature_review_checks/results")
    mkpath(outdir)

    @printf("Running 3-way comparison: circuit=%s L=%d steps=%d dt=%g chi_max=%d sites=%s\n",
            circuit, L, steps, dt, chi_max, join(string.(sites), ","))

    if circuit != "Heisenberg"
        error("Only circuit=Heisenberg is implemented in this driver right now.")
    end

    # --- Run CircuitTDVP (Julia) ---
    jl_script = joinpath(@__DIR__, "heisenberg_circuitTDVP.jl")
    jl_tag = get(kv, "tag_jl", "circuitTDVP")
    cmd_jl = `julia --project=$(abspath(joinpath(@__DIR__, ".."))) $jl_script --L=$L --steps=$steps --dt=$dt --chi_max=$chi_max --sites=$(join(sites, ",")) --state=$state_jl --periodic=$(periodic) --outdir=$outdir --tag=$jl_tag`
    @printf("-> %s\n", string(cmd_jl))
    run(cmd_jl)

    # --- Run TenPy zip-up ---
    py_zip = joinpath(@__DIR__, "heisenberg16_mpo_zipup.py")
    py_zip_tag = get(kv, "tag_zipup", "tenpy_zipup")
    cmd_zip = `python3 $py_zip --L=$L --steps=$steps --dt=$dt --chi-max=$chi_max --sites=$(join(sites, ",")) --state=$state_py --outdir=$outdir --tag=$py_zip_tag`
    @printf("-> %s\n", string(cmd_zip))
    run(cmd_zip)

    # --- Run TenPy variational ---
    py_var = joinpath(@__DIR__, "heisenberg16_mpo_variational.py")
    py_var_tag = get(kv, "tag_var", "tenpy_variational")
    min_sweeps = parse(Int, get(kv, "min_sweeps", "1"))
    max_sweeps = parse(Int, get(kv, "max_sweeps", "2"))
    cmd_var = `python3 $py_var --L=$L --steps=$steps --dt=$dt --chi-max=$chi_max --sites=$(join(sites, ",")) --state=$state_py --outdir=$outdir --tag=$py_var_tag --min-sweeps=$min_sweeps --max-sweeps=$max_sweeps`
    @printf("-> %s\n", string(cmd_var))
    run(cmd_var)

    # --- Run Qiskit exact (statevector) ---
    py_ex = joinpath(@__DIR__, "qiskit_exact_heisenberg.py")
    py_ex_tag = get(kv, "tag_exact", "qiskit_exact")
    cmd_ex = `python3 $py_ex --L=$L --steps=$steps --dt=$dt --sites=$(join(sites, ",")) --state=$state_py --periodic=$(periodic) --outdir=$outdir --tag=$py_ex_tag`
    @printf("-> %s\n", string(cmd_ex))
    run(cmd_ex)

    # --- Load results ---
    jl_obs = joinpath(outdir, "$(jl_tag)_obs.csv")
    jl_chi = joinpath(outdir, "$(jl_tag)_chi.csv")
    zip_obs = joinpath(outdir, "$(py_zip_tag)_obs.csv")
    zip_chi = joinpath(outdir, "$(py_zip_tag)_chi.csv")
    var_obs = joinpath(outdir, "$(py_var_tag)_obs.csv")
    var_chi = joinpath(outdir, "$(py_var_tag)_chi.csv")
    ex_obs = joinpath(outdir, "$(py_ex_tag)_obs.csv")
    ex_chi = joinpath(outdir, "$(py_ex_tag)_chi.csv")

    _python_plot!(; outdir=outdir, jl_tag=jl_tag, zip_tag=py_zip_tag, var_tag=py_var_tag, exact_tag=py_ex_tag)
end

main()

