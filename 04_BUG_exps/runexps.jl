"""
Convenience runner for the Julia experiment scripts in this folder.

Examples:
  julia --project=. 04_BUG_exps/runexps.jl --outdir=04_BUG_exps/results
"""

include("exp_runs.jl")                # defines module BUGExpRuns
include("exp_max_bond_dim_scaling.jl") # defines module BUGExpMaxBondDimScaling
include("exp_threshold_scaling.jl")    # defines module BUGExpThresholdScaling

function main(args=ARGS)
    # Forward args to each exp (each script reads `ARGS` by default, so call their `main` explicitly).
    BUGExpRuns.main(args)
    BUGExpMaxBondDimScaling.main(args)
    BUGExpThresholdScaling.main(args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

