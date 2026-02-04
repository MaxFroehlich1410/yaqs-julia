"""
This file runs simulations that determine the effect of the threshold.
"""

from .eval_util import main
from .ising_sim_util import BUGKind

if __name__ == "__main__":
    main(param_name="threshold",
         param_dtype=float,
         plot_kwargs={"xscale": "log", "xlabel": "Truncation Threshold"},
         bug_kinds=[BUGKind.ADAPTIVE,
                    BUGKind.DOUBLEADAPTIVE,
                    BUGKind.HYBRID])
