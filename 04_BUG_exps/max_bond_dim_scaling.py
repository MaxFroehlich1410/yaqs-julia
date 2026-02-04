"""
This file runs simulations that determine the effect of the maximum bond dimension.
"""

from .eval_util import main

if __name__ == "__main__":
    main(param_name="max_bond_dim",
         plot_kwargs={"xlabel": "Maximum Bond Dimension"})
