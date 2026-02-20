#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    csvfile = here / "timeseries.csv"
    outpng = here / "se_drop_first_bug.png"

    with csvfile.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader]

    data = np.array(rows, dtype=float)
    col = {name: i for i, name in enumerate(header)}

    t = data[:, col["t"]]
    zref = data[:, col["z_ref_qutip"]]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))

    for name in header:
        if name in ("t", "z_ref_qutip"):
            continue

        z = data[:, col[name]]

        if name == "DOUBLEADAPTIVE":
            # "shift BUG left by one step": compare z_bug[k+1] to z_ref[k]
            err_sq = (z[1:] - zref[:-1]) ** 2
            ax.plot(
                t[:-1],
                err_sq,
                linewidth=1.6,
                alpha=0.9,
                label=f"{name} (drop first sample)",
            )
        else:
            err_sq = (z - zref) ** 2
            ax.plot(t, err_sq, linewidth=1.2, alpha=0.65, label=name)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"|Δ⟨Z⟩|²")
    ax.set_title("Squared error vs exact (qutip): DOUBLEADAPTIVE shifted left by 1 step")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpng, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpng}")


if __name__ == "__main__":
    main()

