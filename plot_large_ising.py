import pandas as pd
import matplotlib.pyplot as plt
import sys

try:
    df = pd.read_csv("large_ising_results.csv")
except FileNotFoundError:
    print("Error: large_ising_results.csv not found.")
    sys.exit(1)

sites = sorted(df["Site"].unique())

plt.figure(figsize=(10, 6))
plt.title("Large Ising Chain Simulation (L=30, 2-TDVP, D=32)")

for site in sites:
    data = df[df["Site"] == site]
    # Map 1-based to 0-based for legend if user thinks 0-based
    # User asked for 0, 15, 29.
    # Site 1 -> Qubit 0
    # Site 16 -> Qubit 15
    # Site 30 -> Qubit 29
    label_idx = site - 1
    plt.plot(data["Time"], data["ExpVal"], label=f"Qubit {label_idx} <Z>", marker='o', markersize=3, linestyle='-')

plt.xlabel("Time")
plt.ylabel("<Z>")
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("large_ising_simulation.png")
print("Saved large_ising_simulation.png")

