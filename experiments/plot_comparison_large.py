import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def plot_comparison():
    julia_file = "experiments/julia_benchmark_results.csv"
    python_file = "experiments/python_benchmark_results.csv"
    
    if not os.path.exists(julia_file) or not os.path.exists(python_file):
        print("Error: Result files not found. Please run benchmarks first.")
        return

    df_julia = pd.read_csv(julia_file)
    df_python = pd.read_csv(python_file)

    # Plot settings
    plt.figure(figsize=(14, 8))
    
    # Columns to compare
    cols = ["Z_First", "Z_Middle", "Z_Last"]
    colors = ['b', 'g', 'r']
    
    for i, col in enumerate(cols):
        # Plot Julia (Solid)
        plt.plot(df_julia["Time"], df_julia[col], label=f"Julia {col}", 
                 color=colors[i], linestyle='-', linewidth=2, alpha=0.8)
        
        # Plot Python (Dashed)
        plt.plot(df_python["Time"], df_python[col], label=f"Python {col}", 
                 color=colors[i], linestyle='--', linewidth=2, alpha=0.8)

    plt.xlabel("Time")
    plt.ylabel("<Z>")
    plt.title("Analog TJM Comparison: Julia vs Python\n(L=30, Traj=200, Ising Chain)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_img = "experiments/comparison_large.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    plot_comparison()

