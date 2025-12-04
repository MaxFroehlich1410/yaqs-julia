print("import starts...")

import os
import sys
import math
import pickle
import numpy as np


def _format_float_short(value: float) -> str:
    """Format floats compactly for filenames (e.g., 0.1 -> 0p1)."""
    return f"{value:.4g}".replace('.', 'p')


def _build_experiment_name(
    num_qubits: int,
    num_layers: int,
    tau: float,
    noise_strength: float,
    basis_label: str,
    observable_basis: str = "Z",
    error_label: str = "XYZ",
) -> str:
    tokens = [
        "exact_dm",
        f"N{num_qubits}",
        f"L{num_layers}",
        f"tau{_format_float_short(tau)}",
        f"noise{_format_float_short(noise_strength)}",
        f"basis{basis_label}",
        f"obs{observable_basis}",
        f"err{error_label}",
    ]
    return "_".join(tokens)


def staggered_magnetization(expvals: np.ndarray, num_qubits: int) -> float:
    """Compute staggered magnetization from expectation values (works for Z/X/Y basis)."""
    return np.sum([(-1) ** i * expvals[i] for i in range(num_qubits)]) / num_qubits


if __name__ == "__main__":
    # Simulation parameters
    num_qubits = 8
    num_layers = 20
    tau = 0.1

    # What to plot/measure
    observable_basis = "Z"  # Choose from: "Z", "X", "Y"
    noise_strengths = [0.1, 0.01, 0.001]
    # Select which two-qubit Pauli-Lindblad errors to include on neighbors
    # Use any combination of: "X", "Y", "Z"
    error_axes = ["Y"]
    error_label = "".join(error_axes) if len(error_axes) > 0 else "none"

    # Ensure local yaqs package root takes precedence (direct parent of modules)
    yaqs_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if yaqs_pkg_root not in sys.path:
        sys.path.insert(0, yaqs_pkg_root)

    print("import of simulators/circuits starts...")
    # Exact (density matrix) simulator
    from codex_experiments.worker_functions.qiskit_simulators import run_qiskit_exact
    # Circuit builders
    from core.libraries.circuit_library import (
        qaoa_ising_layer,
        create_ising_circuit,
        create_heisenberg_circuit,
        xy_trotter_layer,
        xy_trotter_layer_longrange,
        create_2d_fermi_hubbard_circuit,
        create_1d_fermi_hubbard_circuit,
        create_2d_heisenberg_circuit,
        create_2d_ising_circuit,
        create_cz_brickwork_circuit,
        create_rzz_pi_over_2_brickwork,
        create_clifford_cz_frame_circuit,
        create_echoed_xx_pi_over_2,
        create_sy_cz_parity_frame,
    )

    print("import of qiskit starts...")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli
    from qiskit_aer.noise.errors import PauliLindbladError
    from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
    print("imports done")

    # Define circuit configurations (one Trotter step builders)
    def make_ising_step(periodic: bool):
        return create_ising_circuit(
            L=num_qubits, J=1.0, g=0.5, dt=tau, timesteps=1, periodic=periodic
        )

    def make_heisenberg_step(periodic: bool):
        return create_heisenberg_circuit(
            L=num_qubits, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0, dt=tau, timesteps=1, periodic=periodic
        )

    def make_1d_fermi_hubbard_step():
        L = num_qubits // 2  # two qubits per site
        return create_1d_fermi_hubbard_circuit(
            L=L, u=1.0, t=0.5, mu=0.0, num_trotter_steps=1, dt=tau, timesteps=1
        )

    def make_2d_fermi_hubbard_step():
        num_sites = num_qubits // 2
        num_rows = int(math.sqrt(num_sites))
        while num_sites % num_rows != 0 and num_rows > 1:
            num_rows -= 1
        num_cols = num_sites // num_rows
        return create_2d_fermi_hubbard_circuit(
            Lx=num_cols, Ly=num_rows, u=1.0, t=0.5, mu=0.0, num_trotter_steps=1, dt=tau, timesteps=1
        )

    def make_2d_heisenberg_step():
        num_rows = int(math.sqrt(num_qubits))
        while num_qubits % num_rows != 0 and num_rows > 1:
            num_rows -= 1
        num_cols = num_qubits // num_rows
        return create_2d_heisenberg_circuit(
            num_rows=num_rows, num_cols=num_cols, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0, dt=tau, timesteps=1
        )

    def make_2d_ising_step():
        num_rows = int(math.sqrt(num_qubits))
        while num_qubits % num_rows != 0 and num_rows > 1:
            num_rows -= 1
        num_cols = num_qubits // num_rows
        return create_2d_ising_circuit(
            num_rows=num_rows, num_cols=num_cols, J=1.0, g=0.5, dt=tau, timesteps=1
        )

    # Size choices:
    # - Default runs at 8 qubits
    # - 2D Ising/Heisenberg specifically at 9 qubits (3x3 grid)
    spin2d_rows = 3
    spin2d_cols = 3

    # Fermi-Hubbard targets 8 qubits:
    # - 1D FH: 2 qubits per site -> L=4
    # - 2D FH: 2 qubits per site -> Lx*Ly=4 e.g., 2x2
    fh_1d_L = 4
    fh_2d_Lx, fh_2d_Ly = 2, 2

    circuit_configs = [
        {"label": "QAOA_layer", "builder": lambda: qaoa_ising_layer(num_qubits)},
        {"label": "XY_layer", "builder": lambda: xy_trotter_layer(num_qubits, tau, order="YX")},
        {"label": "XY_longrange", "builder": lambda: xy_trotter_layer_longrange(num_qubits, tau, order="YX")},
        {"label": "Ising_open", "builder": lambda: make_ising_step(False)},
        {"label": "Ising_periodic", "builder": lambda: make_ising_step(True)},
        {"label": "Heisenberg_open", "builder": lambda: make_heisenberg_step(False)},
        {"label": "Heisenberg_periodic", "builder": lambda: make_heisenberg_step(True)},
        {"label": "2D_Fermi_Hubbard", "builder": lambda: create_2d_fermi_hubbard_circuit(Lx=fh_2d_Lx, Ly=fh_2d_Ly, u=1.0, t=0.5, mu=0.0, num_trotter_steps=1, dt=tau, timesteps=1)},
        {"label": "1D_Fermi_Hubbard", "builder": lambda: create_1d_fermi_hubbard_circuit(L=fh_1d_L, u=1.0, t=0.5, mu=0.0, num_trotter_steps=1, dt=tau, timesteps=1)},
        {"label": "2D_Heisenberg", "builder": lambda: create_2d_heisenberg_circuit(num_rows=spin2d_rows, num_cols=spin2d_cols, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0, dt=tau, timesteps=1)},
        {"label": "2D_Ising", "builder": lambda: create_2d_ising_circuit(num_rows=spin2d_rows, num_cols=spin2d_cols, J=1.0, g=0.5, dt=tau, timesteps=1)},
    ]







    # Prepare base output directory
    base_dir = os.path.dirname(__file__)
    parent_dir = os.path.join(base_dir, "Density_Matrix_Tests")
    os.makedirs(parent_dir, exist_ok=True)

    for noise_strength in noise_strengths:
        print("=" * 70)
        print("Exact Density Matrix Simulation")
        print("=" * 70)
        print(f"System: {num_qubits} qubits, {num_layers} layers")
        print(f"Noise strength: {noise_strength}")
        print(f"Observable basis: {observable_basis}")
        print("Mode: Exact density matrix")
        print("=" * 70)

        for cfg in circuit_configs:
            basis_label = cfg["label"]
            print("-" * 70)
            print(f"Running circuit: {basis_label} @ noise {noise_strength}")

            # One Trotter step for selected model (decides actual qubit count)
            trotter_step = cfg["builder"]()
            nq = trotter_step.num_qubits

            # Output paths (use actual qubit count of the circuit)
            experiment_name = _build_experiment_name(
                nq,
                num_layers,
                tau,
                noise_strength,
                basis_label,
                observable_basis,
                error_label,
            )
            output_dir = os.path.join(parent_dir, experiment_name)
            png_path = os.path.join(output_dir, f"{experiment_name}.png")
            pkl_path = os.path.join(output_dir, f"{experiment_name}.pkl")
            md_path = os.path.join(output_dir, f"{experiment_name}.md")
            if os.path.exists(png_path) and os.path.exists(pkl_path) and os.path.exists(md_path):
                print(f"✓ Skipping '{experiment_name}' (outputs already exist)")
                continue
            os.makedirs(output_dir, exist_ok=True)

            # Initial state circuit: X on qubits where i % 4 == 3 (yields Z = -1 there)
            init_circuit = QuantumCircuit(nq)
            for i in range(nq):
                if i % 4 == 3:
                    init_circuit.x(i)

            # Qiskit noise model: XX, YY, ZZ Pauli-Lindblad-like noise on nearest neighbors
            qiskit_noise_model = QiskitNoiseModel()
            TwoQubit_XX_error = PauliLindbladError(
                [Pauli("IX"), Pauli("XI"), Pauli("XX")],
                [noise_strength, noise_strength, noise_strength],
            )
            TwoQubit_YY_error = PauliLindbladError( 
                [Pauli("IY"), Pauli("YI"), Pauli("YY")],
                [noise_strength, noise_strength, noise_strength],
            )
            TwoQubit_ZZ_error = PauliLindbladError(
                [Pauli("IZ"), Pauli("ZI"), Pauli("ZZ")],
                [noise_strength, noise_strength, noise_strength],
            )
            for qubit in range(nq):
                next_qubit = (qubit + 1) % nq
                if "X" in error_axes:
                    qiskit_noise_model.add_quantum_error(
                        TwoQubit_XX_error,
                        ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx", "cp"],
                        [qubit, next_qubit],
                    )
                if "Y" in error_axes:
                    qiskit_noise_model.add_quantum_error(
                        TwoQubit_YY_error,
                        ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx", "cp"],
                        [qubit, next_qubit],
                    )
                if "Z" in error_axes:
                    qiskit_noise_model.add_quantum_error(
                        TwoQubit_ZZ_error,
                        ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx", "cp"],
                        [qubit, next_qubit],
                    )

            # Initial expectation values for chosen basis
            if observable_basis == "Z":
                expvals_initial = np.array([1.0 if i % 4 != 3 else -1.0 for i in range(nq)])
                y_label_local = r"$\langle Z_i \rangle$"
                y_label_stag = r"$S^z(\pi)$"
            elif observable_basis == "X":
                expvals_initial = np.zeros(nq)
                y_label_local = r"$\langle X_i \rangle$"
                y_label_stag = r"$S^x(\pi)$"
            else:  # "Y"
                expvals_initial = np.zeros(nq)
                y_label_local = r"$\langle Y_i \rangle$"
                y_label_stag = r"$S^y(\pi)$"

            # Exact density matrix simulation (returns shape: (num_qubits, num_layers))
            print("Running exact density matrix simulation (reference)...")
            expvals_exact = run_qiskit_exact(
                    nq,
                    num_layers,
                    init_circuit,
                    trotter_step,
                qiskit_noise_model,
                method="density_matrix",
                observable_basis=observable_basis,
            )
            print("Exact reference computed.\n")

            # Assemble per-qubit time series including t=0
            # expvals_plot: shape (num_qubits, num_layers+1)
            expvals_plot = np.column_stack((expvals_initial, expvals_exact))

            # Staggered magnetization time series including t=0
            exact_stag = [staggered_magnetization(expvals_initial, nq)] + [
                staggered_magnetization(expvals_exact[:, t], nq) for t in range(num_layers)
            ]

            # Plot: two subplots (local expvals for all qubits; staggered magnetization)
            import matplotlib.pyplot as plt

            times = np.arange(num_layers + 1) * tau
            fig, (ax_exp, ax_stag) = plt.subplots(1, 2, figsize=(16, 5))

            # Subplot 1: Local expectation values for all qubits
            for q in range(nq):
                ax_exp.plot(times, expvals_plot[q, :], lw=1.5, alpha=0.9)
            ax_exp.set_xlabel("Time", fontsize=12)
            ax_exp.set_ylabel(y_label_local, fontsize=12)
            ax_exp.set_title("Local Expectation Values (all qubits)", fontsize=13)
            ax_exp.grid(True, linestyle="--", alpha=0.5)

            # Subplot 2: Staggered magnetization
            ax_stag.plot(times, exact_stag, "-", color="black", lw=2.0)
            ax_stag.set_xlabel("Time", fontsize=12)
            ax_stag.set_ylabel(y_label_stag, fontsize=12)
            ax_stag.set_title("Staggered Magnetization", fontsize=13)
            ax_stag.grid(True, linestyle="--", alpha=0.5)

            # Figure title with key parameters
            title = (
                f"N={nq}, L={num_layers}, tau={tau}, noise={noise_strength}, "
                f"basis={basis_label}, obs={observable_basis}, mode=DM"
            )
            fig.suptitle(title, fontsize=14)
            plt.tight_layout(rect=(0, 0, 1, 0.95))

            # Save outputs
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {
                        "num_qubits": nq,
                "num_layers": num_layers,
                "tau": tau,
                "noise_strength": noise_strength,
                        "observable_basis": observable_basis,
                "basis_label": basis_label,
                        "error_axes": error_axes,
                        "times": times,
                        "expvals_per_qubit": expvals_plot,
                        "staggered_magnetization": np.array(exact_stag),
                    },
                    f,
                )

            plt.savefig(png_path, dpi=300)
            plt.close(fig)

            print(f"Saved plot to: {png_path}")
            print(f"Saved data to: {pkl_path}")

            # Save markdown summary
            md_lines = []
            md_lines.append("# Exact Density Matrix Simulation")
            md_lines.append("")
            md_lines.append(f"**Experiment**: `{experiment_name}`")
            md_lines.append("")
            md_lines.append("## Parameters")
            md_lines.append("")
            md_lines.append(f"- **N (qubits)**: {nq}")
            md_lines.append(f"- **L (layers)**: {num_layers}")
            md_lines.append(f"- **tau**: {tau}")
            md_lines.append(f"- **noise strength**: {noise_strength}")
            md_lines.append(f"- **circuit**: {basis_label}")
            md_lines.append(f"- **observable basis**: {observable_basis}")
            md_lines.append(f"- **mode**: DM (exact)")
            md_lines.append(f"- **errors**: {', '.join(error_axes) if error_axes else 'none'}")
            md_lines.append("")
            md_lines.append("## Outputs")
            md_lines.append("")
            md_lines.append("- Two subplots: all-qubit local expectation values, and staggered magnetization")
            md_lines.append("")
            md_lines.append(f"![Plot]({os.path.basename(png_path)})")
            md_content = "\n".join(md_lines)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            print(f"Saved markdown to: {md_path}")


