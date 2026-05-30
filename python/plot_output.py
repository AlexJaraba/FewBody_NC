import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import subprocess
import argparse

from dataclasses import dataclass
from pathlib import Path

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output.csv"
DEFAULT_DIAGNOSTICS_PATH = PROJECT_ROOT / "diagnostics.csv"
DEFAULT_PARAM_PATH = PROJECT_ROOT / "data" / "param.txt"
DEFAULT_EXECUTABLE_PATH = PROJECT_ROOT / "few_body_nc.exe"

# ============================================================
# Configuration
# ============================================================

@dataclass
class PlotConfig:
    G: float = 0.000296014912
    epsilon: float = 1e-300
    figure_size: tuple = (16, 10)
    orbit_marker_size: float = 4.0
    start_marker_size: float = 60.0

# ============================================================
# Reading data
# ============================================================

def read_output(path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Output file not found: {path}")
    df = pd.read_csv(path)

    required = {"time", "id", "x", "y", "z", "vx", "vy", "vz", "mass"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in output: {sorted(missing)}")
    
    df = df.sort_values(by=["time", "id"]).reset_index(drop=True)

    return df

def read_diagnostics(path: Path = DEFAULT_DIAGNOSTICS_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    
    df = pd.read_csv(path)
    required = {"time", "total_energy", "angular_momentum", "linear_momentum", "com_drift", "shadow_energy", "timestep"}
    missing = required - set(df.columns)

    if missing:
        print(f"Warning: Missing required columns in diagnostics: {sorted(missing)}")
        return None
    
    return df.sort_values(by="time").reset_index(drop=True)

# ============================================================
# Vectorized diagnostics from output.csv
# ============================================================

def compute_diagnostics_from_output(df: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    times = np.sort(df["time"].unique())
    body_ids = np.sort(df["id"].unique())

    nt = len(times)
    nb = len(body_ids)

    expected_rows = nt * nb
    if len(df) != expected_rows:
        raise ValueError("Output.csv does not look rectangular." "Expected one row per body per time step.")
    
    ordered = df.sort_values(["time", "id"])
    
    mass = ordered["mass"].to_numpy().reshape(nt, nb)
    pos = ordered[["x", "y", "z"]].to_numpy().reshape(nt, nb, 3)
    vel = ordered[["vx", "vy", "vz"]].to_numpy().reshape(nt, nb, 3)

    # Kinetic Energy
    kinetic = 0.5 * np.sum(mass * np.sum(vel * vel, axis=2), axis=1)

    # Potential Energy
    potential = np.zeros(nt)

    for i in range(nb):
        for j in range(i+1, nb):
            dr = pos[:, j, :] - pos[:, i, :]
            r = np.linalg.norm(dr, axis=1)
            potential -= config.G * mass[:, i] * mass[:, j] / np.maximum(r, config.epsilon)
    
    # Total Energy
    total_energy = kinetic + potential

    # Linear Momentum
    momentum_vec = np.sum(mass[:, :, None] * vel, axis=1)
    linear_momentum = np.linalg.norm(momentum_vec, axis=1)

    # Angular Momentum
    angular_vec = np.sum(np.cross(pos, mass[:, :, None] * vel), axis=1)
    angular_momentum = np.linalg.norm(angular_vec, axis=1)

    # Center of Mass Drift
    total_mass = np.sum(mass, axis=1)
    rcm = np.sum(mass[:, :, None] * pos, axis=1) / total_mass[:, None]
    com_drift = np.linalg.norm(rcm, axis=1)

    # Print Statements
    diagnostics = pd.DataFrame({
        "time": times,
        "kinetic Energy": kinetic,
        "potential Energy": potential,
        "total_energy": total_energy,
        "angular_momentum": angular_momentum,
        "linear_momentum": linear_momentum,
        "com_drift": com_drift,
    })
    return diagnostics

# ============================================================
# Error helpers
# ============================================================

def absolute_error(values: np.ndarray) -> np.ndarray:
    return np.abs(values - values[0])

def relative_error(values: np.ndarray, epsilon: float = 1e-300) -> np.ndarray:
    denom = max(np.abs(values[0]), epsilon)
    return np.abs((values - values[0]) / denom)

def print_summary(diagnostics: pd.DataFrame, config: PlotConfig) -> None:
    energy = diagnostics["total_energy"].to_numpy()
    angular = diagnostics["angular_momentum"].to_numpy()
    linear = diagnostics["linear_momentum"].to_numpy()
    com = diagnostics["com_drift"].to_numpy()

    dE = relative_error(energy, config.epsilon)
    dL = relative_error(angular, config.epsilon)
    dP = absolute_error(linear)
    dRcm = absolute_error(com)

    print("Summary of Diagnostics:")
    print(f"Max |dE/E0|:", np.max(dE))
    print(f"Max |dL/L0|:", np.max(dL))
    print(f"Max |dP|:", np.max(dP))
    print(f"Max |dRcm|:", np.max(dRcm))
    print(f"Final dE/E0:", dE[-1])
    print(f"Final dL/L0:", dL[-1])
    print(f"Final dP:", dP[-1])
    print(f"Final dRcm:", dRcm[-1])

# ============================================================
# Plotting
# ============================================================

def plot_orbits(ax, df: pd.DataFrame, config: PlotConfig) -> None:
    for body_id, body in df.groupby("id", sort=True):
        x = body["x"].to_numpy()
        y = body["y"].to_numpy()

        ax.plot(x, y, linewidth=1.2, label=f'Body {body_id}')
        ax.scatter(x[0], y[0], s=config.start_marker_size, zorder=5)

    ax.set_title('Orbits of Bodies')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

def plot_error(ax, time, values, title, ylabel, floor=1e-300) -> None:
    ax.semilogy(time, np.maximum(values, floor))
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)

def plot_verification_suite(output_df: pd.DataFrame, diagnostics: pd.DataFrame, config: PlotConfig) -> None:
    time = diagnostics["time"].to_numpy()
    dE = relative_error(diagnostics["total_energy"].to_numpy(), config.epsilon)
    dL = relative_error(diagnostics["angular_momentum"].to_numpy(), config.epsilon)
    dP = absolute_error(diagnostics["linear_momentum"].to_numpy())
    dRcm = absolute_error(diagnostics["com_drift"].to_numpy())

    print_summary(diagnostics, config)

    fig = plt.figure(figsize=config.figure_size, constrained_layout=True)
    gs = fig.add_gridspec(2, 4)

    ax_orbit = fig.add_subplot(gs[:, 0:2])
    plot_orbits(ax_orbit, output_df, config)

    ax_energy = fig.add_subplot(gs[0, 2])
    plot_error(ax_energy, time, dE, "Relative Energy Error", r"$|E - E_0|/|E_0|$", floor=1e-18)

    ax_angular = fig.add_subplot(gs[0, 3])
    plot_error(ax_angular, time, dL, "Relative Angular Momentum Error", r"$|L - L_0|/|L_0|$", floor=1e-18)

    ax_linear = fig.add_subplot(gs[1, 2])
    plot_error(ax_linear, time, dP, "Linear Momentum Error", r"$|P - P_0|$", floor=1e-30)

    ax_com = fig.add_subplot(gs[1, 3])
    plot_error(ax_com, time, dRcm, "Center of Mass Drift", r"$|R_{\rm cm} - R_{\rm cm,0}|$", floor=1e-30)

    fig.suptitle("FewBodyNC Verification Suite", fontsize=16)
    plt.show()

def plot_shadow_hamiltonian(path: Path = DEFAULT_DIAGNOSTICS_PATH) -> None:
    diagnostics = read_diagnostics(path)

    if diagnostics is None:
        raise FileNotFoundError(f"No usable diagnostics.csv found.")

    time = diagnostics["time"].to_numpy()
    shadow = diagnostics["shadow_energy"].to_numpy()

    dH = relative_error(shadow)

    plt.figure(figsize=(8, 5), constrained_layout=True)
    plt.semilogy(time, np.maximum(dH, 1e-18), label="Shadow Hamiltonian")
    plt.xlabel("Time")
    plt.ylabel(r"$|(\tilde{H} - \tilde{H}_0) / \tilde{H}_0|$")
    plt.title("Shadow Hamiltonian Conservation")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

# ============================================================
# Parameter rewriting and convergence study
# ============================================================

def rewrite_param(dt: float, runtime: float, param_path: Path = DEFAULT_PARAM_PATH) -> None:
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find param file: {param_path}")

    lines = param_path.read_text().splitlines()
    updated = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("timestep"):
            updated.append(f"timestep {dt}")
        elif stripped.startswith("runtime"):
            updated.append(f"runtime {runtime}")
        else:
            updated.append(line)
    param_path.write_text("\n".join(updated) + "\n")

def run_executable(executable_path: Path = DEFAULT_EXECUTABLE_PATH) -> None:
    if not executable_path.exists():
        raise FileNotFoundError(f"Could not find executable: {executable_path}")
    
    result = subprocess.run([str(executable_path)], cwd=str(PROJECT_ROOT), capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Simulation failed.")

def final_positions(df: pd.DataFrame) -> dict[int, np.ndarray]:
    final_time = df["time"].max()
    final_step = df[df["time"] == final_time]

    positions = {}

    for body_id, body in final_step.groupby("id", sort=True):
        positions[int(body_id)] = body[["x", "y", "z"]].to_numpy()[0]
    
    return positions

def rms_position_error(test_positions: dict[int, np.ndarray], reference_positions: dict[int, np.ndarray]) -> float:
    total = 0.0
    count = 0

    for body_id, r_ref in reference_positions.items():
        r = test_positions[body_id]
        dr = r - r_ref

        total += np.dot(dr, dr)
        count += 1

    return np.sqrt(total / count)

def run_timestep_scaling_study(dt_ref: float = 0.00025, dts: tuple = (0.01, 0.005, 0.0025, 0.00125), runtime: float = 1.0) -> None:
    print("\nRunning reference solution...")

    rewrite_param(dt_ref, runtime)
    if DEFAULT_OUTPUT_PATH.exists():
        DEFAULT_OUTPUT_PATH.unlink()

    run_executable()

    df_ref = read_output(DEFAULT_OUTPUT_PATH)
    reference = final_positions(df_ref)

    errors = []

    for dt in dts:
        print(f"\nRunning dt = {dt}")
        rewrite_param(dt, runtime)
        if DEFAULT_OUTPUT_PATH.exists():
            DEFAULT_OUTPUT_PATH.unlink()
        
        run_executable()

        df = read_output(DEFAULT_OUTPUT_PATH)
        err = rms_position_error(final_positions(df), reference)

        errors.append(err)

        print(f"RMS position error: {err}")
    
    dts_array = np.array(dts)
    errors_array = np.array(errors)

    plt.figure(figsize=(7, 5), constrained_layout=True)
    plt.loglog(dts_array, errors_array, marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Timestep dt")
    plt.ylabel("RMS final-position error")
    plt.title("Timestep convergence")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.show()

    print("\nConvergence Ratios:")
    for i in range(len(errors_array) - 1):
        ratio = errors_array[i] / errors_array[i + 1]
        print(f"{dts_array[i]} -> {dts_array[i + 1]} : ratio = {ratio}")

# ============================================================
# Command line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FewBodyNC plotting and verification utility.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to output.csv.",)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH, help="Path to diagnostics.csv.",)
    parser.add_argument("--G", type=float, default=0.000296014912, help="Gravitational constant used for recomputed diagnostics.",)
    parser.add_argument("--use-diagnostics-csv", action="store_true", help="Use diagnostics.csv for diagnostic plots instead of recomputing from output.csv.",)
    parser.add_argument("--shadow", action="store_true", help="Plot shadow Hamiltonian error from diagnostics.csv.",)
    parser.add_argument("--convergence", action="store_true", help="Run timestep convergence study.",)

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = PlotConfig(G=args.G)

    if args.shadow:
        plot_shadow_hamiltonian(args.diagnostics)
        return
    if args.convergence:
        run_timestep_scaling_study()
        return
    output_df = read_output(args.output)
    if args.use_diagnostics_csv:
        diagnostics = read_diagnostics(args.diagnostics)
        output_df = read_output(args.output)
        if diagnostics is None:
            print("Falling back to recomputing diagnostics from output.csv.")
            diagnostics = compute_diagnostics_from_output(output_df, config)
    else:
        diagnostics = compute_diagnostics_from_output(output_df, config)
    
    plot_verification_suite(output_df, diagnostics, config)

# ====================================================

if __name__ == "__main__":
    main()

# === DIFFERENT ARGUMENTS TO RUN ===
# Regular Plot Output: python python/plot_output.py
# Use Diagnostics instead of recomputing enegry: python python/plot_output.py --use-diagnostics-csv
# Plot Shadow Hamiltonian: python python/plot_output.py --shadow
# Run Convergence Study: python python/plot_output.py --convergence
# Use Different Graviational Constant: python python/plot_output.py --G 0.000296014912