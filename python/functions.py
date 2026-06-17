import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import subprocess

from dataclasses import dataclass
from pathlib import Path

"""
FewBodyNC plotting and benchmark utilities

This file contains reusable helper functions used by plot_output.py

Responsibilities:
    - Read output.csv and diagnostics.csv
    - Compute diagnostics from saved Cartesian output
    - Plot orbits and conservation diagnostics
    - Run timestep convergence studies.
    - Run the full benchmark suite and save plots.

The C++ code always writes physical Cartesian output, even when the simulation is evolved internally in Jacobi coordinates.

"""


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_INITIAL_CONDITIONS_PATH = PROJECT_ROOT / "data" / "initial_conditions.txt"
DEFAULT_BENCHMARK_PLOT_DIR = PROJECT_ROOT / "benchmark_plots"
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
# Benchmark Tests:
#   Benchmark systems used to demonstrate both the strengths and limits of the current integrators.
#   These are written directly into data/initial_conditions.txt before each benchmark run.
# ============================================================

BENCHMARK_TESTS = [
    {
        "name": "test1_binary",
        "dt": 0.1,
        "runtime": 100000,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0, -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,  0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
        ],
    },
    {
        "name": "test2_circumbinary_triple",
        "dt": 0.02,
        "runtime": 100000,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (1e-3,  5.0, 0.0, 0.0, 0.0,  0.01089,  0.0),
        ],
    },
    {
        "name": "test3_stronger_perturbed_triple",
        "dt": 0.01,
        "runtime": 50000,
        "output_frequency": 500,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (0.05,  4.0, 0.0, 0.0, 0.0,  0.01220,  0.0),
        ],
    },
    {
        "name": "test4_scattering_escape",
        "dt": 0.02,
        "runtime": 100000,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0, -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,  0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (0.1,  3.0, 0.0, 0.0, 0.0,  0.0040,   0.0),
        ],
    },
    {
        "name": "test5_figure8",
        "dt": 0.001,
        "runtime": 1000,
        "output_frequency": 10,
        "initial_conditions": [
            (1.0, -0.97000436,  0.24308753, 0.0,  0.008019029,  0.007436436, 0.0),
            (1.0,  0.97000436, -0.24308753, 0.0,  0.008019029,  0.007436436, 0.0),
            (1.0,  0.0,         0.0,        0.0, -0.016038058, -0.014872872, 0.0),
        ],
    },
    {
        "name": "test6_close_encounter",
        "dt": 0.002,
        "runtime": 10000,
        "output_frequency": 100,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0,  0.0,    -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0,  0.0,     0.012166, 0.0),
            (0.01,  1.2, 0.2, 0.0, -0.0020,  0.0040,   0.0),
        ],
    },
    {
        "name": "test7_solar_system",
        "dt": 1.0,
        "runtime": 36525,
        "output_frequency": 100,
        "initial_conditions": [
            (1.0,        0.0,            0.0,            0.0,  0.0,            0.0,            0.0),
            (1.6601e-7,  0.3637531341,   0.1323953134,   0.0, -0.0094579726,  0.0259855662,  0.0),
            (2.4478e-6,  0.1872120975,   0.6986850598,   0.0, -0.0195403467,  0.0052358201,  0.0),
            (3.0035e-6, -0.6427876097,   0.7660444431,   0.0, -0.0131798787, -0.0110592314,  0.0),
            (3.2272e-7, -1.3195447212,  -0.7618395000,   0.0,  0.0069691551, -0.0120709307,  0.0),
            (9.5458e-4,  2.6022000000,  -4.5071426115,   0.0,  0.0065344536,  0.0037726685,  0.0),
            (2.8588e-4,  7.3406974806,   6.1595765486,   0.0, -0.0035730960,  0.0042582500,  0.0),
            (4.3662e-5, -18.0593886633,  6.5730799225,   0.0, -0.0013423302, -0.0036880218,  0.0),
            (5.1514e-5, -5.2285466296, -29.6525614432,   0.0,  0.0030879059, -0.0005444811,  0.0),
        ],
    },
    {
        "name": "test8_inner_planets",
        "dt": 0.1,
        "runtime": 36525,
        "output_frequency": 10,
        "initial_conditions": [
            (1.0,        0.0,            0.0,           0.0,  0.0,            0.0,            0.0),
            (1.6601e-7,  0.3637531341,   0.1323953134,  0.0, -0.0094579726,  0.0259855662,  0.0),
            (2.4478e-6,  0.1872120975,   0.6986850598,  0.0, -0.0195403467,  0.0052358201,  0.0),
            (3.0035e-6, -0.6427876097,   0.7660444431,  0.0, -0.0131798787, -0.0110592314,  0.0),
            (3.2272e-7, -1.3195447212,  -0.7618395000,  0.0,  0.0069691551, -0.0120709307,  0.0),
        ],
    },
]

# ============================================================
# Reading data
# ============================================================

def read_param(param_path: Path = DEFAULT_PARAM_PATH) -> dict:
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find param file: {param_path}")
    
    params = {}

    for line in param_path.read_text().splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue
        
        parts = stripped.split()

        if len(parts) < 2:
            continue

        key = parts[0]
        value = parts[1]
        params[key] = value
    
    return params

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
# Compute functions
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
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "total_energy": total_energy,
        "angular_momentum": angular_momentum,
        "linear_momentum": linear_momentum,
        "com_drift": com_drift,
    })
    return diagnostics

def compute_final_positions(df: pd.DataFrame) -> dict[int, np.ndarray]:
    final_time = df["time"].max()
    final_step = df[df["time"] == final_time]

    positions = {}

    for body_id, body in final_step.groupby("id", sort=True):
        positions[int(body_id)] = body[["x", "y", "z"]].to_numpy()[0]
    
    return positions

def compute_finite_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return float("nan")
    
    return float(np.max(finite))
# ============================================================
# Error calculations
# ============================================================

def error_absolute(values: np.ndarray) -> np.ndarray:
    return np.abs(values - values[0])

def error_relative(values: np.ndarray, epsilon: float = 1e-300) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    safe_epsilon = max(float(epsilon), 1e-14)
    finite = np.isfinite(values)

    result = np.full_like(values, np.nan, dtype=float)

    if not np.any(finite):
        return result
    
    reference = values[0]

    if not np.isfinite(reference):
        reference = values[finite][0]
    if abs(reference) < safe_epsilon:
        scale = np.nanmax(np.abs(values[finite]))
        if not np.isfinite(scale) or scale < safe_epsilon:
            scale = 1.0
    else:
        scale = abs(reference)
    
    result[finite] = np.abs(values[finite] - reference) / scale
    return result

def error_rms_position(test_positions: dict[int, np.ndarray], reference_positions: dict[int, np.ndarray]) -> float:
    total = 0.0
    count = 0

    for body_id, r_ref in reference_positions.items():
        r = test_positions[body_id]
        dr = r - r_ref

        total += np.dot(dr, dr)
        count += 1

    return np.sqrt(total / count)

def error_safe_log_values(values: np.ndarray, floor: float = 1e-300, ceiling: float = 1e50) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    
    safe = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)

    safe[finite] = np.abs(values[finite])
    safe[finite] = np.clip(safe[finite], floor, ceiling)

    return safe

def error_print_summary(diagnostics: pd.DataFrame, config: PlotConfig) -> None:
    energy = diagnostics["total_energy"].to_numpy()
    angular = diagnostics["angular_momentum"].to_numpy()
    linear = diagnostics["linear_momentum"].to_numpy()
    com = diagnostics["com_drift"].to_numpy()

    dE = error_relative(energy, config.epsilon)
    dL = error_relative(angular, config.epsilon)
    dP = error_absolute(linear)
    dRcm = error_absolute(com)

    print("Summary of Diagnostics:")
    print(f"Max |dE/E0|:", np.max(dE))
    print(f"Max |dL/L0|:", np.max(dL))
    print(f"Max |dP|:", np.max(dP))
    print(f"Max |dRcm|:", np.max(dRcm))
    print(f"Final dE/E0:", dE[-1])
    print(f"Final dL/L0:", dL[-1])
    print(f"Final dP:", dP[-1])
    print(f"Final dRcm:", dRcm[-1])

def error_diagnostic_metric_summary(diagnostics: pd.DataFrame, config: PlotConfig) -> dict:
    energy = diagnostics["total_energy"].to_numpy()
    angular = diagnostics["angular_momentum"].to_numpy()
    linear = diagnostics["linear_momentum"].to_numpy()
    com = diagnostics["com_drift"].to_numpy()

    dE = error_relative(energy, config.epsilon)
    dL = error_relative(angular, config.epsilon)
    dP = error_absolute(linear)
    dRcm = error_absolute(com)

    return {"max_dE_over_E0": compute_finite_max(dE), 
            "final_dE_over_E0": float(dE[-1]), 
            "max_dL_over_L0": compute_finite_max(dL), 
            "final_dL_over_L0": float(dL[-1]), 
            "max_dP": compute_finite_max(dP), 
            "final_dP": float(dP[-1]), 
            "max_dRcm": compute_finite_max(dRcm), 
            "final_dRcm": float(dRcm[-1]),}

# ============================================================
# Plotting
# ============================================================

def plot_orbits(ax, df: pd.DataFrame, config: PlotConfig) -> None:
    max_abs_plot_value = 1e50

    for body_id, body in df.groupby("id", sort=True):
        x = body["x"].to_numpy(dtype=float)
        y = body["y"].to_numpy(dtype=float)

        valid = (np.isfinite(x) & np.isfinite(y) & (np.abs(x) < max_abs_plot_value) & (np.abs(y) < max_abs_plot_value))

        if not np.any(valid):
            continue

        ax.plot(x[valid], y[valid], linewidth=1, label=f'Body {body_id}')
        first_valid = np.flatnonzero(valid)[0]
        ax.scatter(x[first_valid], y[first_valid], s=config.start_marker_size, zorder=5)

    ax.set_title('Orbits of Bodies')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

def plot_error(ax, time, values, title, ylabel, floor=1e-300) -> None:
    time = np.asarray(time, dtype=float)
    y = error_safe_log_values(values, floor=floor)

    valid = np.isfinite(time) & np.isfinite(y)

    if np.any(valid):
        ax.semilogy(time[valid], y[valid])
        ymin = np.nanmin(y[valid])
        ymax = np.nanmax(y[valid])

        if ymin == ymax:
            ymin = max(ymin * 0.5, floor)
            ymax = min(ymax * 2.0, 1e50)
        else:
            ymin = max(ymin * 0.5, floor)
            ymax = min(ymax * 2.0, 1e50)
        
        if not np.isfinite(ymin) or ymin <= 0.0:
            ymin = floor
        if not np.isfinite(ymax) or ymax <= ymin:
            ymax = ymin * 10.0
        
        ax.set_ylim(ymin, ymax)
    else:
        ax.text(0.5, 0.5, "No finite diagnostic values", ha="center", va="center", transform=ax.transAxes,)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)

def plot_verification_suite(output_df: pd.DataFrame, 
                            diagnostics: pd.DataFrame, 
                            config: PlotConfig, 
                            save_path: Path | None = None, 
                            show: bool = True,) -> None:
    
    time = diagnostics["time"].to_numpy()
    dE = error_relative(diagnostics["total_energy"].to_numpy(), config.epsilon)
    dL = error_relative(diagnostics["angular_momentum"].to_numpy(), config.epsilon)
    dP = error_absolute(diagnostics["linear_momentum"].to_numpy())
    dRcm = error_absolute(diagnostics["com_drift"].to_numpy())

    error_print_summary(diagnostics, config)

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
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_shadow_hamiltonian(path: Path = DEFAULT_DIAGNOSTICS_PATH) -> None:
    diagnostics = read_diagnostics(path)

    if diagnostics is None:
        raise FileNotFoundError(f"No usable diagnostics.csv found.")

    time = diagnostics["time"].to_numpy()
    shadow = diagnostics["shadow_energy"].to_numpy()

    dH = error_relative(shadow)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    plot_error(ax, time, dH, "Shadow Hamiltonian Conservation", r"$|(\tilde{H} - \tilde{H}_0) / \tilde{H}_0|$", floor=1e-18)
    plt.show()

# ============================================================
# Rewrite param.txt and initial_conditions.txt
# ============================================================

def rewrite_param(dt: float, 
                  runtime: float, 
                  output_frequency: int, 
                  integrator: str, 
                  coordinate_mode: str, 
                  G: float = 0.000296014912, 
                  param_path: Path = DEFAULT_PARAM_PATH) -> None:

    if not param_path.exists():
        raise FileNotFoundError(f"Could not find param file: {param_path}")

    replacements = {"output_frequency": f"output_frequency {output_frequency}",
                    "runtime": f"runtime {runtime}",  
                    "timestep": f"timestep {dt}", 
                    "integrator": f"integrator {integrator}", 
                    "coordinate_mode": f"coordinate_mode {coordinate_mode}", 
                    "gravitational_constant": f"gravitational_constant {G}"}

    lines = param_path.read_text().splitlines()
    updated = []
    seen = set()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            updated.append(line)
            continue
        key = stripped.split()[0]
        if key in replacements:
            updated.append(replacements[key])
            seen.add(key)
        else:
            updated.append(line)
        
    for key, value in replacements.items():
        if key not in seen:
            updated.append(value)

    param_path.write_text("\n".join(updated) + "\n")

def rewrite_timestep_only(dt: float, param_path: Path = DEFAULT_PARAM_PATH) -> None:
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find param file: {param_path}")
    lines = param_path.read_text().splitlines()
    updated = []
    found_timestep = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("timestep"):
            updated.append(f"timestep {dt}")
            found_timestep = True
        else:
            updated.append(line)
    if not found_timestep:
        updated.append(f"timestep {dt}")

    param_path.write_text("\n".join(updated) + "\n")

def rewrite_adaptive_settings(adaptive_timesteps: bool, timestep_levels: int | None = None, timestep_eta: float | None = None, param_path: Path = DEFAULT_PARAM_PATH) -> None:
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find param file: {param_path}")
    
    replacements = {"adaptive_timesteps": f"adaptive_timesteps {'true' if adaptive_timesteps else 'false'}"
    }

    if timestep_levels is not None:
        replacements["timestep_levels"] = f"timestep_levels {timestep_levels}"
    if timestep_eta is not None:
        replacements["timestep_eta"] = f"timestep_eta {timestep_eta}"
    
    lines = param_path.read_text().splitlines()
    updated =[]
    seen = set()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            updated.append(line)
            continue
        if stripped.startswith('#'):
            updated.append(line)
            continue

        key = stripped.split()[0]

        if key in replacements:
            updated.append(replacements[key])
            seen.add(key)
        else:
            updated.append(line)
    for key, value in replacements.items():
        if key not in seen:
            updated.append(value)
    
    param_path.write_text("\n".join(updated) + "\n")

# Write initial conditions in the exact format expected by the C++ reader:
#   mass x y z vx vy vz
# No header row is written

def rewrite_initial_conditions(rows: list[tuple[float, float, float, float, float, float, float]],
                             output_path: Path = DEFAULT_INITIAL_CONDITIONS_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for row in rows:
            if len(row) != 7:
                raise ValueError("Each initial-condition row must have 7 values: mass, x, y, z, vx, vy, vz")
            f.write(f"{row[0]:.16g} " f"{row[1]:.16g} {row[2]:.16g} {row[3]:.16g} " f"{row[4]:.16g} {row[5]:.16g} {row[6]:.16g}\n")

# ============================================================
# Run executable and other tests
# ============================================================

def run_executable(executable_path: Path = DEFAULT_EXECUTABLE_PATH) -> None:
    if not executable_path.exists():
        raise FileNotFoundError(f"Could not find executable: {executable_path}")
    
    result = subprocess.run([str(executable_path)], cwd=str(PROJECT_ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Simulation failed.")

    print("Simulation finished successfully")

# Run a convergence study using the current settings in data/param.txt
# Only the timestep is changed during the sweep.
# The original param.txt is restored afterward.

def run_timestep_scaling_study(dt_ref: float = 0.00025, 
                               dts: tuple = (0.01, 0.005, 0.0025, 0.00125), 
                               param_path: Path = DEFAULT_PARAM_PATH) -> None:
    
    params = read_param(param_path)

    runtime = float(params.get("runtime", 1.0))
    output_frequency = int(params.get("output_frequency", 10))
    integrator = params.get("integrator", "hernandez")
    coordinate_mode = params.get("coordinate_mode", "jacobi")
    G = float(params.get("gravitational_constant", 0.000296014912))

    print("\nConvergence test using current param.txt settings:")
    print(f"runtime              = {runtime}")
    print(f"output_frequency     = {output_frequency}")
    print(f"integrator           = {integrator}")
    print(f"coordinate_mode      = {coordinate_mode}")
    print(f"gravitational_constant = {G}")

    original_text = param_path.read_text()

    try:
        print("\nRunning reference solution...")

        rewrite_timestep_only(dt_ref, param_path)

        if DEFAULT_OUTPUT_PATH.exists():
            DEFAULT_OUTPUT_PATH.unlink()
        if DEFAULT_DIAGNOSTICS_PATH.exists():
            DEFAULT_DIAGNOSTICS_PATH.unlink()

        run_executable()

        df_ref = read_output(DEFAULT_OUTPUT_PATH)
        reference = compute_final_positions(df_ref)

        errors = []

        for dt in dts:
            print(f"\nRunning dt = {dt}")
            rewrite_timestep_only(dt, param_path)

            if DEFAULT_OUTPUT_PATH.exists():
                DEFAULT_OUTPUT_PATH.unlink()
            if DEFAULT_DIAGNOSTICS_PATH.exists():
                DEFAULT_DIAGNOSTICS_PATH.unlink()
            
            run_executable()

            df = read_output(DEFAULT_OUTPUT_PATH)
            err = error_rms_position(compute_final_positions(df), reference)

            errors.append(err)

            print(f"RMS position error: {err}")
        
        dts_array = np.array(dts)
        errors_array = np.array(errors)

        plt.figure(figsize=(7, 5), constrained_layout=True)
        plt.loglog(dts_array, errors_array, marker="o")
        plt.gca().invert_xaxis()
        plt.xlabel("Timestep dt")
        plt.ylabel("RMS final-position error")
        plt.title(f"Timestep convergence: {integrator}, {coordinate_mode}")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.show()

        print("\nConvergence Ratios:")
        for i in range(len(errors_array) - 1):
            ratio = errors_array[i] / errors_array[i + 1]
            print(f"{dts_array[i]} -> {dts_array[i + 1]} : ratio = {ratio}")
    
    finally:
        param_path.write_text(original_text)
        print("\nRestored original param.txt settings.")

# Run every benchmark test in every selected mode and save the resulting plots.
# This function temporarily overwrites data/initial_conditions.txt and data/param.txt.
# The original files should be restored at the end of the run.

def run_benchmark_suite(modes: list[dict] | None = None, output_dir: Path = DEFAULT_BENCHMARK_PLOT_DIR, use_diagnostics_csv: bool = False,) -> None:
    if modes is None:
        modes = [{"name": "hernandez_jacobi", "integrator": "hernandez", "coordinate_mode": "jacobi"},
                 {"name": "hernandez_cartesian", "integrator": "hernandez", "coordinate_mode": "cartesian"},
                 {"name": "leapfrog", "integrator": "leapfrog", "coordinate_mode": "cartesian"}]
    
    config = PlotConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_param_text = None
    original_initial_conditions_text = None

    if DEFAULT_PARAM_PATH.exists():
        original_param_text = DEFAULT_PARAM_PATH.read_text()
    if DEFAULT_INITIAL_CONDITIONS_PATH.exists():
        original_initial_conditions_text = DEFAULT_INITIAL_CONDITIONS_PATH.read_text()

    try:
        print("\nRunning benchmark suite...")

        total_runs = len(BENCHMARK_TESTS) * len(modes)
        run_number = 0

        for mode in modes:
            mode_dir = output_dir / mode["name"]
            mode_dir.mkdir(parents=True, exist_ok=True)

            for test in BENCHMARK_TESTS:
                run_number += 1

                print("\n" + "=" * 70)
                print(f"Run {run_number}/{total_runs}")
                print(f"Mode: {mode['name']}")
                print(f"Test: {test['name']}")
                print("=" * 70)

                rewrite_initial_conditions(test["initial_conditions"])
                rewrite_param(dt = test["dt"], 
                            runtime = test["runtime"], 
                            output_frequency = test["output_frequency"], 
                            integrator = mode["integrator"], 
                            coordinate_mode = mode["coordinate_mode"],
                            G = config.G,)
                
                if DEFAULT_OUTPUT_PATH.exists():
                    DEFAULT_OUTPUT_PATH.unlink()
                if DEFAULT_DIAGNOSTICS_PATH.exists():
                    DEFAULT_DIAGNOSTICS_PATH.unlink()
                
                run_executable()
                output = read_output(DEFAULT_OUTPUT_PATH)

                if use_diagnostics_csv:
                    diagnostics = read_diagnostics(DEFAULT_DIAGNOSTICS_PATH)
                    if diagnostics is None:
                        print("Falling back to recomputing diagnostics from output.csv.")
                        diagnostics = compute_diagnostics_from_output(output, config)
                else:
                    diagnostics = compute_diagnostics_from_output(output, config)
                
                plot_path = mode_dir / f"{test['name']}.png"
                plot_verification_suite(output_df = output, diagnostics = diagnostics, config = config, save_path = plot_path, show=False,)
                print(f"Saved plot to {plot_path}")
        
        print("\nBenchmark suite complete.")
        print(f"Plots saved to {output_dir}")
    
    finally:
        if original_param_text is not None:
            DEFAULT_PARAM_PATH.write_text(original_param_text)
        if original_initial_conditions_text is not None:
            DEFAULT_INITIAL_CONDITIONS_PATH.write_text(original_initial_conditions_text)
        print("\nRestored original param.txt and initial_conditions.txt settings.")

def run_adaptive_comparison_study(timestep_levels: int | None = None, timestep_eta: float | None = None, param_path: Path = DEFAULT_PARAM_PATH) -> None:
    config = PlotConfig()

    if not param_path.exists():
        raise FileNotFoundError(f"Could not find param file: {param_path}")
    
    original_param_text = param_path.read_text()
    params = read_param(param_path)

    coordinate_mode = params.get("coordinate_mode", "jacobi")
    integrator = params.get("integrator", "hernandez")

    if coordinate_mode != "jacobi":
        raise RuntimeError("Adaptive comparison currently only applies to coordinate_mode jacobi.")
    if timestep_levels is None:
        timestep_levels = int(params.get("timestep_levels", 2))
    if timestep_levels <= 0:
        timestep_levels = 2
    if timestep_eta is None:
        timestep_eta = float(params.get("timestep_eta", 0.001))
    
    print("\n=== Step 10.4 Adaptive Comparison Study ===")
    print(f"coordinate_mode    = {coordinate_mode}")
    print(f"integrator         = {integrator}")
    print(f"timestep_levels    = {timestep_levels}")
    print(f"timestep_eta       = {timestep_eta}")
    print("This test runs the same setup twice: adaptive off, then adaptive on.")

    results = {}

    try:
        for label, adaptive_flag in [("fixed", False), ("adaptive", True)]:
            print("\n" + "=" * 70)
            print(f"Running {label} case")
            print("=" * 70)

            rewrite_adaptive_settings(adaptive_timesteps=adaptive_flag, timestep_levels=timestep_levels, timestep_eta=timestep_eta, param_path=param_path)

            clear_simulation_outputs()
            run_executable()

            output_df = read_output(DEFAULT_OUTPUT_PATH)
            diagnostics = compute_diagnostics_from_output(output_df, config)
            final_positions = compute_final_positions(output_df)
            summary = error_diagnostic_metric_summary(diagnostics, config)

            results[label] = {
                "output" : output_df,
                "diagnostics" : diagnostics,
                "final_positions" : final_positions,
                "summary" : summary,
            }

        fixed_summary = results["fixed"]["summary"]
        adaptive_summary = results["adaptive"]["summary"]
        fixed_positions = results["fixed"]["final_positions"]
        adaptive_positions = results["adaptive"]["final_positions"]

        final_positions_difference = error_rms_position(adaptive_positions, fixed_positions)

        rows = []

        for metric in fixed_summary.keys():
            fixed_value = fixed_summary[metric]
            adaptive_value = adaptive_summary[metric]
            if (np.isfinite(fixed_value) and np.isfinite(adaptive_value) and adaptive_value != 0.0):
                improvement_factor = fixed_value / adaptive_value
            else:
                improvement_factor = float("nan")
            
            rows.append({"metric": metric, "fixed": fixed_value, "adaptive": adaptive_value, "fixed_over_adaptive": improvement_factor})
        
        comparison = pd.DataFrame(rows)
        comparison_path = PROJECT_ROOT / "adaptive_comparison.csv"
        comparison.to_csv(comparison_path, index=False)

        print("\n=== Adaptive Comparison Summary ===")
        print(comparison.to_string(index=False))
        print()
        print(f"RMS final-positions difference adaptive vs fixed: {final_positions_difference}")
        print(f"Saved comparison table to {comparison_path}")

        # Plot Energy-Error comparison
        fixed_diag = results["fixed"]["diagnostics"]
        adaptive_diag = results["adaptive"]["diagnostics"]
        fixed_time = fixed_diag["time"].to_numpy()
        adaptive_time = adaptive_diag["time"].to_numpy()
        fixed_dE = error_relative(fixed_diag["total_energy"].to_numpy(), config.epsilon)
        adaptive_dE = error_relative(adaptive_diag["total_energy"].to_numpy(), config.epsilon)

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        ax.semilogy(fixed_time, error_safe_log_values(fixed_dE, floor=1e-18), label="adaptive false")
        ax.semilogy(adaptive_time, error_safe_log_values(adaptive_dE, floor=1e-18), label="adaptive true")
        ax.set_title("Step 10.4 Adaptive vs Fixed Energy Error")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$|E - E_0|/|E_0|$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        plt.show()
    
    finally:
        param_path.write_text(original_param_text)
        print("\nRestored original param.txt settings.")

# ============================================================
# Clear .csv files
# ============================================================

def clear_simulation_outputs() -> None:
    if DEFAULT_OUTPUT_PATH.exists():
        DEFAULT_OUTPUT_PATH.unlink()
    if DEFAULT_DIAGNOSTICS_PATH.exists():
        DEFAULT_DIAGNOSTICS_PATH.unlink()