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

class SimulationRunError(RuntimeError):
    def __init__(self, message: str, stdout: str, stderr: str, returncode: int):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

# ============================================================
# Benchmark Tests:
#   Benchmark systems used to demonstrate both the strengths and limits of the current integrators.
#   These are written directly into data/initial_conditions.txt before each benchmark run.
# ============================================================

BENCHMARK_TESTS = [
    {
        "name": "Test1_Binary",
        "dt": 0.1,
        "runtime": 100000,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0, -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,  0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
        ],
    },
    {
        "name": "Test2_CircumbinaryTriple",
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
        "name": "Test3_StrongerPerturbedTriple",
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
        "name": "Test4_ScatteringEscape",
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
        "name": "Test5_Figure8",
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
        "name": "Test6_CloseEncounter",
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
        "name": "Test7_SolarSystem",
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
        "name": "Test8_SolarSystemInnerPlanets",
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
# Benchmark Modes
# ============================================================
# These are the current supported integrator/mode combinations.
# The benchmark suite runs every BENCHMARK_TESTS entry through every mode below.
#
# Cartesian:
#   - leapfrog fixed global timestep
#   - Hernandez fixed timestep with canonical/strength/auto pair order
#   - Hernandez adaptive block mode with canonical/strength/auto pair order
#
# Jacobi:
#   - leapfrog fixed/adaptive full-integrator subcycling
#   - Hernandez fixed/adaptive full-integrator subcycling
#   - Yoshida4 fixed/adaptive full-integrator subcycling

DEFAULT_BENCHMARK_MODES = [
    {
        "name": "cartesian_leapfrog_fixed",
        "integrator": "leapfrog",
        "coordinate_mode": "cartesian",
        "pair_order": "canonical",
        "adaptive_timesteps": False,
    },
    {
        "name": "cartesian_hernandez_canonical_fixed",
        "integrator": "hernandez",
        "coordinate_mode": "cartesian",
        "pair_order": "canonical",
        "adaptive_timesteps": False,
    },
    {
        "name": "cartesian_hernandez_strength_fixed",
        "integrator": "hernandez",
        "coordinate_mode": "cartesian",
        "pair_order": "strength",
        "adaptive_timesteps": False,
    },
    {
        "name": "cartesian_hernandez_auto_fixed",
        "integrator": "hernandez",
        "coordinate_mode": "cartesian",
        "pair_order": "auto",
        "adaptive_timesteps": False,
    },
    {
        "name": "cartesian_hernandez_canonical_adaptive",
        "integrator": "hernandez",
        "coordinate_mode": "cartesian",
        "pair_order": "canonical",
        "adaptive_timesteps": True,
        "timestep_levels": 4,
        "timestep_eta": 0.05,
        "timestep_refresh_interval": 1,
        "timestep_level_decrease_delay": 3,
    },
    {
        "name": "cartesian_hernandez_strength_adaptive",
        "integrator": "hernandez",
        "coordinate_mode": "cartesian",
        "pair_order": "strength",
        "adaptive_timesteps": True,
        "timestep_levels": 4,
        "timestep_eta": 0.05,
        "timestep_refresh_interval": 1,
        "timestep_level_decrease_delay": 3,
    },
    {
        "name": "cartesian_hernandez_auto_adaptive",
        "integrator": "hernandez",
        "coordinate_mode": "cartesian",
        "pair_order": "auto",
        "adaptive_timesteps": True,
        "timestep_levels": 4,
        "timestep_eta": 0.05,
        "timestep_refresh_interval": 1,
        "timestep_level_decrease_delay": 3,
    },
    {
        "name": "jacobi_hernandez_fixed",
        "integrator": "hernandez",
        "coordinate_mode": "jacobi",
        "pair_order": "canonical",
        "adaptive_timesteps": False,
    },
    {
        "name": "jacobi_hernandez_adaptive",
        "integrator": "hernandez",
        "coordinate_mode": "jacobi",
        "pair_order": "canonical",
        "adaptive_timesteps": True,
        "timestep_levels": 4,
        "timestep_eta": 0.05,
        "timestep_refresh_interval": 1,
        "timestep_level_decrease_delay": 3,
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

def read_initial_conditions(path: Path = DEFAULT_INITIAL_CONDITIONS_PATH) -> list[tuple[float, float, float, float, float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find initial-condition file: {path}")
    
    rows = []

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()

        if len(parts) != 7:
            raise ValueError(f"Initial-conditions line {line_number} must have 7 values: mass x y z vx vy vz")
        
        rows.append(tuple(float(value) for value in parts))

    if not rows:
        raise ValueError(f"No bodies were found in {path}")
    return rows

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

    # Diagnostic Table
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

# ============================================================
# REBOUND comparison helpers
# ============================================================

def rebound_require():
    try:
        import rebound
    except ImportError as exc:
        raise RuntimeError("REBOUND is not installed. Install it with: pip install rebound") from exc
    return rebound

def rebound_run_simulation(initial_conditions: list[tuple[float, float, float, float, float, float, float]], 
                            dt: float, runtime: float, 
                            output_frequency: int, 
                            G: float = 0.000296014912,
                            rebound_integrator: str = "whfast",
                            move_to_com: bool = False,
                            config: PlotConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the same physical Cartesian initial conditions wiht REBOUND.
    The returned output DataFrame uses the same columns as FewBodyNC output.csv:
        time, id, x, y, z, vx, vy, vz, mass
    Diagnostics are recomputed with this file's diagnostics routine.
    The benchmark table compares REBOUND and FewBodyNC using the same metric code.
    """

    rebound = rebound_require()

    if config is None:
        config = PlotConfig(G=G)
    else: config = PlotConfig(G=G, epsilon=config.epsilon, figure_size=config.figure_size, orbit_marker_size=config.orbit_marker_size, start_marker_size=config.start_marker_size)
    
    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = rebound_integrator
    sim.dt = dt
    
    for row in initial_conditions:
        mass, x, y, z, vx, vy, vz = row
        sim.add(m=mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    if move_to_com:
        sim.move_to_com()

    times = benchmark_output_times(dt=dt, runtime=runtime, output_frequency=output_frequency)
    output_rows = []
    
    for time_value in times:
        if abs(float(time_value) - sim.t) > 1e-15:
            try:
                sim.integrate(float(time_value), exact_finish_time=1)
            except:
                sim.integrate(float(time_value))
        for body_id, particle in enumerate(sim.particles):
            output_rows.append({"time": float(time_value),
                                "id": body_id,
                                "x": particle.x,
                                "y": particle.y,
                                "z": particle.z,
                                "vx": particle.vx,
                                "vy": particle.vy,
                                "vz": particle.vz,
                                "mass": particle.m,})
    
    output = pd.DataFrame(output_rows)
    output = output.sort_values(by=["time", "id"]).reset_index(drop=True)
    diagnostics = compute_diagnostics_from_output(output, config)

    return output, diagnostics

# ============================================================
# Helper Functions
# ============================================================

def benchmark_output_times(dt: float, runtime: float, output_frequency: int) -> np.ndarray:
    if dt <= 0.0:
        raise ValueError("dt must be positive for REBOUND comparison")
    if runtime < 0.0:
        raise ValueError("runtime must be non-negative for REBOUND comparison")
    
    output_frequency = max(int(output_frequency), 1)
    total_steps = int(round(runtime / dt))
    step_numbers = list(range(0, total_steps + 1, output_frequency))

    if not step_numbers or step_numbers[-1] != total_steps:
        step_numbers.append(total_steps)
    
    return np.array(step_numbers, dtype=float) * float(dt)

def append_benchmark_row(benchmark_rows: list[dict],
                         run_number: int,
                         mode: dict,
                         test: dict,
                         status: str,
                         error: str = "",
                         returncode: float = 0,
                         stdout_tail: str = "",
                         stderr_tail: str = "",
                         plot_path: Path | None = None,
                         diagnostics: pd.DataFrame | None = None,
                         engine: str = "fewbodync",
                         failure_type: str = "",
                         failure_message: str = "",
                         rebound_comparison: dict | None = None) -> None:
    row = {"run_number": run_number,
           "engine": engine,
           "mode": mode["name"],
           "integrator": mode["integrator"],
           "coordinate_mode": mode["coordinate_mode"],
           "pair_order": mode.get("pair_order", "canonical"),
           "adaptive_timesteps": mode.get("adaptive_timesteps", False),
           "timestep_levels": mode.get("timestep_levels", np.nan),
           "timestep_eta": mode.get("timestep_eta", np.nan),
           "timestep_refresh_interval": mode.get("timestep_refresh_interval", np.nan),
           "timestep_level_decrease_delay": mode.get("timestep_level_decrease_delay", np.nan),
           "test": test["name"],
           "dt": test["dt"],
           "runtime": test["runtime"],
           "output_frequency": test["output_frequency"],
           "status": status,
           "error": error,
           "failure_type": failure_type,
           "failure_message": failure_message,
           "returncode": returncode,
           "stdout_tail": stdout_tail,
           "stderr_tail": stderr_tail,
           "plot_path": str(plot_path) if plot_path is not None else "",
           "final_time": np.nan,
           "initial_energy": np.nan,
           "final_energy": np.nan,
           "max_dE_over_E0": np.nan,
           "final_dE_over_E0": np.nan,
           "initial_angular_momentum": np.nan,
           "final_angular_momentum": np.nan,
           "max_dL_over_L0": np.nan,
           "final_dL_over_L0": np.nan,
           "initial_linear_momentum": np.nan,
           "final_linear_momentum": np.nan,
           "max_dP": np.nan,
           "final_dP": np.nan,
           "initial_com_drift": np.nan,
           "final_com_drift": np.nan,
           "max_dRcm": np.nan,
           "final_dRcm": np.nan,
           "rebound_compare": False,
           "rebound_integrator": "",
           "rebound_status": "not_requested",
           "rebound_error": ""}
    
    if diagnostics is not None and not diagnostics.empty:
        config = PlotConfig()
        energy = diagnostics["total_energy"].to_numpy()
        angular = diagnostics["angular_momentum"].to_numpy()
        linear = diagnostics["linear_momentum"].to_numpy()
        com = diagnostics["com_drift"].to_numpy()
        time = diagnostics["time"].to_numpy()
        dE = error_relative(energy, config.epsilon)
        dL = error_relative(angular, config.epsilon)
        dP = error_absolute(linear)
        dRcm = error_absolute(com)

        row.update({"final_time": time[-1] if len(time) else np.nan,
                    "initial_energy": energy[0] if len(energy) else np.nan,
                    "final_energy": energy[-1] if len(energy) else np.nan,
                    "max_dE_over_E0": compute_finite_max(dE),
                    "final_dE_over_E0": float(dE[-1]) if len(dE) else np.nan,
                    "initial_angular_momentum": angular[0] if len(angular) else np.nan,
                    "final_angular_momentum": angular[-1] if len(angular) else np.nan,
                    "max_dL_over_L0": compute_finite_max(dL),
                    "final_dL_over_L0": float(dL[-1]) if len(dL) else np.nan,
                    "initial_linear_momentum": linear[0] if len(linear) else np.nan,
                    "final_linear_momentum": linear[-1] if len(linear) else np.nan,
                    "max_dP": compute_finite_max(dP),
                    "final_dP": float(dP[-1]) if len(dP) else np.nan,
                    "initial_com_drift": com[0] if len(com) else np.nan,
                    "final_com_drift": com[-1] if len(com) else np.nan,
                    "max_dRcm": compute_finite_max(dRcm),
                    "final_dRcm": float(dRcm[-1]) if len(dRcm) else np.nan,})
    if rebound_comparison is not None:
        row.update(rebound_comparison)
    
    benchmark_rows.append(row)

def _safe_ratio(numerator: float, denominator: float, epsilon:float = 1e-300) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= epsilon:
        return float("nan")
    return float(numerator / denominator)

def compare_diagnostics_to_rebound(fewbody_diagnostics: pd.DataFrame, rebound_diagnostics: pd.DataFrame, config: PlotConfig, rebound_integrator: str) -> dict:
    comparison = {"rebound_compare": True, "rebound_integrator": rebound_integrator, "rebound_status": "success", "rebound_error": ""}

    few = fewbody_diagnostics.copy()
    ref = rebound_diagnostics.copy()
    few["time_key"] = np.round(few["time"].to_numpy(dtype=float), 12)
    ref["time_key"] = np.round(ref["time"].to_numpy(dtype=float), 12)
    merged = few.merge(ref, on="time_key", suffixes=("_fewbody", "_rebound"))

    comparison["rebound_matched_diagnostic_rows"] = int(len(merged))

    if merged.empty:
        comparison["rebound_status"] = "warning"
        comparison["rebound_error"] = "No matching diagnostic output times"
        return comparison
    
    def add_series_comparison(metric:str, label:str) -> None:
        few_values = merged[f"{metric}_fewbody"].to_numpy(dtype=float)
        ref_values = merged[f"{metric}_rebound"].to_numpy(dtype=float)
        diff = few_values - ref_values
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / np.maximum(np.abs(ref_values), config.epsilon)

        comparison[f"rms_{label}_difference_vs_rebound"] = float(np.sqrt(np.nanmean(diff * diff)))
        comparison[f"max_abs_{label}_difference_vs_rebound"] = compute_finite_max(abs_diff)
        comparison[f"max_rel_{label}_difference_vs_rebound"] = compute_finite_max(rel_diff)
        comparison[f"final_{label}_difference_vs_rebound"] = float(diff[-1])
        comparison[f"final_abs_{label}_difference_vs_rebound"] = float(abs_diff[-1])
        comparison[f"final_rel_{label}_difference_vs_rebound"] = float(rel_diff[-1])
    
    add_series_comparison("total_energy", "energy")
    add_series_comparison("angular_momentum", "angular_momentum")
    add_series_comparison("linear_momentum", "linear_momentum")
    add_series_comparison("com_drift", "com_drift")

    rebound_summary = error_diagnostic_metric_summary(rebound_diagnostics, config)
    few_summary = error_diagnostic_metric_summary(fewbody_diagnostics, config)

    for key, value in rebound_summary.items():
        comparison[f"rebound_{key}"] = value
    for key, few_value in few_summary.items():
        rebound_value = rebound_summary.get(key, np.nan)
        comparison[f"difference_{key}_vs_rebound"] = (float(few_value - rebound_value)
                                                      if np.isfinite(few_value) and np.isfinite(rebound_value) else float("nan"))
        comparison[f"ratio_{key}_to_rebound"] = _safe_ratio(few_value, rebound_value, config.epsilon)

    return comparison

def compare_output_to_rebound(fewbody_output: pd.DataFrame, rebound_output: pd.DataFrame, comparison: dict) -> dict:
    few = fewbody_output.copy()
    ref = rebound_output.copy()
    few["time_key"] = np.round(few["time"].to_numpy(dtype=float), 12)
    ref["time_key"] = np.round(ref["time"].to_numpy(dtype=float), 12)
    merged = few.merge(ref, on=["time_key", "id"], suffixes=("_fewbody", "_rebound"))
    comparison["rebound_matched_diagnostic_rows"] = int(len(merged))

    if merged.empty:
        comparison["rebound_status"] = "warning"
        comparison["rebound_error"] = "No matching output rows for position/velocity comparison"
        return comparison
    
    dx = merged["x_fewbody"].to_numpy(dtype=float) - merged["x_rebound"].to_numpy(dtype=float)
    dy = merged["y_fewbody"].to_numpy(dtype=float) - merged["y_rebound"].to_numpy(dtype=float)
    dz = merged["z_fewbody"].to_numpy(dtype=float) - merged["z_rebound"].to_numpy(dtype=float)
    dvx = merged["vx_fewbody"].to_numpy(dtype=float) - merged["vx_rebound"].to_numpy(dtype=float)
    dvy = merged["vy_fewbody"].to_numpy(dtype=float) - merged["vy_rebound"].to_numpy(dtype=float)
    dvz = merged["vz_fewbody"].to_numpy(dtype=float) - merged["vz_rebound"].to_numpy(dtype=float)
    
    position_error = np.sqrt(dx * dx + dy * dy + dz * dz)
    velocity_error = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)

    comparison["rms_position_error_vs_rebound_all_outputs"] = float(np.sqrt(np.nanmean(position_error * position_error)))
    comparison["max_position_error_vs_rebound_all_outputs"] = compute_finite_max(position_error)
    comparison["rms_velocity_error_vs_rebound_all_outputs"] = float(np.sqrt(np.nanmean(velocity_error * velocity_error)))
    comparison["max_velocity_error_vs_rebound_all_outputs"] = compute_finite_max(velocity_error)

    final_time_key = merged["time_key"].max()
    final_rows = merged[merged["time_key"] == final_time_key]

    fdx = final_rows["x_fewbody"].to_numpy(dtype=float) - final_rows["x_rebound"].to_numpy(dtype=float)
    fdy = final_rows["y_fewbody"].to_numpy(dtype=float) - final_rows["y_rebound"].to_numpy(dtype=float)
    fdz = final_rows["z_fewbody"].to_numpy(dtype=float) - final_rows["z_rebound"].to_numpy(dtype=float)
    fdvx = final_rows["vx_fewbody"].to_numpy(dtype=float) - final_rows["vx_rebound"].to_numpy(dtype=float)
    fdvy = final_rows["vy_fewbody"].to_numpy(dtype=float) - final_rows["vy_rebound"].to_numpy(dtype=float)
    fdvz = final_rows["vz_fewbody"].to_numpy(dtype=float) - final_rows["vz_rebound"].to_numpy(dtype=float)

    final_position_error = np.sqrt(fdx * fdx + fdy * fdy + fdz * fdz)
    final_velocity_error = np.sqrt(fdvx * fdvx + fdvy * fdvy + fdvz * fdvz)

    comparison["final_rms_position_error_vs_rebound"] = float(np.sqrt(np.nanmean(final_position_error * final_position_error)))
    comparison["final_max_position_error_vs_rebound"] = compute_finite_max(final_position_error)
    comparison["final_rms_velocity_error_vs_rebound"] = float(np.sqrt(np.nanmean(final_velocity_error * final_velocity_error)))
    comparison["final_max_velocity_error_vs_rebound"] = compute_finite_max(final_velocity_error)


    return comparison

def compare_fewbody_to_rebound(fewbody_output: pd.DataFrame, 
                               fewbody_diagnostics: pd.DataFrame, 
                               rebound_output: pd.DataFrame, 
                               rebound_diagnostics: pd.DataFrame, 
                               config: PlotConfig, 
                               rebound_integrator: str) -> dict:
    comparison = compare_diagnostics_to_rebound(fewbody_diagnostics=fewbody_diagnostics, rebound_diagnostics=rebound_diagnostics, config=config, rebound_integrator=rebound_integrator)
    comparison = compare_output_to_rebound(fewbody_output=fewbody_output, rebound_output=rebound_output, comparison=comparison)
    return comparison

# ============================================================
# Compute Functions
# ============================================================

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

    if save_path is not None:
        title = f"FewBodyNc Verification Suite: {save_path.parent.name} / {save_path.stem}"
    else:
        title = "FewBodyNC Verification Suite"
        
    fig.suptitle(title, fontsize=16)
    
    if save_path is not None:
        save_path = Path(save_path)
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
                  pair_order: str = "canonical", 
                  adaptive_timesteps: bool | None = None,
                  timestep_levels: int | None = None,
                  timestep_eta: float | None = None,
                  timestep_refresh_interval: int | None = None,
                  timestep_level_decrease_delay: int | None = None,
                  param_path: Path = DEFAULT_PARAM_PATH) -> None:
    
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find param file: {param_path}")

    replacements = {"output_frequency": f"output_frequency {output_frequency}",
                    "runtime": f"runtime {runtime}",  
                    "timestep": f"timestep {dt}", 
                    "integrator": f"integrator {integrator}", 
                    "coordinate_mode": f"coordinate_mode {coordinate_mode}", 
                    "gravitational_constant": f"gravitational_constant {G}",
                    "pair_order": f"pair_order {pair_order}"}
    
    if adaptive_timesteps is not None:
        replacements["adaptive_timesteps"] = f"adaptive_timesteps {'true' if adaptive_timesteps else 'false'}"
    if timestep_levels is not None:
        replacements["timestep_levels"] = f"timestep_levels {timestep_levels}"
    if timestep_eta is not None:
        replacements["timestep_eta"] = f"timestep_eta {timestep_eta}"
    if timestep_refresh_interval is not None:
        replacements["timestep_refresh_interval"] = f"timestep_refresh_interval {timestep_refresh_interval}"
    if timestep_level_decrease_delay is not None:
        replacements["timestep_level_decrease_delay"] = f"timestep_level_decrease_delay {timestep_level_decrease_delay}"

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
    
    replacements = {"adaptive_timesteps": f"adaptive_timesteps {'true' if adaptive_timesteps else 'false'}"}

    if timestep_levels is not None:
        replacements["timestep_levels"] = f"timestep_levels {timestep_levels}"
    if timestep_eta is not None:
        replacements["timestep_eta"] = f"timestep_eta {timestep_eta}"
    
    lines = param_path.read_text().splitlines()
    updated = []
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

def rewrite_initial_conditions(rows: list[tuple[float, float, float, float, float, float, float]], output_path: Path = DEFAULT_INITIAL_CONDITIONS_PATH) -> None:
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
        raise SimulationRunError("Simulation failed.", stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)

    print("Simulation finished successfully")

# Run a convergence study using the current settings in data/param.txt
# Only the timestep is changed during the sweep.
# The original param.txt is restored afterward.

def run_timestep_scaling_study(dt_ref: float = 0.00025,  
                               dts: tuple = (0.01, 0.005, 0.0025, 0.00125),  
                               param_path: Path = DEFAULT_PARAM_PATH,
                               rebound_compare: bool = False,
                               rebound_integrator: str = "whfast",
                               rebound_move_to_com: bool = False) -> None:
    params = read_param(param_path)

    runtime = float(params.get("runtime", 1.0))
    output_frequency = int(params.get("output_frequency", 10))
    integrator = params.get("integrator", "hernandez")
    coordinate_mode = params.get("coordinate_mode", "cartesian")
    G = float(params.get("gravitational_constant", 0.000296014912))

    print("\nConvergence test using current param.txt settings:")
    print(f"runtime              = {runtime}")
    print(f"output_frequency     = {output_frequency}")
    print(f"integrator           = {integrator}")
    print(f"coordinate_mode      = {coordinate_mode}")
    print(f"gravitational_constant = {G}")

    if rebound_compare:
        print(f"REBOUND comparison   = true ({rebound_integrator})")
        initial_conditions = read_initial_conditions(DEFAULT_INITIAL_CONDITIONS_PATH)
    else:
        initial_conditions = []

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
        rebound_errors = []

        for dt in dts:
            print(f"\nRunning dt = {dt}")
            rewrite_timestep_only(dt, param_path)

            if DEFAULT_OUTPUT_PATH.exists():
                DEFAULT_OUTPUT_PATH.unlink()
            if DEFAULT_DIAGNOSTICS_PATH.exists():
                DEFAULT_DIAGNOSTICS_PATH.unlink()
            
            run_executable()

            df = read_output(DEFAULT_OUTPUT_PATH)
            final_positions = compute_final_positions(df)
            err = error_rms_position(compute_final_positions(df), reference)

            errors.append(err)

            print(f"RMS position error vs FewBodyNC dt_ref: {err}")

            if rebound_compare:
                rebound_output, _ = rebound_run_simulation(initial_conditions=initial_conditions,
                                                           dt=dt,
                                                           runtime=runtime,
                                                           output_frequency=output_frequency,
                                                           G=G,
                                                           rebound_integrator=rebound_integrator,
                                                           move_to_com=rebound_move_to_com)
                
                rebound_final_positions = compute_final_positions(rebound_output)
                rebound_error = error_rms_position(final_positions, rebound_final_positions)
                rebound_errors.append(rebound_error)

                print(f"RMS final-position difference vs REBOUND {rebound_integrator}: {rebound_error}")
        
        dts_array = np.array(dts)
        errors_array = np.array(errors)

        plt.figure(figsize=(7, 5), constrained_layout=True)
        plt.loglog(dts_array, errors_array, marker="o")

        if rebound_compare:
            rebound_errors_array = np.array(rebound_errors)
            plt.loglog(dts_array, rebound_errors_array, marker='s', label=f"FewBodyNC vs REBOUND {rebound_integrator}")
            plt.legend()

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
        if rebound_compare:
            comparison = pd.DataFrame({"dt": dts_array, "rms_vs_fewbody_dt_ref": errors_array, f"rms_vs_rebound_{rebound_integrator}": np.array(rebound_errors)})
            comparison_path = PROJECT_ROOT / "rebound_convergence_comparison.csv"
            comparison.to_csv(comparison_path, index=False)
            print("\nREBOUND convergence comparison:")
            print(comparison.to_string(index=False))
            print(f"Saved REBOUND convergence comparison to {comparison_path}")
    
    finally:
        param_path.write_text(original_text)
        print("\nRestored original param.txt settings.")

# Run every benchmark test in every selected mode and save the resulting plots.
# This function temporarily overwrites data/initial_conditions.txt and data/param.txt.
# The original files should be restored at the end of the run.

def run_benchmark_suite(modes: list[dict] | None = None, 
                        output_dir: Path = DEFAULT_BENCHMARK_PLOT_DIR, 
                        use_diagnostics_csv: bool = True,
                        rebound_compare: bool = False,
                        rebound_integrator: str = "whfast",
                        rebound_move_to_com: bool = False,
                        save_rebound_reference_plots: bool = False) -> None:
    if modes is None:
        modes = DEFAULT_BENCHMARK_MODES
    
    config = PlotConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_param_text = None
    original_initial_conditions_text = None

    if DEFAULT_PARAM_PATH.exists():
        original_param_text = DEFAULT_PARAM_PATH.read_text()
    if DEFAULT_INITIAL_CONDITIONS_PATH.exists():
        original_initial_conditions_text = DEFAULT_INITIAL_CONDITIONS_PATH.read_text()
    
    rebound_cache: dict[str, dict] = {}
    rebound_reference_rows = []
    def get_rebound_reference(test: dict) -> dict:
        if test["name"] in rebound_cache:
            return rebound_cache[test["name"]]
        print("\n" + "-" * 70)
        print(f"Computing REBOUND reference for {test['name']} using {rebound_integrator}")
        print("-" * 70)

        output, diagnostics = rebound_run_simulation(initial_conditions=test["initial_conditions"], 
                                                     dt=test["dt"], 
                                                     runtime=test["runtime"],
                                                     output_frequency=test["output_frequency"],
                                                     G=config.G,
                                                     rebound_integrator=rebound_integrator,
                                                     move_to_com=rebound_move_to_com,
                                                     config=config)
        reference = {"output": output, "diagnostics": diagnostics}
        rebound_cache[test["name"]] = reference
        summary = error_diagnostic_metric_summary(diagnostics, config)
        reference_row = {"test": test["name"],
                         "rebound_integrator": rebound_integrator,
                         "dt": test["dt"],
                         "runtime": test["runtime"],
                         "output_frequency": test["output_frequency"],
                         "initial_energy": diagnostics["total_energy"].iloc[0] if not diagnostics.empty else np.nan,
                         "final_energy": diagnostics["total_energy"].iloc[-1] if not diagnostics.empty else np.nan,
                         "initial_angular_momentum": diagnostics["angular_momentum"].iloc[0] if not diagnostics.empty else np.nan,
                         "final_angular_momentum": diagnostics["angular_momentum"].iloc[-1] if not diagnostics.empty else np.nan,
                         "initial_linear_momentum": diagnostics["linear_momentum"].iloc[0] if not diagnostics.empty else np.nan,
                         "final_linear_momentum": diagnostics["linear_momentum"].iloc[-1] if not diagnostics.empty else np.nan,
                         "initial_com_drift": diagnostics["com_drift"].iloc[0] if not diagnostics.empty else np.nan,
                         "final_com_drift": diagnostics["com_drift"].iloc[-1] if not diagnostics.empty else np.nan}
        reference_row.update({f"rebound_{key}": value for key, value in summary.items()})
        rebound_reference_rows.append(reference_row)

        if save_rebound_reference_plots:
            rebound_dir = output_dir / f"rebound_reference_{rebound_integrator}"
            rebound_dir.mkdir(parents=True, exist_ok=True)
            plot_path = rebound_dir / f"{test['name']}.png"
            plot_verification_suite(output_df=output, diagnostics=diagnostics, config=config, save_path=plot_path, show=False)
            print(f"Saved REBOUND reference plot to {plot_path}")

        return reference

    try:
        print("\nRunning benchmark suite...")

        total_runs = len(BENCHMARK_TESTS) * len(modes)
        run_number = 0
        benchmark_rows = []

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

                rebound_reference = None
                rebound_comparison = None
                if rebound_compare:
                    try:
                        rebound_reference = get_rebound_reference(test)
                    except Exception as exc:
                        print(f"REBOUND reference failed for {test['name']}: {exc}")
                        rebound_comparison = {"rebound_compare": True, "rebound_integrator": rebound_integrator, "rebound_status": "failed", "rebound_error": str(exc)}

                rewrite_initial_conditions(test["initial_conditions"])
                rewrite_param(dt = test["dt"], 
                            runtime = test["runtime"], 
                            output_frequency = test["output_frequency"], 
                            integrator = mode["integrator"], 
                            coordinate_mode = mode["coordinate_mode"],
                            G = config.G,
                            pair_order = mode.get("pair_order", "canonical"),
                            adaptive_timesteps = mode.get("adaptive_timesteps", False),
                            timestep_levels = mode.get("timestep_levels"),
                            timestep_eta = mode.get("timestep_eta"),
                            timestep_refresh_interval = mode.get("timestep_refresh_interval"),
                            timestep_level_decrease_delay = mode.get("timestep_level_decrease_delay"))
                
                if DEFAULT_OUTPUT_PATH.exists():
                    DEFAULT_OUTPUT_PATH.unlink()
                if DEFAULT_DIAGNOSTICS_PATH.exists():
                    DEFAULT_DIAGNOSTICS_PATH.unlink()
                
                try:
                    run_executable()
                except RuntimeError as exc:
                    stdout_tail = exc.stdout[-4000:] if isinstance(exc, SimulationRunError) else ""
                    stderr_tail = exc.stderr[-4000:] if isinstance(exc, SimulationRunError) else ""
                    returncode = exc.returncode if isinstance(exc, SimulationRunError) else np.nan
                    print(f"Benchmark run failed: {exc}")
                    append_benchmark_row(benchmark_rows=benchmark_rows,
                                         run_number=run_number,
                                         mode=mode,
                                         test=test,
                                         status="failed",
                                         error=str(exc),
                                         returncode=returncode,
                                         stdout_tail=stdout_tail,
                                         stderr_tail=stderr_tail,
                                         engine="fewbodync",
                                         failure_type=type(exc).__name__,
                                         failure_message=str(exc),
                                         rebound_comparison=rebound_comparison)
                    continue

                output = read_output(DEFAULT_OUTPUT_PATH)

                if use_diagnostics_csv:
                    diagnostics = read_diagnostics(DEFAULT_DIAGNOSTICS_PATH)
                    if diagnostics is None:
                        print("Falling back to recomputing diagnostics from output.csv.")
                        diagnostics = compute_diagnostics_from_output(output, config)
                else:
                    diagnostics = compute_diagnostics_from_output(output, config)
                if rebound_compare and rebound_reference is not None:
                    try:
                        rebound_comparison = compare_fewbody_to_rebound(
                            fewbody_output=output,
                            fewbody_diagnostics=diagnostics,
                            rebound_output=rebound_reference["output"],
                            rebound_diagnostics=rebound_reference["diagnostics"],
                            config=config,
                            rebound_integrator=rebound_integrator)
                    except Exception as exc:
                        print(f"Comparison to REBOUND failed: {exc}")
                        rebound_comparison = {"rebound_compare": True, "rebound_integrator": rebound_integrator, "rebound_status": "failed", "rebound_error": str(exc)}
                
                plot_path = mode_dir / f"{test['name']}.png"
                plot_verification_suite(output_df = output, diagnostics = diagnostics, config = config, save_path = plot_path, show=False,)
                print(f"Saved plot to {plot_path}")

                append_benchmark_row(benchmark_rows=benchmark_rows,
                                     run_number=run_number,
                                     mode=mode,
                                     test=test,
                                     status="success",
                                     plot_path=plot_path,
                                     diagnostics=diagnostics,
                                     engine="fewbodync",
                                     rebound_comparison=rebound_comparison)
        
        print("\nBenchmark suite complete.")
        print(f"Plots saved to {output_dir}")

        if not benchmark_rows:
            print("No benchmark results were collected")
            return
        
        summary = pd.DataFrame(benchmark_rows)
        summary_path = output_dir / "benchmark_summary.csv"
        summary.to_csv(summary_path, index=False)
        if rebound_reference_rows:
            rebound_reference_summary = pd.DataFrame(rebound_reference_rows)
            rebound_reference_path = output_dir / f"rebound_reference_{rebound_integrator}_summary.csv"
            rebound_reference_summary.to_csv(rebound_reference_path, index=False)
            print(f"Saved REBOUND reference summary CSV to {rebound_reference_path}")
        status_counts = summary["status"].value_counts(dropna=False)
        print("\n" + "=" * 90)
        print("BENCHMARK STATUS SUMMARY")
        print("=" * 90)
        print(f"Total runs: {len(summary)}")
        print(f"Successful runs: {int(status_counts.get('success', 0))}")
        print(f"Failed runs: {int(status_counts.get('failed', 0))}")
        print(f"Skipped runs: {int(status_counts.get('skipped', 0))}")
        print(f"Warning runs: {int(status_counts.get('warning', 0))}")
        print(f"Saved summary CSV to {summary_path}")

        failed_summary = summary[summary["status"] == "failed"].copy()
        if not failed_summary.empty:
            failures_path = output_dir / "benchmark_failures.csv"
            failed_summary.to_csv(failures_path, index=False)
            print(f"Saved failure CSV to {failures_path}")

        successful_summary = summary[summary["status"] == "success"].copy()

        if successful_summary.empty:
            print("No successful benchmark runs were collected.")
            return

        base_comparison_columns = [
            "test",
            "mode",
            "integrator",
            "coordinate_mode",
            "pair_order",
            "adaptive_timesteps",
            "timestep_levels",
            "timestep_eta",
            "timestep_refresh_interval",
            "timestep_level_decrease_delay",
            "dt",
            "runtime",
            "status",
            "failure_type",
            "failure_message",
            "returncode",
            "max_dE_over_E0",
            "final_dE_over_E0",
            "max_dL_over_L0",
            "final_dL_over_L0",
            "max_dP",
            "final_dP",
            "max_dRcm",
            "final_dRcm",
            "final_com_drift",
            "plot_path",
        ]
        rebound_comparison_columns = [
            "rebound_compare",
            "rebound_integrator",
            "rebound_status",
            "rebound_error",
            "final_rms_position_error_vs_rebound",
            "final_max_position_error_vs_rebound",
            "final_rms_velocity_error_vs_rebound",
            "final_max_velocity_error_vs_rebound",
            "rms_position_error_vs_rebound_all_outputs",
            "max_position_error_vs_rebound_all_outputs",
            "rms_velocity_error_vs_rebound_all_outputs",
            "max_velocity_error_vs_rebound_all_outputs",
            "final_abs_energy_difference_vs_rebound",
            "final_rel_energy_difference_vs_rebound",
            "max_abs_energy_difference_vs_rebound",
            "max_rel_energy_difference_vs_rebound",
            "rebound_max_dE_over_E0",
            "ratio_max_dE_over_E0_to_rebound",
            "rebound_final_dE_over_E0",
            "ratio_final_dE_over_E0_to_rebound",
            "rebound_max_dL_over_L0",
            "ratio_max_dL_over_L0_to_rebound",
            "rebound_max_dP",
            "ratio_max_dP_to_rebound",
            "rebound_max_dRcm",
            "ratio_max_dRcm_to_rebound",
        ]
        comparison_columns = [column for column in base_comparison_columns + rebound_comparison_columns if column in summary.columns]

        # Comparison Table
        print("\n" + "=" * 90)
        print("BENCHMARK COMPARISON TABLE")
        print("=" * 90)
        print(f"Saved summary CSV to {summary_path}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220, "display.float_format", "{:.6e}".format):
            print(summary[comparison_columns].to_string(index=False))

        # Best Runs
        print("\n" + "=" * 90)
        print("BEST MODE PER TEST BY MAX |dE/E0|")
        print("=" * 90)
        best_rows = (successful_summary.sort_values(["test", "max_dE_over_E0", "max_dL_over_L0", "max_dP", "max_dRcm"]).groupby("test", as_index=False).first())
        best_columns = ["test", "mode", "pair_order", "adaptive_timesteps", "max_dE_over_E0", "max_dL_over_L0", "max_dP", "max_dRcm"]
        best_path = output_dir / "best_by_energy.csv"
        best_rows.to_csv(best_path, index=False)
        print(f"Saved best-by-energy CSV to {best_path}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 180, "display.float_format", "{:.6e}".format):
            print(best_rows[best_columns].to_string(index=False))
        
        # Worst Runs
        print("\n" + "=" * 90)
        print("WORST 10 RUNS BY MAX |dE/E0|")
        worst_rows = successful_summary.sort_values("max_dE_over_E0", ascending=False).head(10)
        worst_path = output_dir / "worst_by_energy.csv"
        worst_rows.to_csv(worst_path, index=False)
        print(f"Saved worst-by-energy CSV to {worst_path}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 180, "display.float_format", "{:.6e}".format):
            print(worst_rows[best_columns].to_string(index=False))
        
        if rebound_compare and "final_rms_position_error_vs_rebound" in successful_summary.columns:
            print("\n" + "=" * 90)
            print("BEST MODE PER TEST VS REBOUND BY FINAL RMS POSITION ERROR")
            print("=" * 90)
            valid_vs_rebound = successful_summary[np.isfinite(successful_summary["final_rms_position_error_vs_rebound"])].copy()
            if not valid_vs_rebound.empty:
                best_vs_rebound = (valid_vs_rebound.sort_values([
                        "test",
                        "final_rms_position_error_vs_rebound",
                        "final_rms_velocity_error_vs_rebound",
                        "max_rel_energy_difference_vs_rebound",
                    ]).groupby("test", as_index=False).first())

                best_vs_rebound_path = output_dir / f"best_vs_rebound_{rebound_integrator}.csv"
                best_vs_rebound.to_csv(best_vs_rebound_path, index=False)
                print(f"Saved best-vs-REBOUND CSV to {best_vs_rebound_path}")
                best_rebound_columns = [column
                    for column in [
                        "test",
                        "mode",
                        "pair_order",
                        "adaptive_timesteps",
                        "final_rms_position_error_vs_rebound",
                        "final_rms_velocity_error_vs_rebound",
                        "final_rel_energy_difference_vs_rebound",
                        "ratio_max_dE_over_E0_to_rebound",]
                    if column in best_vs_rebound.columns]
                with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 220, "display.float_format", "{:.6e}".format,):
                    print(best_vs_rebound[best_rebound_columns].to_string(index=False))
            else:
                print("No finite REBOUND comparison rows were available.")

        # Ranking by median max
        print("\n" + "=" * 90)
        print("MODE RANKINGS BY MEDIAN MAX |dE/E0|")
        print("=" * 90)
        mode_rankings = (successful_summary.groupby("mode", as_index=False).agg(
            median_max_dE_over_E0=("max_dE_over_E0", "median"),
            mean_max_dE_over_E0=("max_dE_over_E0", "mean"),
            worst_max_dE_over_E0=("max_dE_over_E0", "max"),
            median_max_dL_over_L0=("max_dL_over_L0", "median"),
            median_max_dP=("max_dP", "median"),
            median_max_dRcm=("max_dRcm", "median"),
            successful_runs=("status", "count")).sort_values("median_max_dE_over_E0")
        )
        mode_rankings_path = output_dir / "mode_rankings.csv"
        mode_rankings.to_csv(mode_rankings_path, index=False)
        print(f"Saved mode rankings CSV to {mode_rankings_path}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 180, "display.float_format", "{:.6e}".format):
            print(mode_rankings.to_string(index=False))

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
    
    print("\n=== Adaptive Comparison Study ===")
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
        ax.set_title("Adaptive vs Fixed Energy Error")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$|E - E_0|/|E_0|$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        plt.show()
    
    finally:
        param_path.write_text(original_param_text)
        print("\nRestored original param.txt settings.")

# ============================================================
# Test Convergence and Benchmark
# ============================================================

def TestBenchmark(reboundcompare: bool = False, rebound_integrator: str = "whfast", rebound_move_to_com: bool = False, **kwargs) -> None:
    run_benchmark_suite(rebound_compare=reboundcompare, rebound_integrator=rebound_integrator, rebound_move_to_com=rebound_move_to_com, **kwargs)

def TestConvergence(reboundcompare: bool = False, rebound_integrator: str = "whfast", rebound_move_to_com: bool = False, **kwargs) -> None:
    run_timestep_scaling_study(rebound_compare=reboundcompare, rebound_integrator=rebound_integrator, rebound_move_to_com=rebound_move_to_com, **kwargs)


# ============================================================
# Clear .csv files
# ============================================================

def clear_simulation_outputs() -> None:
    if DEFAULT_OUTPUT_PATH.exists():
        DEFAULT_OUTPUT_PATH.unlink()
    if DEFAULT_DIAGNOSTICS_PATH.exists():
        DEFAULT_DIAGNOSTICS_PATH.unlink()