from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import subprocess
import math
import shutil
import time



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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output.csv"
DEFAULT_DIAGNOSTICS_PATH = PROJECT_ROOT / "diagnostics.csv"
DEFAULT_PARAM_PATH = PROJECT_ROOT / "data" / "param.txt"
DEFAULT_INITIAL_CONDITIONS_PATH = PROJECT_ROOT / "data" / "initial_conditions.txt"
DEFAULT_EXECUTABLE_PATH = PROJECT_ROOT / "few_body_nc.exe"
DEFAULT_BENCHMARK_PLOT_DIR = PROJECT_ROOT / "benchmark_plots"

# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class PlotConfig:
    gravitational_constant: float = 0.000296014912
    singularity_tolerance: float = 1.0e-14

    def __post_init__(self) -> None:
        if not math.isfinite(self.gravitational_constant) or self.gravitational_constant <= 0.0:
            raise ValueError("gravitational_constant must be finite and positive.")
        if not math.isfinite(self.singularity_tolerance) or self.singularity_tolerance <= 0.0:
            raise ValueError("singularity_tolerance must be finite and positive.")

@dataclass (frozen=True)
class ConvergenceResult:
    timestep: float
    steps: int
    maximum_relative_energy_error: float
    final_relative_energy_error: float

@dataclass(frozen=True)
class BenchmarkResult:
    test_name: str
    mode_name: str
    integrator: str
    pair_order: str
    timestep: float
    runtime: float
    steps: int
    output_frequency: int
    elapsed_seconds: float
    status: str
    maximum_relative_energy_error: float = math.nan
    final_relative_energy_error: float = math.nan
    maximum_linear_momentum_error: float = math.nan
    maximum_angular_momentum_error: float = math.nan
    maximum_COM_error: float = math.nan
    failure_message: str = ""

# ============================================================
# Default Benchmark Tests
# ============================================================

DEFAULT_BENCHMARK_MODES = [
    {
        "name": "hernande_canonical",
        "integrator": "hernandez",
        "pair_order": "canonical"
    },
    {
        "name": "hernande_strnegth",
        "integrator": "hernandez",
        "pair_order": "strength"
    },
    {
        "name": "leapfrog",
        "integrator": "leapfrog",
        "pair_order": "canonical"
    },
]

BENCHMARK_TESTS = [
    {
        "name": "Test1_Binary",
        "dt": 0.1,
        "runtime": 258230.8,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0, -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,  0.5, 0.0, 0.0, 0.0,  0.012166, 0.0)],
        "extreme_duration": False
    },
    {
        "name": "Test2_CircumbinaryTriple",
        "dt": 0.001,
        "runtime": 258230.82,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (1e-3,  5.0, 0.0, 0.0, 0.0,  0.01089,  0.0)],
        "extreme_duration": False
    },
    {
        "name": "Test3_StrongerPerturbedTriple",
        "dt": 0.001,
        "runtime": 258230.82,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (0.05,  4.0, 0.0, 0.0, 0.0,  0.01220,  0.0)],
        "extreme_duration": False
    },
    {
        "name": "Test4_ScatteringEscape",
        "dt": 0.02,
        "runtime": 129115.42,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0, -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,  0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (0.1,  3.0, 0.0, 0.0, 0.0,  0.0040,   0.0)],
        "extreme_duration": False
    },
    {
        "name": "Test5_Figure8",
        "dt": 0.02,
        "runtime": 367677.02,
        "output_frequency": 10000,
        "initial_conditions": [
            (1.0, -0.97000436,  0.24308753, 0.0,  0.008019029,  0.007436436, 0.0),
            (1.0,  0.97000436, -0.24308753, 0.0,  0.008019029,  0.007436436, 0.0),
            (1.0,  0.0,         0.0,        0.0, -0.016038058, -0.014872872, 0.0)],
        "extreme_duration": False
    },
    {
        "name": "Test6_CloseEncounter",
        "dt": 0.002,
        "runtime": 10329.232,
        "output_frequency": 10000,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0,  0.0,    -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0,  0.0,     0.012166, 0.0),
            (0.01,  1.2, 0.2, 0.0, -0.0020,  0.0040,   0.0)],
        "extreme_duration": False
    },
    {
        "name": "Test7_SolarSystem",
        "dt": 1.0,
        "runtime": 1668096750000.0,
        "output_frequency": 100000,
        "initial_conditions": [
            (1.0,        0.0,            0.0,            0.0,  0.0,            0.0,            0.0),
            (1.6601e-7,  0.3637531341,   0.1323953134,   0.0, -0.0094579726,  0.0259855662,  0.0),
            (2.4478e-6,  0.1872120975,   0.6986850598,   0.0, -0.0195403467,  0.0052358201,  0.0),
            (3.0035e-6, -0.6427876097,   0.7660444431,   0.0, -0.0131798787, -0.0110592314,  0.0),
            (3.2272e-7, -1.3195447212,  -0.7618395000,   0.0,  0.0069691551, -0.0120709307,  0.0),
            (9.5458e-4,  2.6022000000,  -4.5071426115,   0.0,  0.0065344536,  0.0037726685,  0.0),
            (2.8588e-4,  7.3406974806,   6.1595765486,   0.0, -0.0035730960,  0.0042582500,  0.0),
            (4.3662e-5, -18.0593886633,  6.5730799225,   0.0, -0.0013423302, -0.0036880218,  0.0),
            (5.1514e-5, -5.2285466296, -29.6525614432,   0.0,  0.0030879059, -0.0005444811,  0.0)],
        "extreme_duration": True
    },
    {
        "name": "Test8_SolarSystemInnerPlanets",
        "dt": 1.0,
        "runtime": 1668096750000.0,
        "output_frequency": 100000,
        "initial_conditions": [
            (1.0,        0.0,            0.0,           0.0,  0.0,            0.0,            0.0),
            (1.6601e-7,  0.3637531341,   0.1323953134,  0.0, -0.0094579726,  0.0259855662,  0.0),
            (2.4478e-6,  0.1872120975,   0.6986850598,  0.0, -0.0195403467,  0.0052358201,  0.0),
            (3.0035e-6, -0.6427876097,   0.7660444431,  0.0, -0.0131798787, -0.0110592314,  0.0),
            (3.2272e-7, -1.3195447212,  -0.7618395000,  0.0,  0.0069691551, -0.0120709307,  0.0)],
        "extreme_duration": True
    },
    {
        "name": "Test9_HernandezTest1",
        "dt": 0.0008718353300301555,
        "runtime": 81.3710068577045,
        "output_frequency": 20,
        "initial_conditions": [
            (0.5, -0.0595, 0.0, 0.0, 0.0, -0.0582073219045089, 0.0),
            (0.5, -0.0595, 0.0, 0.0, 0.0, -0.0187361524126806, 0.0),
            (1.0,  0.05,   0.0, 0.0, 0.0,  0.0384717371585948, 0.0)],
        "extreme_duration": False
    },
    {
        "name": "Test10_Chaotic",
        "dt": 0.002906117766767185,
        "runtime": 287.7056589099513,
        "output_frequency": 20,
        "initial_conditions": [
            (0.1,  0.4321919867168497,   0.93749651675052,   -0.3041222100911036, -0.005394528430293234, -0.1697804727188034,   -0.006025377253689681),
            (0.1,  0.4331919867168497,   0.93749651675052,   -0.3041222100911036, -0.005394528430293234,  0.1656083561947428,   -0.006025377253689681),
            (0.2, -0.2985300967322199,  -0.4919293319164365, -0.5557745169988454, -0.001859350004705104,  0.003411727705707647,  0.001978413022720391),
            (0.2, -0.5826179710240658,   0.2756894938605597,  0.3036806597227503, -0.001652788771678327,  0.0003375649551102012, 0.005562409700927202),
            (0.2, -0.01385949468766773, -0.1589178591348066,  0.1121512835701479,  0.01388290624726882,  -0.009590567695588893, -0.005512735194550181),
            (0.2,  0.4623155757271038,  -0.5623388195598369,  0.4440647837970507, -0.00497623904059215,   0.007927333296801377,  0.00399728972459227)],
        "extreme_duration": False
    },
]

# ============================================================
# Reading data
# ============================================================

def read_params(path: Path = DEFAULT_PARAM_PATH) -> dict[str, str]:
    """Read a strict two-column parameter file."""
    params: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected exactly one key and one value")
            key, value = fields
            if key in params:
                raise ValueError(f"{path}:{line_number}: duplicate parameter {key!r}")
            params[key] = value
    return params

def write_params_file(params: dict[str, object], path: Path = DEFAULT_PARAM_PATH) -> None:
    """Write parameters supported by the fixed-step Cartesian solver."""
    supported_order = ("output_frequency", 
                       "runtime",
                       "timestep",
                       "gravitational_constant",
                       "integrator",
                       "pair_order")
    unknown = set(params) - set(supported_order)
    if unknown:
        raise ValueError(f"unsupported parameter key: {sorted(unknown)}")

    missing = set(supported_order) - set(params)
    if missing:
        raise ValueError(f"missing parameter key: {sorted(missing)}")

    lines = ["# Fixed-step Cartesian Integration."]
    lines.extend(f"{key} {params[key]}" for key in supported_order)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def read_initial_conditions(path: Path = DEFAULT_INITIAL_CONDITIONS_PATH) -> list[tuple[float, float, float, float, float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find initial-condition file: {path}")
    
    rows = []

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()

        if len(parts) not in (7, 8):
            raise ValueError(f"Initial-conditions line {line_number} must have 7 or 8 values: mass x y z vx vy vz [radius]")
        
        rows.append(tuple(float(value) for value in parts))

    if not rows:
        raise ValueError(f"No bodies were found in {path}")
    return rows

def read_output(path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    if "radius" not in dataframe.columns:
        dataframe["radius"] = 0.0
    required = {"time", "id", "x", "y", "z", "vx", "vy", "vz", "mass", "radius"}

    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns in output: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError(f"{path} contains no body states")

    numeric_columns = sorted(required - {"id"})
    if not np.isfinite(dataframe[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite numeric values")
    if (dataframe["mass"] <= 0.0).any() or (dataframe["radius"] < 0.0).any():
        raise ValueError(f"{path} contains an invalid mass or radius")

    return dataframe.sort_values(["time", "id"], kind="stable").reset_index(drop=True)

def read_diagnostics(path: Path = DEFAULT_DIAGNOSTICS_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    required = {"time", 
                "total_energy", 
                "kinetic_energy", 
                "potential_energy", 
                "dE_over_E0", 
                "linear_momentum_x",
                "linear_momentum_y",
                "linear_momentum_z",
                "angular_momentum_x",
                "angular_momentum_y",
                "angular_momentum_z",
                "COM_invariant"}

    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError(f"{path} contains no diagnostic rows")
    
    return dataframe.sort_values("time", kind="stable").reset_index(drop=True)

# ============================================================
# Helper Functions
# ============================================================

def _normalize_initial_condition_record(record: Sequence[object], body_index: int) -> tuple[float, np.ndarray, np.ndarray, float]:
    if len(record) in {7, 8}:
        values = np.asarray(record, dtype=float)
        mass = float(values[0])
        position = values[1:3]
        velocity = values[4:6]
        radius = float(values[7]) if len(record) == 8 else 0.0
    elif len(record) in {3, 4}:
        mass = float(record[0])
        position = np.asarray(record[1], dtype=float)
        velocity = np.asarray(record[2], dtype=float)
        radius = float(record[3]) if len(record) == 4 else 0.0
    else:
        raise ValueError(f"body {body_index} must be either (mass, position, velocity, [radius]) or mass x y z vx vy vz [radius]")

    if mass <= 0.0 or not math.isfinite(mass):
        raise ValueError(f"body {body_index} has an invalid mass")
    if radius < 0.0 or not math.isfinite(radius):
        raise ValueError(f"body {body_index} has an invalid radius")
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError(f"body {body_index} position and velocity must each have three components")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError(f"body {body_index} has non-finite coordinates")

    return mass, position, velocity, radius

def _aligned_step_count(runtime: float, timestep: float) -> int:
    if not math.isfinite(runtime) or runtime < 0.0:
        raise ValueError("runtime must be finite and non-negative")
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("timestep must be finite and positive")

# ============================================================
# Compute Functions
# ============================================================

def compute_diagnostics_from_output(output: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    """Recompute Newtonian invariants from physical Cartesian snapshots."""
    records: list[dict[str, float]] = []

    for time, snapshot in output.groupby("time", sort=True):
        snapshot = snapshot.sort_values("id", kind="stable")
        masses = snapshot["mass"].to_numpy(dtype=float)
        radii = snapshot["radius"].to_numpy(dtype=float)
        positions = snapshot[["x", "y", "z"]].to_numpy(dtype=float)
        velocities = snapshot[["vx", "vy", "vz"]].to_numpy(dtype=float)
        total_mass = float(masses.sum())
        kinetic_energy = float(0.5 * np.sum(masses * np.sum(velocities * velocities, axis=1)))
        potential_energy = 0.0
        for first in range(len(snapshot)):
            for second in range(first + 1, len(snapshot)):
                separation = float(np.linalg.norm(positions[first] - positions[second]))
                minimum_separation = max(radii[first] + radii[second], config.singularity_tolerance)
                if not math.isfinite(separation) or separation <= minimum_separation:
                    raise ValueError(f"overlapping or singular pair at time {time}: ids {snapshot.iloc[first]["id"]} and {snapshot.iloc[second]["id"]}")
                potential_energy -= (config.gravitational_constant * masses[first] * masses[second] / separation)

        momenta = masses[:, None] * velocities
        linear_momentum = momenta.sum(axis=0)
        angular_momentum = np.cross(positions, momenta).sum(axis=0)
        COM_position = (masses[:, None] * positions).sum(axis=0) / total_mass
        COM_velocities = (masses[:, None] * velocities).sum(axis=0) / total_mass

        records.append({
            "time": float(time),
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "total_energy": kinetic_energy + potential_energy
            "linear_momentum_x": float(linear_momentum[0]),
            "linear_momentum_y": float(linear_momentum[1]),
            "linear_momentum_z": float(linear_momentum[2]),
            "angular_momentum_x": float(angular_momentum[0]),
            "angular_momentum_y": float(angular_momentum[1]),
            "angular_momentum_z": float(angular_momentum[2]),
            "com_x": float(center_of_mass_position[0]),
            "com_y": float(center_of_mass_position[1]),
            "com_z": float(center_of_mass_position[2]),
            "com_vx": float(center_of_mass_velocity[0]),
            "com_vy": float(center_of_mass_velocity[1]),
            "com_vz": float(center_of_mass_velocity[2])
        })

    diagnostics = pd.DataFrame.from_records(records)
    energy = diagnostics["total_energy"].to_numpy(dtype=float)
    diagnostics["dE_over_E0"] = _relative_error(energy)
    diagnostics["dE_abs"] = np.abs(energy - energy[0])
    
    initial_time = float(diagnostics.loc[0, "time"])
    elapsed_time = diagnostics["time"].to_numpy(dtype=float) - initial_time
    initial_position = diagnostics.loc[0, ["com_x", "com_y", "com_z"]].to_numpy(dtype=float)
    initial_velocity = diagnostics.loc[0, ["com_vx", "com_vy", "com_vz"]].to_numpy(dtype=float)
    positions = diagnostics[["com_vx", "com_vy", "com_vz"]].to_numpy(dtype=float)
    com_invariant = positions - initial_position - elapsed_time[:, None] * initial_velocity
    diagnostics[["com_invariant_x", "com_invariant_y", "com_invariant_z"]] = com_invariant
    diagnostics["com_invariant"] = np.linalg.norm(com_invariant, axis=1)

    initial_linear = diagnostics.loc[0, ["linear_momentum_x", "linear_momentum_y", "linear_momentum_z"]].to_numpy(dtype=float)
    linear = diagnostics[["linear_momentum_x", "linear_momentum_y", "linear_momentum_z"]].to_numpy(dtype=float)
    diagnostics["dP"] = np.linalg.norm(linear - initial_linear, axis=1)

    initial_angular = diagnostics.loc[0, ["angular_momentum_x", "angular_momentum_y", "angular_momentum_z"]].to_numpy(dtype=float)
    angular = diagnostics[["angular_momentum_x", "angular_momentum_y", "angular_momentum_z"]].to_numpy(dtype=float)
    diagnostics["dL"] = np.linalg.norm(angular - initial_angular, axis=1)

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

def _relative_error(values: np.ndarray) -> np.ndarray:
    reference = float(values[0])
    denominator = max(abs(reference), np.finfo(float).tiny)
    return np.abs(values - reference) / denominator

def error_rms_position(test_positions: dict[int, np.ndarray], reference_positions: dict[int, np.ndarray]) -> float:
    total = 0.0
    count = 0

    for body_id, r_ref in reference_positions.items():
        r = test_positions[body_id]
        dr = r - r_ref

        total += np.dot(dr, dr)
        count += 1

    return np.sqrt(total / count)

def error_signed_relative_energy(energy: np.ndarray, epsilon: float = 1e-300) -> np.ndarray:
    energy = np.asarray(energy, dtype=float)
    safe_epsilon = max(float(epsilon), 1e-14)
    finite = np.isfinite(energy)
    result = np.full_like(energy, np.nan, dtype=float)

    if not np.any(finite):
        return result
    
    reference = energy[0]

    if not np.isfinite(reference):
        reference = energy[finite][0]

    scale = max(abs(reference), safe_epsilon)
    result[finite] = (energy[finite] - reference) / scale

    return result

def error_safe_log_values(values: np.ndarray, floor: float = 1e-300, ceiling: float = 1e50) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    
    safe = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)

    safe[finite] = np.abs(values[finite])
    safe[finite] = np.clip(safe[finite], floor, ceiling)

    return safe

def error_print_summary(diagnostics: pd.DataFrame, config: PlotConfig) -> None:
    diagnostics = add_diagnostics_error_columns(diagnostics, config)

    dE = diagnostics["dE_over_E0"].to_numpy(dtype=float)
    dLz = diagnostics["dLz"].to_numpy(dtype=float)
    dPx = diagnostics["dPx"].to_numpy(dtype=float)
    dPy = diagnostics["dPy"].to_numpy(dtype=float)
    dXcm = diagnostics["dRcm_x"].to_numpy(dtype=float)
    dYcm = diagnostics["dRcm_y"].to_numpy(dtype=float)

    nine = diagnostics["nine_integral_of_motion_error_max"].to_numpy(dtype=float)

    print("Summary of Diagnostics:")
    print(f"Max |dE/E0|:", compute_finite_max(dE))
    print(f"Max |dLz|:", compute_finite_max(dLz))
    print(f"Max |dPcm_x|:", compute_finite_max(dPx))
    print(f"Max |dPcm_y|:", compute_finite_max(dPy))
    print(f"Max |dXcm|:", compute_finite_max(dXcm))
    print(f"Max |dYcm|:", compute_finite_max(dYcm))
    print(f"Max nine-integral-of-motion error:", compute_finite_max(nine))
    print(f"Final dE/E0:", float(dE[-1]) if len(dE) else np.nan)
    print(f"Final dLz:", float(dLz[-1]) if len(dLz) else np.nan)
    print(f"Final dPcm_x:", float(dPx[-1]) if len(dPx) else np.nan)
    print(f"Final dPcm_y:", float(dPy[-1]) if len(dPy) else np.nan)
    print(f"Final dXcm:", float(dXcm[-1]) if len(dXcm) else np.nan)
    print(f"Final dYcm:", float(dYcm[-1]) if len(dYcm) else np.nan)

def error_diagnostic_metric_summary(diagnostics: pd.DataFrame, config: PlotConfig) -> dict:
    diagnostics = add_diagnostics_error_columns(diagnostics, config)

    dE = diagnostics["dE_over_E0"].to_numpy(dtype=float)
    dLz = diagnostics["dLz"].to_numpy(dtype=float)
    dP = np.nanmax(diagnostics[["dPx", "dPy", "dPz"]].to_numpy(dtype=float), axis=1)
    dRcm = np.nanmax(diagnostics[["dRcm_x", "dRcm_y", "dRcm_z"]].to_numpy(dtype=float), axis=1)
    nine = diagnostics["nine_integral_of_motion_error_max"].to_numpy(dtype=float)

    summary = {"max_dE_over_E0": compute_finite_max(dE),
               "final_dE_over_E0": float(dE[-1]) if len(dE) else np.nan,
               "max_dL_over_L0": compute_finite_max(dLz),
               "final_dL_over_L0": float(dLz[-1]) if len(dLz) else np.nan,
               "max_dP": compute_finite_max(dP),
               "final_dP": float(dP[-1]) if len(dP) else np.nan,
               "max_dRcm": compute_finite_max(dRcm),
               "final_dRcm": float(dRcm[-1]) if len(dRcm) else np.nan,
               "max_dLx": compute_finite_max(diagnostics["dLx"].to_numpy(dtype=float)),
               "max_dLy": compute_finite_max(diagnostics["dLy"].to_numpy(dtype=float)),
               "max_dLz": compute_finite_max(diagnostics["dLz"].to_numpy(dtype=float)),
               "max_dPx": compute_finite_max(diagnostics["dPx"].to_numpy(dtype=float)),
               "max_dPy": compute_finite_max(diagnostics["dPy"].to_numpy(dtype=float)),
               "max_dPz": compute_finite_max(diagnostics["dPz"].to_numpy(dtype=float)),
               "max_dCcm_x": compute_finite_max(diagnostics["dCcm_x"].to_numpy(dtype=float)),
               "max_dCcm_y": compute_finite_max(diagnostics["dCcm_y"].to_numpy(dtype=float)),
               "max_dCcm_z": compute_finite_max(diagnostics["dCcm_z"].to_numpy(dtype=float)),
               "overall_max_nine_integral_of_motion_error": compute_finite_max(nine),
               "final_nine_integral_of_motion_error": float(nine[-1]) if len(nine) else np.nan}
               
    summary.update(energy_boundedness_summary(diagnostics, config))

    return summary

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

        x_plot = x[valid]
        y_plot = y[valid]

        ax.plot(x_plot, y_plot, linewidth=1, label=f'Body {body_id}')
        first_valid = np.flatnonzero(valid)[0]
        ax.scatter(x[first_valid], y[first_valid], s=config.start_marker_size, zorder=5)

    ax.set_title('Orbits of Bodies')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

def plot_error(ax, time, values, title, ylabel, floor=1e-300, ymin_fixed=None, max_points: int = 5000) -> None:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    time, values = thin_for_plotting(time, values, max_points=max_points)
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
        
        if ymin_fixed is not None and np.isfinite(ymin_fixed) and ymin_fixed > 0.0:
            ymin = ymin_fixed
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

def plot_verification_suite(output: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    """Display orbit and conservation plots."""
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    for body_id, body_history in output.groupby("id", sort=True):
        axis.plot(body_history["x"], body_history["y"], body_history["z"], label=f"Body {body_id}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title("Cartesian Trejectories")
    axis.legend()
    figure.tight_layout()

    plt.figure()
    plt.semilogy(diagnostics["time"], np.maximum(diagnostics["dE_over_E0"], np.finfo(float).tiny))
    plt.xlabel("Time")
    plt.ylabel(r"$|E-E_0|/|E_0|$")
    plt.title("Relative Energy Error")
    plt.tight_layout()

    plt.figure()
    momentum_error = diagnostics.get("dP")
    if momentum_error is None:
        components = diagnostics[["linear_momentum_x", "linear_momentum_y", "linear_momentum_z"]].to_numpy(dtype=float)
        momentum_error = np.linalg.norm(components - components[0], axis=1)
    plt.semilogy(diagnostics["time"], np.maximum(momentum_error, np.finfo(float).tiny))
    plt.xlabel("Time")
    plt.ylabel(r"$|\mathbf{P}-\mathbf{P}_0|$")
    plt.title("Linear-momentum conservation")
    plt.tight_layout()

    plt.figure()
    angular_error = diagnostics.get("dP")
    if angular_error is None:
        components = diagnostics[["angular_momentum_x", "angular_momentum_y", "angular_momentum_z"]].to_numpy(dtype=float)
        angular_error = np.linalg.norm(components - components[0], axis=1)
    plt.semilogy(diagnostics["time"], np.maximum(angular_error, np.finfo(float).tiny))
    plt.xlabel("Time")
    plt.ylabel(r"$|\mathbf{:}-\mathbf{L}_0|$")
    plt.title("Angular-momentum conservation")
    plt.tight_layout()

    plt.figure()
    plt.semilogy(diagnostics["time"], np.maximum(diagnostics["com_invariant"], np.finfo(float).tiny))
    plt.xlabel("Time")
    plt.ylabel(r"$|\mathbf{R}_{cm}(t)-\mathbf{R}_{cm}(0)-t\mathbf{V}_{cm}(0)|$")
    plt.title("COM Galilean invariant")
    plt.tight_layout()
    plt.show()

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

"""
Write initial conditions in the exact format expected by the C++ reader:
    mass x y z vx vy vz
No header row is written
"""
def write_initial_conditions(bodies: Sequence[Sequence[object]], path: Path = DEFAULT_INITIAL_CONDITIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, record in enumerate(bodies):
        mass, position, velocity, radius = _normalize_initial_condition_record(record, index)
        values = [mass, *position.tolist(), *velocity.tolist(), radius]
        lines.append(" ".join(f"{value:.17g}" for value in values))
    if not lines:
        raise ValueError("at least one body is required")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ============================================================
# Run executable and other tests
# ============================================================

def run_simulation(executable: Path = DEFAULT_EXECUTABLE_PATH, *, timeout_seconds: float | None = None) -> subprocess.CompletedProcess[str]:
    if not executable.exists():
        raise FileNotFoundError(f"solver executable not found: {executable}")
    completed = subprocess.run([str(executable)], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False, timeout=timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(f"solver exited with code {completed.returncode}\n"
                           f"stdout:\n{completed.stdout}\n"
                           f"stderr:\n{completed.stderr}\n")
    return completed

"""
Run a convergence study using the current settings in data/param.txt
Only the timestep is changed during the sweep.
The original param.txt is restored afterward.
"""
def run_convergence_case(case_name: str, 
                         initial_conditions: list[tuple[float, float, float, float, float, float, float]], 
                         dt_ref: float, 
                         dts: tuple, 
                         runtime: float, 
                         output_frequency: int, 
                         integrator: str,
                         coordinate_mode: str,
                         pair_order: str,
                         adaptive_timesteps: bool,
                         G: float,
                         use_diagnostics_csv: bool = True) -> pd.DataFrame:
    config = PlotConfig(G=G)
    rows = []

    print("\n" + "-" * 90)
    print(f"Convergence case: {case_name}")
    print(f"integrator           = {integrator}")
    print(f"coordinate_mode      = {coordinate_mode}")
    print(f"pair_order           = {pair_order}")
    print(f"adaptive_timesteps   = {adaptive_timesteps}")
    print(f"runtime              = {runtime}")
    print(f"output_frequency     = {output_frequency}")
    print(f"G                    = {G}")
    print(f"reference dt         = {dt_ref}")
    print(f"dt ladder            = {dts}")
    print("-" * 90)

    rewrite_initial_conditions(initial_conditions)

    print(f"\nRunning reference solutions: dt = {dt_ref}")
    rewrite_param(dt=dt_ref, runtime=runtime, output_frequency=output_frequency, integrator=integrator, coordinate_mode=coordinate_mode, G=G, pair_order=pair_order, adaptive_timesteps=adaptive_timesteps)
    clear_simulation_outputs()
    run_executable()
    reference_output = read_output(DEFAULT_OUTPUT_PATH)
    reference_positions = compute_final_positions(reference_output)

    for dt in dts:
        print(f"\nRunning {case_name}: dt = {dt}")
        rewrite_param(dt=dt, runtime=runtime, output_frequency=output_frequency, integrator=integrator, coordinate_mode=coordinate_mode, G=G, pair_order=pair_order, adaptive_timesteps=adaptive_timesteps)
        param_snapshot = DEFAULT_PARAM_PATH.read_text() if DEFAULT_PARAM_PATH.exists() else ""
        clear_simulation_outputs()
        try:
            run_executable()
            output = read_output(DEFAULT_OUTPUT_PATH)
            if use_diagnostics_csv:
                diagnostics = read_diagnostics(DEFAULT_DIAGNOSTICS_PATH)
                if diagnostics is None:
                    print("Falling back to recomputing diagnostics from output.csv")
                    diagnostics = compute_diagnostics_from_output(output, config)
            else:
                diagnostics = compute_diagnostics_from_output(output, config)
            final_positions = compute_final_positions(output)
            rms_error = error_rms_position(final_positions, reference_positions)
            diagnostics_summary = error_diagnostic_metric_summary(diagnostics, config)
            row = {
                "case": case_name,
                "dt": dt,
                "dt_ref": dt_ref,
                "runtime": runtime,
                "output_frequency": output_frequency,
                "integrator": integrator,
                "coordinate_mode": coordinate_mode,
                "pair_order": pair_order,
                "adaptive_timesteps": adaptive_timesteps,
                "G": G,
                "rms_final_position_error": rms_error,
                "error_ratio_to_next_finer": np.nan,
                "observed_order_to_next_finer": np.nan,
                "status": "success",
                "error": "",
                "param_snapshot": param_snapshot,
            }
            row.update(diagnostics_summary)
        except Exception as exc:
            row = {
                "case": case_name,
                "dt": dt,
                "dt_ref": dt_ref,
                "runtime": runtime,
                "output_frequency": output_frequency,
                "integrator": integrator,
                "coordinate_mode": coordinate_mode,
                "pair_order": pair_order,
                "adaptive_timesteps": adaptive_timesteps,
                "G": G,
                "rms_final_position_error": np.nan,
                "error_ratio_to_next_finer": np.nan,
                "observed_order_to_next_finer": np.nan,
                "max_dE_over_E0": np.nan,
                "final_dE_over_E0": np.nan,
                "max_dL_over_L0": np.nan,
                "final_dL_over_L0": np.nan,
                "max_dP": np.nan,
                "final_dP": np.nan,
                "max_dRcm": np.nan,
                "final_dRcm": np.nan,
                "status": "failed",
                "error": str(exc),
                "param_snapshot": param_snapshot if "param_snapshot" in locals() else "",
            }
            print(f"Convergence run failed: {exc}")
        
        rows.append(row)
    rows = sorted(rows, key=lambda item: item["dt"], reverse=True)

    for i in range(len(rows) - 1):
        coarse = rows[i]
        fine = rows[i + 1]
        coarse_error = coarse["rms_final_position_error"]
        fine_error = fine["rms_final_position_error"]

        if (coarse["status"] == "success" and fine["status"] == "success" and np.isfinite(coarse_error) and np.isfinite(fine_error) and fine_error > 0.0):
            coarse["error_ratio_to_next_finer"] = coarse_error / fine_error
            coarse["observed_order_to_next_finer"] = finite_log2_ratio(coarse_error, fine_error)
        
    print("\nConvergence ratios:")
    for row in rows:
        print(f"   dt = {row['dt']:.8g}, "
                f"error = {row['rms_final_position_error']:.6e}, "
                f"ratio = {row['error_ratio_to_next_finer']:.6e}, "
                f"order = {row['observed_order_to_next_finer']:.6e}")
        
    return pd.DataFrame(rows)

def run_timestep_convergence_study(timestep_factors: Iterable[float] = (2.0, 1.0, 0.5, 0.25, 0.125),  
                                   param_path: Path = DEFAULT_PARAM_PATH,
                                   diagnostics_path: Path = DEFAULT_DIAGNOSTICS_PATH) -> list[ConvergenceResult]:
    original_text = param_path.read_text(encoding="utf-8")
    original_diagnostics = diagnostics_path.read_bytes() if diagnostics_path.exists() else None
    params = read_params(param_path)
    base_timestep = float(params["timestep"])
    runtime = float(params["runtime"])
    results: list[ConvergenceResult] = []

    try:
        for factor in timestep_factors:
            timestep = base_timestep * float(factor)
            exact_steps = runtime / timestep
            steps = int(round(exact_steps))
            if not math.isclose(exact_steps, steps, rel_tol=1.0e-10, abs_tol=1.0e-10):
                raise ValueError(f"runtime {runtime} is not aligned with timestep {timestep}")
            updated = {"output_frequency": max(1, steps // 2000),
                       "runtime": runtime,
                       "timestep": timestep,
                       "gravitational_constant": float(params["gravitational_constant"]),
                       "integrator": params.get("integrator", "hernandez"),
                       "pair_order": params.get("pair_order", "canonical")}
            write_params_file(updated, param_path)
            run_simulation()
            diagnostics = read_diagnostics(diagnostics_path)
            errors = diagnostics["dE_over_E0"].to_numpy(dtype=float)
            results.append(ConvergenceResult(timestep=timestep, 
                                             steps=steps, 
                                             maximum_relative_energy_error=float(np.max(errors)), 
                                             final_relative_energy_error=float(errors[-1])))
    finally:
        param_path.write_text(original_text, encoding="utf-8")
        if original_diagnostics is None:
            diagnostics_path.unlink(missing_ok=True)
        else:
            diagnostics_path.write_bytes(original_diagnostics)

    results.sort(key=lambda result: result.timestep)
    timesteps = np.array([result.timestep for result in results])
    errors = np.array([result.maximum_relative_energy_error for result in results])

    plt.figure()
    plt.loglog(timesteps, np.maximum(errors, np.finfo(float).tiny), marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("Maximum relative energy error")
    integrator_name = params.get("integrator", "hernandez")
    plt.title(f"Fixed-step {integrator_name.capitalize()} convergence")
    plt.tight_layout()
    plt.show()
    return results

def run_convergence_suite(tests: list[dict] | None = None, output_dir: Path = DEFAULT_BENCHMARK_PLOT_DIR / "convergence", use_diagnostics_csv: bool = True) -> pd.DataFrame:
    if tests is None:
        tests = CONVERGENCE_TESTS
    
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    readable_dir = output_dir / "readable"
    readable_dir.mkdir(parents=True, exist_ok=True)

    original_param_text = DEFAULT_PARAM_PATH.read_text() if DEFAULT_PARAM_PATH.exists() else None
    original_initial_conditions_text = DEFAULT_INITIAL_CONDITIONS_PATH.read_text() if DEFAULT_INITIAL_CONDITIONS_PATH.exists() else None

    all_results = []

    convergence_summary_readable_columns = [
        "case",
        "successful_runs",
        "median_observed_order",
        "min_observed_order",
        "max_observed_order",
        "max_rms_final_position_error",
        "max_dE_over_E0",        
    ]

    try:
        print("\n" + "=" * 90)
        print("ROADMAP 2 STEP D: CARTESIAN HERNANDEZ CONVERGENCE SUITE")
        print("=" * 90)
        print("This reuses the existing timestep scaling engine.")
        print("Mode:")
        print("  coordinate_mode      = cartesian")
        print("  integrator           = hernandez")
        print("  pair_order           = canonical")
        print("  adaptive_timesteps   = false")
        print("  G                    = 0.000296014912")
        for test in tests:
            result = run_convergence_case(
                case_name=test["name"],
                initial_conditions=test["initial_conditions"],
                dt_ref=test["dt_ref"],
                dts=test["dts"],
                runtime=test["runtime"],
                output_frequency=test["output_frequency"],
                integrator="hernandez",
                coordinate_mode="cartesian",
                pair_order="canonical",
                adaptive_timesteps=False,
                G=0.000296014912,
                use_diagnostics_csv=use_diagnostics_csv)
            
            all_results.append(result)
        
        results = pd.concat(all_results, ignore_index=True)

        raw_path = raw_dir / "convergence.csv"
        save_raw_table(results, raw_path)

        readable_path = readable_dir / "convergence_readable.csv"
        save_readable_table(readable, readable_path)

        successful = results[results["status"] == "success"].copy()

        if not successful.empty:
            summary = (successful.groupby("case", as_index=False).agg(
                median_observed_order=("observed_order_to_next_finer", "median"),
                min_observed_order=("observed_order_to_next_finer", "min"),
                max_observed_order=("observed_order_to_next_finer", "max"),
                max_rms_final_position_error=("rms_final_position_error", "max"),
                max_dE_over_E0=("max_dE_over_E0", "max"),
                successful_runs=("status", "count")))
            summary_path = raw_dir / "convergence_summary.csv"
            save_raw_table(summary, summary_path)
            summary_readable = compact_benchmark_table(summary, convergence_summary_readable_columns)
            summary_readable_path = readable_dir / "convergence_summary_readable.csv"
            save_readable_table(summary_readable, summary_readable_path)
        
        print("\nConvergence Suite Complete.")
        print(f"Results saved under: {output_dir}")

        return results
    
    finally:
        if  original_param_text is not None:
            DEFAULT_PARAM_PATH.write_text(original_param_text)
        if original_initial_conditions_text is not None:
            DEFAULT_INITIAL_CONDITIONS_PATH.write_text(original_initial_conditions_text)
        
        print("\nRestored original param.txt and initial_conditions.txt settings.")

"""
# Run every benchmark test in every selected mode and save the resulting plots.
# This function temporarily overwrites data/initial_conditions.txt and data/param.txt.
# The original files should be restored at the end of the run.

"""
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
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    readable_dir = output_dir / "readable"
    readable_dir.mkdir(parents=True, exist_ok=True)

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
            rebound_dir = output_dir / f"rebound_reference_{safe_filename(rebound_integrator)}"
            rebound_dir.mkdir(parents=True, exist_ok=True)
            plot_path = rebound_dir / f"{safe_filename(test['name'])}.png"
            plot_verification_suite(output_df=output, diagnostics=diagnostics, config=config, save_path=plot_path, show=False)
            print(f"Saved REBOUND reference plot to {plot_path}")

        return reference

    try:
        print("\nRunning benchmark suite...")

        total_runs = len(BENCHMARK_TESTS) * len(modes)
        run_number = 0
        benchmark_rows = []

        for mode in modes:
            mode_dir = output_dir / safe_filename(mode["name"])
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
                
                param_snapshot = DEFAULT_PARAM_PATH.read_text() if DEFAULT_PARAM_PATH.exists() else ""
                initial_conditions_snapshot = (DEFAULT_INITIAL_CONDITIONS_PATH.read_text() if DEFAULT_INITIAL_CONDITIONS_PATH.exists() else "")

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
                    failure_details = parse_benchmark_failure_details(stdout_tail=stdout_tail, stderr_tail=stderr_tail, mode_name=mode["name"], test_name=test["name"])
                    failure_message = failure_details["failure_message"] or str(exc)
                    print(f"Benchmark run failed: {failure_message}")
                    append_benchmark_row(benchmark_rows=benchmark_rows,
                                         run_number=run_number,
                                         mode=mode,
                                         test=test,
                                         status="failed",
                                         error=failure_message,
                                         returncode=returncode,
                                         stdout_tail=stdout_tail,
                                         stderr_tail=stderr_tail,
                                         engine="fewbodync",
                                         failure_type=type(exc).__name__,
                                         failure_message=failure_message,
                                         rebound_comparison=rebound_comparison,
                                         config=config,
                                         param_snapshot=param_snapshot,
                                         initial_conditions_snapshot=initial_conditions_snapshot,
                                         failure_details=failure_details)
                    continue
                
                try:
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
                    
                    plot_path = mode_dir / f"{safe_filename(test['name'])}.png"
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
                                        rebound_comparison=rebound_comparison,
                                        config=config,
                                        param_snapshot=param_snapshot,
                                        initial_conditions_snapshot=initial_conditions_snapshot)
                except Exception as exc:
                    failure_message = f"Benchmark analysis failed after executable completed: {exc}"
                    print(failure_message)
                    append_benchmark_row(benchmark_rows=benchmark_rows,
                                         run_number=run_number,
                                         mode=mode,
                                         test=test,
                                         status="failed",
                                         error=failure_message,
                                         returncode=0,
                                         stdout_tail="",
                                         stderr_tail="",
                                         engine="fewbodync",
                                         failure_type=type(exc).__name__,
                                         failure_message=failure_message,
                                         rebound_comparison=rebound_comparison,
                                         config=config,
                                         param_snapshot=param_snapshot,
                                         initial_conditions_snapshot=initial_conditions_snapshot)
                    continue
        
        print("\nBenchmark suite complete.")
        print(f"Plots saved to {output_dir}")

        if not benchmark_rows:
            print("No benchmark results were collected")
            return
        
        # Benchmark Summary
        summary = pd.DataFrame(benchmark_rows)
        summary_path = raw_dir / "benchmark_summary.csv"
        save_raw_table(summary, summary_path)
        if rebound_reference_rows:
            rebound_reference_summary = pd.DataFrame(rebound_reference_rows)
            rebound_reference_path = raw_dir / f"rebound_reference_{rebound_integrator}_summary.csv"
            save_raw_table(rebound_reference_summary, rebound_reference_path)
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

        readable_comparison_columns = [
            "test",
            "mode",
            "rank_in_test",
            "integrator",
            "coordinate_mode",
            "pair_order",
            "adaptive_timesteps",
            "dt",
            "runtime",
            "status",
            "max_dE_over_E0",
            "final_dE_over_E0",
            "energy_drift_slope",
            "energy_drift_over_run",
            "energy_drift_fraction_of_max",
            "energy_rms_dE_over_E0",
            "energy_boundedness_class",
            "max_dL_over_L0",
            "max_dP",
            "max_dRcm",
            "rebound_status",
            "ratio_max_dE_over_E0_to_rebound",
            "plot_path",
        ]

        readable_best_columns = [
            "test",
            "mode",
            "integrator",
            "coordinate_mode",
            "pair_order",
            "adaptive_timesteps",
            "dt",
            "runtime",
            "max_dE_over_E0",
            "final_dE_over_E0",
            "energy_drift_slope",
            "energy_drift_over_run",
            "energy_drift_fraction_of_max",
            "energy_rms_dE_over_E0",
            "energy_boundedness_class",
            "max_dL_over_L0",
            "max_dP",
            "max_dRcm",
            "rebound_status",
            "ratio_max_dE_over_E0_to_rebound",
            "plot_path",
        ]

        readable_worst_columns = [
            "test",
            "mode",
            "integrator",
            "coordinate_mode",
            "pair_order",
            "adaptive_timesteps",
            "dt",
            "runtime",
            "max_dE_over_E0",
            "final_dE_over_E0",
            "energy_drift_slope",
            "energy_drift_over_run",
            "energy_drift_fraction_of_max",
            "energy_rms_dE_over_E0",
            "energy_boundedness_class",
            "max_dL_over_L0",
            "max_dP",
            "max_dRcm",
            "rebound_status",
            "ratio_max_dE_over_E0_to_rebound",
            "plot_path",
        ]

        readable_mode_ranking_columns = [
            "mode",
            "successful_runs",
            "median_max_dE_over_E0",
            "mean_max_dE_over_E0",
            "worst_max_dE_over_E0",
            "median_max_dL_over_L0",
            "median_max_dP",
            "median_max_dRcm",
            "median_abs_energy_drift_slope",
            "median_energy_drift_fraction_of_max",
            "bounded_like_runs",
            "drifting_like_runs",
        ]        

        # Failure Summary
        failed_summary = summary[summary["status"] == "failed"].copy()
        if not failed_summary.empty:
            failures_path = raw_dir / "benchmark_failures.csv"
            save_raw_table(failed_summary, failures_path)

        # Sucess Summary
        successful_summary = summary[summary["status"] == "success"].copy()
        if successful_summary.empty:
            print("No successful benchmark runs were collected.")
            return

        # Raw Comparison Table
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
            "failed_time",
            "failed_dt",
            "failed_pair_i",
            "failed_pair_j",
            "failed_distance",
            "failed_relative_speed",
            "failed_iterations",
            "failed_mode",
            "failed_test",
            "returncode",
            "G",
            "body_count",
            "max_dE_over_E0",
            "final_dE_over_E0",
            "energy_drift_slope",
            "energy_abs_drift_slope",
            "energy_drift_over_run",
            "energy_drift_fraction_of_max",
            "energy_rms_dE_over_E0",
            "energy_boundedness_class",
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
        comparison_table = summary[comparison_columns].copy()
        comparison_table = comparison_table.sort_values(["test", "status", "max_dE_over_E0", "max_dL_over_L0", "max_dP", "max_dRcm"], na_position="last")
        comparison_path = raw_dir / "benchmark_comparison_table.csv"
        save_raw_table(comparison_table, comparison_path)

        # Readable Comparison Table
        comparison_readable = add_test_rank(comparison_table, metric="max_dE_over_E0")
        comparison_readable = compact_benchmark_table(comparison_readable, readable_comparison_columns)
        comparison_readable_path = readable_dir / "benchmark_comparison_table_readable.csv"
        save_readable_table(comparison_readable, comparison_readable_path)

        # Raw Best Runs
        best_rows = (successful_summary.sort_values(["test", "max_dE_over_E0", "max_dL_over_L0", "max_dP", "max_dRcm"]).groupby("test", as_index=False).first())
        best_path = raw_dir / "best_by_energy.csv"
        save_raw_table(best_rows, best_path)

        # Readable Best Runs
        best_readable = compact_benchmark_table(best_rows, readable_best_columns)
        best_readable_path = readable_dir / "best_by_energy_readable.csv"
        save_readable_table(best_readable, best_readable_path)
        
        # Raw Worst Runs
        worst_rows = (successful_summary.sort_values("max_dE_over_E0", ascending=False).head(10))
        worst_path = raw_dir / "worst_by_energy.csv"
        save_raw_table(worst_rows, worst_path)

        # Readable Worst Runs
        worst_readable = compact_benchmark_table(worst_rows, readable_worst_columns)
        worst_readable_path = readable_dir / "worst_by_energy_readable.csv"
        save_readable_table(worst_readable, worst_readable_path)
        
        # Rebound Comparison
        if rebound_compare and "final_rms_position_error_vs_rebound" in successful_summary.columns:
            valid_vs_rebound = successful_summary[np.isfinite(successful_summary["final_rms_position_error_vs_rebound"])].copy()
            if not valid_vs_rebound.empty:
                best_vs_rebound = (valid_vs_rebound.sort_values([
                        "test",
                        "final_rms_position_error_vs_rebound",
                        "final_rms_velocity_error_vs_rebound",
                        "max_rel_energy_difference_vs_rebound",
                    ]).groupby("test", as_index=False).first())
                
                # Raw
                best_vs_rebound_path = raw_dir / f"best_vs_rebound_{safe_filename(rebound_integrator)}.csv"
                save_raw_table(best_vs_rebound, best_vs_rebound_path)

                # Readable
                best_vs_rebound_readable_columns = ["test",
                                                    "mode",
                                                    "integrator",
                                                    "coordinate_mode",
                                                    "pair_order",
                                                    "adaptive_timesteps",
                                                    "final_rms_position_error_vs_rebound",
                                                    "final_rms_velocity_error_vs_rebound",
                                                    "final_rel_energy_difference_vs_rebound",
                                                    "ratio_max_dE_over_E0_to_rebound"]
                best_vs_rebound_readable = compact_benchmark_table(best_vs_rebound, best_vs_rebound_readable_columns)
                best_vs_rebound_readable_path = readable_dir / f"best_vs_rebound_{safe_filename(rebound_integrator)}_readable.csv"
                save_readable_table(best_vs_rebound_readable, best_vs_rebound_readable_path)
            else:
                print("No finite REBOUND comparison rows were available.")

        # Raw Ranking by Median Max
        mode_rankings = (successful_summary.groupby("mode", as_index=False).agg(
            median_max_dE_over_E0 = ("max_dE_over_E0", "median"),
            mean_max_dE_over_E0 = ("max_dE_over_E0", "mean"),
            worst_max_dE_over_E0 = ("max_dE_over_E0", "max"),
            median_abs_energy_drift_slope = ("energy_abs_drift_slope", "median"),
            median_energy_drift_fraction_of_max = ("energy_drift_fraction_of_max", "median"),
            median_max_dL_over_L0 = ("max_dL_over_L0", "median"),
            median_max_dP = ("max_dP", "median"),
            median_max_dRcm = ("max_dRcm", "median"),
            bounded_like_runs = ("energy_boudedness_class", lambda values: int(values.astype(str).contains("bounded_like").sum())),
            drifting_like_runs = ("energy_boundedness_class", lambda values: int((values.astype(str) == "drifting_like").sum())),
            successful_runs = ("status", "count")).sort_values("median_max_dE_over_E0"))
        mode_rankings_path = raw_dir / "mode_rankings.csv"
        save_raw_table(mode_rankings, mode_rankings_path)

        # Readable Ranking by Median Max
        mode_rankings_readable = compact_benchmark_table(mode_rankings, readable_mode_ranking_columns)
        mode_rankings_readable_path = readable_dir / "mode_rankings_readable.csv"
        save_readable_table(mode_rankings_readable, mode_rankings_readable_path)

    finally:
        if original_param_text is not None:
            DEFAULT_PARAM_PATH.write_text(original_param_text)
        if original_initial_conditions_text is not None:
            DEFAULT_INITIAL_CONDITIONS_PATH.write_text(original_initial_conditions_text)
        print("\nRestored original param.txt and initial_conditions.txt settings.")

def run_pair_order_policy_suite(modes: list[dict] | None = None, tests: list[dict] | None = None, output_dir: Path = DEFAULT_BENCHMARK_PLOT_DIR / "pair_order_policy", use_diagnostics_csv: bool = True) -> pd.DataFrame:
    if modes is None:
        modes = PAIR_ORDER_POLICY_MODES
    if tests is None:
        tests = PAIR_ORDER_POLICY_TESTS

    config = PlotConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    readable_dir = output_dir / "readable"
    readable_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    original_param_text = DEFAULT_PARAM_PATH.read_text() if DEFAULT_PARAM_PATH.exists() else None
    original_initial_conditions_text = DEFAULT_INITIAL_CONDITIONS_PATH.read_text() if DEFAULT_INITIAL_CONDITIONS_PATH.exists() else None
    rows = []
    run_number = 0
    total_runs = len(modes) * len(tests)

    pair_order_policy_readable_columns = [
        "test",
        "mode",
        "pair_order",
        "pair_order_policy_status",
        "expected_effective_pair_order",
        "adaptive_timesteps",
        "dt",
        "runtime",
        "status",
        "max_dE_over_E0",
        "final_dE_over_E0",
        "max_dL_over_L0",
        "max_dP",
        "max_dRcm",
        "ratio_max_dE_over_E0_to_canonical",
        "ratio_max_dL_over_L0_to_canonical",
        "ratio_max_dP_to_canonical",
        "ratio_max_dRcm_to_canonical",
        "pair_order_recommendation",
        "plot_path",
        "error",
    ]

    try:
        print("\n" + "=" * 90)
        print("PAIR-ORDER POLICY SUITE")
        print("=" * 90)
        print("Policy:")
        print("  canonical = production default")
        print("  strength  = optional fixed-step diagnostic mode")
        print("  auto      = conservative; expected to resolve to canonical")
        print("  adaptive  = not included in this pair-order policy suite")

        for test in tests:
            for mode in modes:
                run_number += 1

                print("\n" + "-" * 90)
                print(f"Run {run_number}/{total_runs}")
                print(f"Test: {test['name']}")
                print(f"Mode: {mode['name']}")
                print(f"Pair order: {mode['pair_order']}")
                print(f"Policy status: {mode['pair_order_policy_status']}")
                print("-" * 90)    

                rewrite_initial_conditions(test["initial_conditions"])
                rewrite_param(
                    dt=test["dt"],
                    runtime=test["runtime"],
                    output_frequency=test["output_frequency"],
                    integrator=mode["integrator"],
                    coordinate_mode=mode["coordinate_mode"],
                    G=config.G,
                    pair_order=mode.get("pair_order", "canonical"),
                    adaptive_timesteps=mode.get("adaptive_timesteps", False),
                    timestep_levels=mode.get("timestep_levels"),
                    timestep_eta=mode.get("timestep_eta"),
                    timestep_refresh_interval=mode.get("timestep_refresh_interval"),
                    timestep_level_decrease_delay=mode.get("timestep_level_decrease_delay"))
                param_snapshot = DEFAULT_PARAM_PATH.read_text() if DEFAULT_PARAM_PATH.exists() else ""
                initial_conditions_snapshot = DEFAULT_INITIAL_CONDITIONS_PATH.read_text() if DEFAULT_INITIAL_CONDITIONS_PATH.exists() else ""

                if DEFAULT_OUTPUT_PATH.exists():
                    DEFAULT_OUTPUT_PATH.unlink()
                if DEFAULT_DIAGNOSTICS_PATH.exists():
                    DEFAULT_DIAGNOSTICS_PATH.unlink()

                try:
                    run_executable()
                    output = read_output(DEFAULT_OUTPUT_PATH)
                    if use_diagnostics_csv:
                        diagnostics = read_diagnostics(DEFAULT_DIAGNOSTICS_PATH)
                        if diagnostics is None:
                            print("Falling back to recomputing diagnostics from output.csv.")
                            diagnostics = compute_diagnostics_from_output(output, config)
                    else:
                        diagnostics = compute_diagnostics_from_output(output, config)

                    case_plot_dir = plot_dir / safe_filename(test["name"])
                    case_plot_dir.mkdir(parents=True, exist_ok=True)
                    plot_path = case_plot_dir / f"{safe_filename(mode['name'])}.png"
                    plot_verification_suite(output_df=output, diagnostics=diagnostics, config=config, save_path=plot_path, show=False)
                    append_benchmark_row(benchmark_rows=rows, 
                                         run_number=run_number, 
                                         mode=mode, 
                                         test=test, 
                                         status="success", 
                                         plot_path=plot_path, 
                                         diagnostics=diagnostics, 
                                         engine="fewbodync", 
                                         config=config,
                                         param_snapshot=param_snapshot,
                                         initial_conditions_snapshot=initial_conditions_snapshot)
                    rows[-1]["pair_order_policy_status"] = mode["pair_order_policy_status"]
                    rows[-1]["expected_effective_pair_order"] = mode["expected_effective_pair_order"]

                    print(f"max |dE/E0|: {rows[-1]['max_dE_over_E0']:.6e}")
                    print(f"policy status: {rows[-1]['pair_order_policy_status']}")
                except Exception as exc:
                    failure_message = str(exc)
                    append_benchmark_row(benchmark_rows=rows, 
                                         run_number=run_number, 
                                         mode=mode, 
                                         test=test, 
                                         status="failed", 
                                         error=failure_message, 
                                         engine="fewbodync",
                                         failure_type=type(exc).__name__,
                                         failure_message=failure_message, 
                                         config=config,
                                         param_snapshot=param_snapshot,
                                         initial_conditions_snapshot=initial_conditions_snapshot)
                    rows[-1]["pair_order_policy_status"] = mode["pair_order_policy_status"]
                    rows[-1]["expected_effective_pair_order"] = mode["expected_effective_pair_order"]

                    print(f"Pair-order policy run failed: {failure_message}")
        results = pd.DataFrame(rows)
        
        # Compare each pair-order mode against canonical for the same test
        for metric in ["max_dE_over_E0", "max_dL_over_L0", "max_dP", "max_dRcm"]:
            ratio_column = f"ratio_{metric}_to_canonical"
            results[ratio_column] = np.nan
            if metric not in results.columns:
                continue
            results[metric] = pd.to_numeric(results[metric], errors="coerce")
            canonical_reference = results[(results["pair_order"] == "canonical") & (results["status"] == "success")][["test", metric]].drop_duplicates(subset=["test"]).set_index("test")[metric]
            canonical_values = results["test"].map(canonical_reference)
            valid = results[metric].notna() & canonical_values.notna() & (canonical_values.abs() > config.epsilon)
            results.loc[valid, ratio_column] = results.loc[valid, metric] / canonical_values.loc[valid]
        
        def recommendation(row: pd.Series) -> str:
            if row["status"] != "success":
                return "failed_do_not_use"
            if row["pair_order"] == "canonical":
                return "production_default"
            if row["pair_order"] == "auto":
                return "allowed_but_expected_to_resolve_to_canonical"
            if row["pair_order"] == "strength":
                ratio = row.get("ratio_max_dE_over_E0_to_canonical", np.nan)
                if np.isfinite(ratio) and ratio <= 1.1:
                    return "optional_fixed_step_only"
                return "demoted_not_better_than_canonical"
            return "unknown"
        
        results["pair_order_recommendation"] = results.apply(recommendation, axis=1)
        raw_path = raw_dir / "pair_order_policy_summary.csv"
        save_raw_table(results, raw_path)
        readable = compact_benchmark_table(results, pair_order_policy_readable_columns)
        readable_path = readable_dir / "pair_order_policy_summary_readable.csv"
        save_readable_table(readable, readable_path)
        
        successful = results[results["status"] == "success"].copy()
        if not successful.empty:
            counts = successful.groupby(["pair_order", "pair_order_policy_status", "pair_order_recommendation"], as_index=False).size().rename(columns={"size": "run_count"})
            counts_path = raw_dir / "pair_order_policy_counts.csv"
            save_raw_table(counts, counts_path)
            counts_readable_path = readable_dir / "pair_order_policy_counts_readable.csv"
            save_readable_table(counts, counts_readable_path)
        
        print("\nPair-Order Policy Suite Complete.")
        print(f"Results saved under: {output_dir}")

        return results
    
    finally:
        if original_param_text is not None:
            DEFAULT_PARAM_PATH.write_text(original_param_text)

        if original_initial_conditions_text is not None:
            DEFAULT_INITIAL_CONDITIONS_PATH.write_text(original_initial_conditions_text)

        print("\nRestored original param.txt and initial_conditions.txt settings.")

"""
# Run comparison for adaptive timestep vs. fixed time step for every integrator and save the resulting plots.
# This function temporarily overwrites data/initial_conditions.txt and data/param.txt.
# The original files should be restored at the end of the run.

"""
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

"""
# Run energy boundedness tests and save the resulting plots.
# This function temporarily overwrites data/initial_conditions.txt and data/param.txt.
# The original files should be restored at the end of the run.

"""
def run_energy_boundedness_suite(cases: list[dict] | None = None, output_dir: Path = DEFAULT_BENCHMARK_PLOT_DIR / "energy_boundedness", use_diagnostics_csv: bool = True) -> pd.DataFrame:
    if cases is None:
        cases = ENERGY_BOUNDEDNESS_CASES

    config = PlotConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    readable_dir = output_dir / "readable"
    readable_dir.mkdir(parents=True, exist_ok=True)
    original_param_text = DEFAULT_PARAM_PATH.read_text() if DEFAULT_PARAM_PATH.exists() else None
    original_initial_conditions_text = DEFAULT_INITIAL_CONDITIONS_PATH.read_text() if DEFAULT_INITIAL_CONDITIONS_PATH.exists() else None
    rows = []
    run_number = 0
    total_runs = sum(len(case["modes"]) for case in cases)

    energy_boundedness_readable_columns = [
        "case",
        "test",
        "mode",
        "integrator",
        "coordinate_mode",
        "pair_order",
        "adaptive_timesteps",
        "dt",
        "runtime",
        "max_dE_over_E0",
        "final_dE_over_E0",
        "energy_drift_slope",
        "energy_abs_drift_slope",
        "energy_drift_over_run",
        "energy_drift_fraction_of_max",
        "energy_rms_dE_over_E0",
        "energy_boundedness_class",
        "max_dL_over_L0",
        "max_dP",
        "max_dRcm",
        "status",
        "error",
    ]

    try:
        print("\n" + "=" * 90)
        print("Energy Boundedness and Drift Suite")
        print("=" * 90)

        for case in cases:
            test = case["test"]
            for mode in case["modes"]:
                run_number += 1
                
                print("\n" + "-" * 90)
                print(f"Run {run_number}/{total_runs}")
                print(f"Case: {case['name']}")
                print(f"Mode: {mode['name']}")
                print(f"Test: {test['name']}")
                print("-" * 90)

                rewrite_initial_conditions(test["initial_conditions"])
                rewrite_param(dt=test["dt"],
                              runtime=test["runtime"],
                              output_frequency=test["output_frequency"],
                              integrator=mode["integrator"],
                              coordinate_mode=mode["coordinate_mode"],
                              G=config.G,
                              pair_order=mode.get("pair_order", "canonical"),
                              adaptive_timesteps=mode.get("adaptive_timesteps", False),
                              timestep_levels=mode.get("timestep_levels"),
                              timestep_eta=mode.get("timestep_eta"),
                              timestep_refresh_interval=mode.get("timestep_refresh_interval"),
                              timestep_level_decrease_delay=mode.get("timestep_level_decrease_delay"))
                param_snapshot = DEFAULT_PARAM_PATH.read_text() if DEFAULT_PARAM_PATH.exists() else ""
                initial_conditions_snapshot = DEFAULT_INITIAL_CONDITIONS_PATH.read_text() if DEFAULT_INITIAL_CONDITIONS_PATH.exists() else ""

                clear_simulation_outputs()

                try:
                    run_executable()
                    output = read_output(DEFAULT_OUTPUT_PATH)
                    if use_diagnostics_csv:
                        diagnostics = read_diagnostics(DEFAULT_DIAGNOSTICS_PATH)
                        if diagnostics is None:
                            print("Falling back to recomputing diagnostics from output.csv")
                            diagnostics = compute_diagnostics_from_output(output, config)
                    else:
                        diagnostics = compute_diagnostics_from_output(output, config)
                    
                    append_benchmark_row(benchmark_rows=rows,
                                         run_number=run_number,
                                         mode=mode,
                                         test=test,
                                         status="success",
                                         diagnostics=diagnostics,
                                         engine="fewbodync",
                                         config=config,
                                         param_snapshot=param_snapshot,
                                         initial_conditions_snapshot=initial_conditions_snapshot)
                    rows[-1]["case"] = case["name"]
                    rows[-1]["case_description"] = case.get("description", "")

                    print(f"max |dE/E0|: {rows[-1]['max_dE_over_E0']:.6e}")
                    print(f"final |dE/E0|: {rows[-1]['final_dE_over_E0']:.6e}")
                    print(f"energy drift slope: {rows[-1]['energy_drift_slope']:.6e}")
                    print(f"energy boundedness: {rows[-1]['energy_boundedness_class']}")     

                except Exception as exc:
                    failure_message = str(exc)
                    append_benchmark_row(benchmark_rows=rows,
                                         run_number=run_number,
                                         mode=mode,
                                         test=test,
                                         status="failed",
                                         error=failure_message,
                                         engine="fewbodync",
                                         failure_type=type(exc).__name__,
                                         failure_message=failure_message,
                                         config=config,
                                         param_snapshot=param_snapshot,
                                         initial_conditions_snapshot=initial_conditions_snapshot)
                    rows[-1]["case"] = case["name"]
                    rows[-1]["case_description"] = case.get("description", "")

                    print(f"Energy Boundedness Run Failed: {failure_message}")
        results = pd.DataFrame(rows)
        raw_path = raw_dir / "energy_boundedness_summary.csv"
        save_raw_table(results, raw_path)
        readable = compact_benchmark_table(results, energy_boundedness_readable_columns)
        readable_path = readable_dir / "energy_boundedness_summary_readable.csv"
        save_readable_table(readable, readable_path)

        successful = results[results["status"] == "success"].copy()

        if not successful.empty:
            ranking = successful.sort_values(["case", "energy_boundedness_class", "energy_drift_fraction_of_max", "max_dE_over_E0"], na_position="last").copy()
            ranking_path = raw_dir / "energy_boundedness_rankings.csv"
            save_raw_table(ranking, ranking_path)
            ranking_readable = compact_benchmark_table(ranking, energy_boundedness_readable_columns)
            ranking_readable_path = readable_dir / "energy_boundedness_rankings_readable.csv"
            save_readable_table(ranking_readable, ranking_readable_path)
        
        print("\nEnergy Boundedness Suite Complete")
        print(f"Results saved under: {output_dir}")

        return results
    
    finally:
        if original_param_text is not None:
            DEFAULT_PARAM_PATH.write_text(original_param_text)
        if original_initial_conditions_text is not None:
            DEFAULT_INITIAL_CONDITIONS_PATH.write_text(original_initial_conditions_text)
        print("\nRestored original param.txt and initial_conditions.txt settings")

# ============================================================
# Test Convergence and Benchmark
# ============================================================

def TestBenchmark(reboundcompare: bool = False, rebound_integrator: str = "whfast", rebound_move_to_com: bool = False, **kwargs) -> None:
    run_benchmark_suite(rebound_compare=reboundcompare, rebound_integrator=rebound_integrator, rebound_move_to_com=rebound_move_to_com, **kwargs)

def TestConvergence(**kwargs) -> None:
    run_timestep_scaling_study( **kwargs)


# ============================================================
# Clear .csv files
# ============================================================

def clear_simulation_outputs() -> None:
    if DEFAULT_OUTPUT_PATH.exists():
        DEFAULT_OUTPUT_PATH.unlink()
    if DEFAULT_DIAGNOSTICS_PATH.exists():
        DEFAULT_DIAGNOSTICS_PATH.unlink()