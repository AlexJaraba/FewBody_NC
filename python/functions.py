import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import subprocess
import re

from dataclasses import dataclass
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

"""
FewBodyNC plotting and benchmark utilities

This file contains reusable helper functions used by plot_output.py

Responsibilities:
    - Read output.csv and diagnostics.csv
    - Compute diagnostics from saved Cartesian output
    - Plot orbits and conservation diagnostics
    - Run timestep convergence studies.
    - Run the full benchmark suite and save plots.

The C++ code writes physical Cartesian body states to output.csv.

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
# Default Benchmark Tests
# ============================================================
# Benchmark Tests:
#   Benchmark systems used to demonstrate both the strengths and limits of the current integrators.
#   These are written directly into data/initial_conditions.txt before each benchmark run.
# The benchmark suite runs every BENCHMARK_TESTS entry through every mode below.

DEFAULT_BENCHMARK_MODES = [
    {
        "name": "leapfrog_fixed",
        "integrator": "leapfrog",
        "pair_order": "canonical",
    },
    {
        "name": "hernandez_canonical_fixed",
        "integrator": "hernandez",
        "pair_order": "canonical",
    },
    {
        "name": "hernandez_strength_fixed",
        "integrator": "hernandez",
        "pair_order": "strength",
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
            (1.0,  0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
        ],
    },
    {
        "name": "Test2_CircumbinaryTriple",
        "dt": 0.001,
        "runtime": 258230.82,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (1e-3,  5.0, 0.0, 0.0, 0.0,  0.01089,  0.0),
        ],
    },
    {
        "name": "Test3_StrongerPerturbedTriple",
        "dt": 0.001,
        "runtime": 258230.82,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (0.05,  4.0, 0.0, 0.0, 0.0,  0.01220,  0.0),
        ],
    },
    {
        "name": "Test4_ScatteringEscape",
        "dt": 0.02,
        "runtime": 129115.42,
        "output_frequency": 1000,
        "initial_conditions": [
            (1.0, -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,  0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (0.1,  3.0, 0.0, 0.0, 0.0,  0.0040,   0.0),
        ],
    },
    {
        "name": "Test5_Figure8",
        "dt": 0.02,
        "runtime": 367677.02,
        "output_frequency": 10000,
        "initial_conditions": [
            (1.0, -0.97000436,  0.24308753, 0.0,  0.008019029,  0.007436436, 0.0),
            (1.0,  0.97000436, -0.24308753, 0.0,  0.008019029,  0.007436436, 0.0),
            (1.0,  0.0,         0.0,        0.0, -0.016038058, -0.014872872, 0.0),
        ],
    },
    {
        "name": "Test6_CloseEncounter",
        "dt": 0.002,
        "runtime": 10329.232,
        "output_frequency": 10000,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0,  0.0,    -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0,  0.0,     0.012166, 0.0),
            (0.01,  1.2, 0.2, 0.0, -0.0020,  0.0040,   0.0),
        ],
    },
    # {
    #     "name": "Test7_SolarSystem",
    #     "dt": 1.0,
    #     "runtime": 1668096750000.0,
    #     "output_frequency": 100000,
    #     "initial_conditions": [
    #         (1.0,        0.0,            0.0,            0.0,  0.0,            0.0,            0.0),
    #         (1.6601e-7,  0.3637531341,   0.1323953134,   0.0, -0.0094579726,  0.0259855662,  0.0),
    #         (2.4478e-6,  0.1872120975,   0.6986850598,   0.0, -0.0195403467,  0.0052358201,  0.0),
    #         (3.0035e-6, -0.6427876097,   0.7660444431,   0.0, -0.0131798787, -0.0110592314,  0.0),
    #         (3.2272e-7, -1.3195447212,  -0.7618395000,   0.0,  0.0069691551, -0.0120709307,  0.0),
    #         (9.5458e-4,  2.6022000000,  -4.5071426115,   0.0,  0.0065344536,  0.0037726685,  0.0),
    #         (2.8588e-4,  7.3406974806,   6.1595765486,   0.0, -0.0035730960,  0.0042582500,  0.0),
    #         (4.3662e-5, -18.0593886633,  6.5730799225,   0.0, -0.0013423302, -0.0036880218,  0.0),
    #         (5.1514e-5, -5.2285466296, -29.6525614432,   0.0,  0.0030879059, -0.0005444811,  0.0),
    #     ],
    # },
    # {
    #     "name": "Test8_SolarSystemInnerPlanets",
    #     "dt": 1.0,
    #     "runtime": 1668096750000.0,
    #     "output_frequency": 100000,
    #     "initial_conditions": [
    #         (1.0,        0.0,            0.0,           0.0,  0.0,            0.0,            0.0),
    #         (1.6601e-7,  0.3637531341,   0.1323953134,  0.0, -0.0094579726,  0.0259855662,  0.0),
    #         (2.4478e-6,  0.1872120975,   0.6986850598,  0.0, -0.0195403467,  0.0052358201,  0.0),
    #         (3.0035e-6, -0.6427876097,   0.7660444431,  0.0, -0.0131798787, -0.0110592314,  0.0),
    #         (3.2272e-7, -1.3195447212,  -0.7618395000,  0.0,  0.0069691551, -0.0120709307,  0.0),
    #     ],
    # },
    {
        "name": "Test9_HernandezTest1",
        "dt": 0.0008718353300301555,
        "runtime": 81.3712974694812,
        "output_frequency": 20,
        "initial_conditions": [
            (0.5, -0.0595, 0.0, 0.0, 0.0, -0.0582073046905078, 0.0),
            (0.5, -0.0405, 0.0, 0.0, 0.0, -0.0187361351991833, 0.0),
            (1.0, 0.05,   0.0, 0.0, 0.0,  0.0384717371585948, 0.0),
        ],
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
            (0.2,  0.4623155757271038,  -0.5623388195598369,  0.4440647837970507, -0.00497623904059215,   0.007927333296801377,  0.00399728972459227),
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

    for line_number, line in enumerate(param_path.read_text().splitlines(), start=1):
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue
        
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"Param file line {line_number} must have exactly 2 values: key value")
        key, value = parts
        if key in params:
            raise ValueError(f"Duplicate key '{key!r}' found in param file at line {line_number}.")
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
        if len(parts) not in (7, 8):
            raise ValueError(f"Initial-conditions line {line_number} must have 7 or 8 values: mass x y z vx vy vz [radius]")
        
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
    
    if "radius" not in df.columns:
        df["radius"] = 0.0
    
    df = df.sort_values(by=["time", "id"]).reset_index(drop=True)

    return df

def read_diagnostics(path: Path = DEFAULT_DIAGNOSTICS_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    
    df = pd.read_csv(path)
    required = {"time", "total_energy"}
    missing = required - set(df.columns)

    if missing:
        print(f"Warning: Missing required columns in diagnostics: {sorted(missing)}")
        return None
    
    if "kinetic_energy" not in df.columns:
        df["kinetic_energy"] = np.nan
    if "potential_energy" not in df.columns:
        df["potential_energy"] = np.nan
    if "timestep" not in df.columns:
        time = df["time"].to_numpy(dtype=float)
        if len(time) >= 2:
            dt = np.diff(time, prepend=time[0])
            if len(dt) > 1:
                dt[0] = dt[1]
            df["timestep"] = dt
        else:
            df["timestep"] = np.nan
    
    alias_default = {
        "linear_momentum_x": "linear_momentum",
        "linear_momentum_y": None,
        "linear_momentum_z": None,
        "angular_momentum_x": None,
        "angular_momentum_y": None,
        "angular_momentum_z": "angular_momentum",
        "com_x": "com_drift",
        "com_y": None,
        "com_z": None,
    }

    for column, fallback in alias_default.items():
        if column not in df.columns:
            if fallback is not None and fallback in df.columns:
                df[column] = df[fallback]
            else:
                df[column] = np.nan
    
    return df.sort_values(by=["time"]).reset_index(drop=True)

# ============================================================
# Helper Functions
# ============================================================

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
                         rebound_comparison: dict | None = None,
                         config: PlotConfig | None = None,
                         param_snapshot: str = "",
                         initial_conditions_snapshot: str = "",
                         failure_details: dict | None = None) -> None:
    
    if config is None:
        config = PlotConfig()
    body_count = len(test.get("initial_conditions", []))
    row = {"run_number": run_number,
           "mode": mode["name"],
           "integrator": mode["integrator"],
           "pair_order": mode.get("pair_order", "canonical"),
           "test": test["name"],
           "dt": test["dt"],
           "runtime": test["runtime"],
           "output_frequency": test["output_frequency"],
           "G": config.G,
           "body_count": body_count,
           "param_snapshot": param_snapshot,
           "initial_conditions_snapshot": initial_conditions_snapshot,
           "status": status,
           "error": error,
           "failure_type": failure_type,
           "failure_message": failure_message,
           "failed_time": np.nan,
           "failed_dt": np.nan,
           "failed_pair_i": np.nan,
           "failed_pair_j": np.nan,
           "failed_distance": np.nan,
           "failed_relative_speed": np.nan,
           "failed_iterations": np.nan,
           "failed_mode": mode["name"],
           "failed_test": test["name"],
           "returncode": returncode,
           "stdout_tail": stdout_tail,
           "stderr_tail": stderr_tail,
           "plot_path": str(plot_path) if plot_path is not None else "",
           "final_time": np.nan,
           "initial_energy": np.nan,
           "final_energy": np.nan,
           "max_dE_over_E0": np.nan,
           "final_dE_over_E0": np.nan,
           "energy_drift_slope": np.nan,
           "energy_abs_drift_slope": np.nan,
           "energy_drift_over_run": np.nan,
           "energy_drift_fraction_of_max": np.nan,
           "energy_rms_dE_over_E0": np.nan,
           "energy_boundedness_class": "",
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
           "final_dRcm": np.nan}
    
    if failure_details is not None:
        row.update(failure_details)
        if not row["failure_message"]:
            row["failure_message"] = failure_message
        if not row["error"]:
            row["error"] = row["failure_message"]
    if diagnostics is not None and not diagnostics.empty:
        time = diagnostics["time"].to_numpy(dtype=float)
        energy = diagnostics["total_energy"].to_numpy()
        angular = diagnostics["angular_momentum"].to_numpy()
        px = diagnostics.get("linear_momentum_x", diagnostics["linear_momentum"]).to_numpy(dtype=float)
        xcm = diagnostics.get("com_x", diagnostics.get("com_drift", pd.Series(np.nan, index=diagnostics.index))).to_numpy(dtype=float)
        metric_summary = error_diagnostic_metric_summary(diagnostics, config)

        row.update({"final_time": time[-1] if len(time) else np.nan,
                    "initial_energy": energy[0] if len(energy) else np.nan,
                    "final_energy": energy[-1] if len(energy) else np.nan,
                    "max_dE_over_E0": metric_summary["max_dE_over_E0"],
                    "final_dE_over_E0": metric_summary["final_dE_over_E0"],
                    "initial_angular_momentum": angular[0] if len(angular) else np.nan,
                    "final_angular_momentum": angular[-1] if len(angular) else np.nan,
                    "max_dL_over_L0": metric_summary["max_dL_over_L0"],
                    "final_dL_over_L0": metric_summary["final_dL_over_L0"],
                    "initial_linear_momentum": px[0] if len(px) else np.nan,
                    "final_linear_momentum": px[-1] if len(px) else np.nan,
                    "max_dP": metric_summary["max_dP"],
                    "final_dP": metric_summary["final_dP"],
                    "initial_com_drift": xcm[0] if len(xcm) else np.nan,
                    "final_com_drift": xcm[-1] if len(xcm) else np.nan,
                    "max_dRcm": metric_summary["max_dRcm"],
                    "final_dRcm": metric_summary["final_dRcm"],})
        row.update(energy_drift_summary(diagnostics, config))

    benchmark_rows.append(row)

def finite_log2_ratio(coarse_error: float, fine_error: float) -> float:
    if not np.isfinite(coarse_error) or not np.isfinite(fine_error):
        return float("nan")
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return float("nan")
    return float(np.log2(coarse_error / fine_error))

def safe_filename(name: str) -> str:
    safe = []
    for char in str(name):
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    result = "".join(safe).strip("._")
    return result if result else "unnamed"

def parse_benchmark_failure_details(stdout_tail: str, stderr_tail: str, mode_name: str, test_name: str) -> dict:
    text = f"{stdout_tail}\n{stderr_tail}"
    details = {"failed_time": np.nan,
               "failed_dt": np.nan,
               "failed_pair_i": np.nan,
               "failed_pair_j": np.nan,
               "failed_distance": np.nan,
               "failed_relative_speed": np.nan,
               "failed_iterations": np.nan,
               "failed_mode": mode_name,
               "failed_test": test_name,
               "failure_message": ""}
    what_match = re.search(r"what\(\):\s*(.*)", text, flags=re.DOTALL)

    if what_match:
        details["failure_message"] = " ".join(what_match.group(1).strip().split())
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            lowered = line.lower()
            if "failed" in lowered or "error" in lowered or "exception" in lowered:
                details["failure_message"] = line
                break
    
    pair_match = re.search(r"pair=\((\d+)\s*,\s*(\d+)\)", text)

    if pair_match:
        details["failed_pair_i"] = int(pair_match.group(1))
        details["failed_pair_j"] = int(pair_match.group(2))
    
    patterns = {"failed_time": r"(?:failed_time|time)\s*=\s*([-+0-9.eE]+)",
                "failed_dt": r"(?:failed_dt|dt)\s*=\s*([-+0-9.eE]+)",
                "failed_distance": r"(?:failed_distance|distance)\s*=\s*([-+0-9.eE]+)",
                "failed_relative_speed": r"(?:failed_relative_speed|relative_speed)\s*=\s*([-+0-9.eE]+)",
                "failed_iterations": r"(?:failed_iterations|last_iterations|iterations)\s*=\s*(\d+)"}
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            details[key] = int(value) if key == "failed_iterations" else float(value)
    
    return details

def compact_benchmark_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing_columns = [column for column in columns if column in df.columns]
    compact = df[existing_columns].copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column]):
            compact[column] = compact[column].map(format_scientific_or_blank)
    return compact

def format_scientific_or_blank(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating, int, np.integer)):
        return f"{float(value):.6e}"
    return str(value)

def finite_linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(finite) < 2:
        return float("nan")
    
    xf = x[finite]
    yf = y[finite]
    x_centered = xf - np.mean(xf)
    y_centered = yf - np.mean(yf)
    denominator = np.sum(x_centered * x_centered)

    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    
    return float(np.sum(x_centered * y_centered) / denominator)

# def classify_energy_boundedness(max_abs_dE: float, final_abs_dE: float, drift_over_run: float, drift_fraction_of_max: float) -> str:
#     if not (np.isfinite(max_abs_dE) and np.isfinite(final_abs_dE) and np.isfinite(drift_over_run) and np.isfinite(drift_fraction_of_max)):
#         return "undetermined"
#     if max_abs_dE <= 1e-14:
#         return "bounded_like_machine_precision"
#     if drift_fraction_of_max <= 0.25 and final_abs_dE <= 0.75 * max_abs_dE:
#         return "bounded_like"
#     return "mixed_unclear"

def energy_drift_summary(diagnostics: pd.DataFrame, config: PlotConfig) -> dict:
    if diagnostics is None or diagnostics.empty:
        return {"energy_drift_slope": float("nan"), 
                "energy_abs_drift_slope": float("nan"), 
                "energy_drift_over_run": float("nan"), 
                "energy_drift_fraction_of_max": float("nan"), 
                "energy_rms_dE_over_E0": float("nan")}
    
    time = diagnostics["time"].to_numpy(dtype=float)
    energy = diagnostics["total_energy"].to_numpy(dtype=float)
    signed_dE = error_signed_relative_energy(energy, config.epsilon)
    abs_dE = np.abs(signed_dE)
    max_abs_dE = compute_finite_max(abs_dE)
    final_abs_dE = float(abs_dE[-1]) if len(abs_dE) else float("nan")
    energy_drift_slope = finite_linear_slope(time, signed_dE)
    energy_abs_drift_slope = finite_linear_slope(time, abs_dE)
    finite_time = time[np.isfinite(time)]

    if finite_time.size >= 2 and np.isfinite(energy_drift_slope):
        duration = float(finite_time[-1] - finite_time[0])
        drift_over_run = abs(energy_drift_slope) * duration
    else:
        drift_over_run = float("nan")
    if np.isfinite(max_abs_dE) and max_abs_dE > 0.0 and np.isfinite(drift_over_run):
        drift_fraction_of_max = drift_over_run / max_abs_dE
    else:
        drift_fraction_of_max = float("nan")

    finite_abs_dE = abs_dE[np.isfinite(abs_dE)]
    rms_dE = float(np.sqrt(np.mean(finite_abs_dE * finite_abs_dE))) if finite_abs_dE.size else float("nan")
    
    return {"energy_drift_slope": energy_drift_slope, 
            "energy_abs_drift_slope": energy_abs_drift_slope, 
            "energy_drift_over_run": drift_over_run,
            "energy_drift_fraction_of_max": drift_fraction_of_max,
            "energy_rms_dE_over_E0": rms_dE}

def add_test_rank(df: pd.DataFrame, metric: str = "max_dE_over_E0", rank_column: str = "rank_in_test") -> pd.DataFrame:
    ranked = df.copy()
    if "test" not in ranked.columns or metric not in ranked.columns:
        ranked[rank_column] = np.nan
        return ranked
    ranked[rank_column] = (ranked.groupby("test")[metric].rank(method="min", ascending=True).astype("Int64"))
    return ranked

def save_raw_table(df: pd.DataFrame, csv_path: Path, write_parquet: bool = True) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved raw CSV to {csv_path}")
    if write_parquet:
        parquet_path = csv_path.with_suffix(".parquet")
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"Saved raw Parquet to {parquet_path}")
        except Exception as exc:
            print(f"Skipped Parquet for {csv_path.name}: {exc}. Install pyarrow with: pip install pyarrow")

def save_readable_table(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    html_path = csv_path.with_suffix(".html")
    df.to_html(html_path, index=False, escape=False)
    print(f"Saved readable CSV to {csv_path}")
    print(f"Saved readable HTML to {html_path}")

def add_diagnostics_error_columns(diagnostics: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    df = diagnostics.copy()

    if df.empty:
        return df
    if "time" not in df.columns and "times" in df.columns:
        df = df.rename(columns={"times": "time"})
    
    def ensure_column(column: str, default=np.nan) -> None:
        if column not in df.columns:
            df[column] = default

    if "linear_momentum_x" not in df.columns and "linear_momentum" in df.columns:
        df["linear_momentum_x"] = df["linear_momentum"]
    if "angular_momentum_z" not in df.columns and "angular_momentum" in df.columns:
        df["angular_momentum_z"] = df["angular_momentum"]
    if "com_x" not in df.columns and "com_drift" in df.columns:
        df["com_x"] = df["com_drift"]
    
    for column in ["angular_momentum_x", "angular_momentum_y", "angular_momentum_z", 
                   "linear_momentum_x", "linear_momentum_y", "linear_momentum_z", 
                   "com_x", "com_y", "com_z",
                   "com_vx", "com_vy", "com_vz", "total_energy"]:
        ensure_column(column)

    time = df["time"].to_numpy(dtype=float)
    t0 = time[0] if len(time) else 0.0
    energy = df["total_energy"].to_numpy(dtype=float)
    if "dE_over_E0" not in df.columns:
        df["dE_over_E0"] = error_relative(energy, config.epsilon)
    if "dE_abs" not in df.columns:
        df["dE_abs"] = error_absolute(energy)

    for component in ["x", "y", "z"]:
        L = df[f"angular_momentum_{component}"].to_numpy(dtype=float)
        P = df[f"linear_momentum_{component}"].to_numpy(dtype=float)
        R = df[f"com_{component}"].to_numpy(dtype=float)
        V = df[f"com_v{component}"].to_numpy(dtype=float)
        R0 = R[0] if len(R) else np.nan
        V0 = V[0] if len(V) else np.nan

        dL = error_absolute(L)
        dP = error_absolute(P)
        dR = error_absolute(R)
        C = R - R0 - V0 * (time - t0)

        df[f"dL_{component}"] = dL
        df[f"dP_{component}"] = dP
        df[f"dRcm_{component}"] = dR
        df[f"com_integral_{component}"] = C
        df[f"dCcm_{component}"] = np.abs(C)
        df[f"dL{component}"] = dL
        df[f"dP{component}"] = dP

    df["dL"] = df["dLz"]
    df["dPcm_x"] = df["dPx"]
    df["dPcm_y"] = df["dPy"]
    df["dXcm"] = df["dRcm_x"]
    df["dYcm"] = df["dRcm_y"]
    
    return df

def thin_for_plotting(time: np.ndarray, values: np.ndarray, max_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(time) <= max_points:
        return time, values
    
    indices = np.linspace(0, len(time) - 1, max_points).astype(int)
    return time[indices], values[indices]

def load_diagnostics(output_df: pd.DataFrame, config: PlotConfig, path: Path = DEFAULT_DIAGNOSTICS_PATH) -> pd.DataFrame:
    diagnostics = read_diagnostics(path)
    required_components = {"linear_momentum_x", "linear_momentum_y", "linear_momentum_z",
                           "angular_momentum_x", "angular_momentum_y", "angular_momentum_z",
                           "com_x", "com_y", "com_z", "com_vx", "com_vy", "com_vz"}
    if diagnostics is None:
        print("No useable diagnostics file found. Computing diagnostics from output...")
        return compute_diagnostics_from_output(output_df, config)
    missing = sorted(required_components - set(diagnostics.columns))
    unusable = []
    for column in sorted(required_components & set(diagnostics.columns)):
        values = pd.to_numeric(diagnostics[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).any():
            unusable.append(column)
    
    if missing or unusable:
        print(f"Diagnostics file is missing or unusable columns: {missing + unusable}. Computing diagnostics from output...")
        return compute_diagnostics_from_output(output_df, config)
    
    return diagnostics

def snapshot_optional_file(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None

def restore_optional_file(path: Path, content: bytes | None) -> None:
    if content is None:
        if path.exists():
            path.unlink()
    else:
        path.write_bytes(content)

# ============================================================
# Compute Functions
# ============================================================

def compute_diagnostics_from_output(df: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    rows = []

    for time_values, group in df.groupby("time", sort=True):
        group = group.sort_values("id")
        mass = group["mass"].to_numpy(dtype=float)
        pos = group[["x", "y", "z"]].to_numpy(dtype=float)
        vel = group[["vx", "vy", "vz"]].to_numpy(dtype=float)

        if mass.size == 0:
            continue

        total_mass = float(np.sum(mass))
        kinetic = 0.5 * np.sum(mass * np.sum(vel * vel, axis=1))
        potential = 0.0
        for i in range(len(mass)):
            dr = pos[i + 1:] - pos[i]
            if len(dr) == 0:
                continue
            r = np.linalg.norm(dr, axis=1)
            valid = r > config.epsilon
            if np.any(valid):
                potential -= np.sum(config.G * mass[i] * mass[i + 1:][valid] / r[valid])
        total_energy = kinetic + potential
        momentum_vec = np.sum(mass[:, None] * vel, axis=0)
        angular_vec = np.sum(np.cross(pos, mass[:, None] * vel), axis=0)

        if total_mass != 0.0:
            rcm = np.sum(mass[:, None] * pos, axis=0) / total_mass
            vcm = momentum_vec / total_mass
        else:
            rcm = np.full(3, np.nan)
            vcm = np.full(3, np.nan)
        
        rows.append({
            "time": time_values,
            "kinetic_energy": float(kinetic),
            "potential_energy": float(potential),
            "total_energy": float(total_energy),

            "angular_momentum": float(np.linalg.norm(angular_vec)),
            "angular_momentum_x": float(angular_vec[0]),
            "angular_momentum_y": float(angular_vec[1]),
            "angular_momentum_z": float(angular_vec[2]),

            "linear_momentum": float(np.linalg.norm(momentum_vec)),
            "linear_momentum_x": float(momentum_vec[0]),
            "linear_momentum_y": float(momentum_vec[1]),
            "linear_momentum_z": float(momentum_vec[2]),

            "com_drift": float(np.linalg.norm(rcm)),
            "com_x": float(rcm[0]),
            "com_y": float(rcm[1]),
            "com_z": float(rcm[2]),

            "com_vx": float(vcm[0]),
            "com_vy": float(vcm[1]),
            "com_vz": float(vcm[2]),
        })
    
    if not rows:
        raise ValueError("No valid data found in output DataFrame to compute diagnostics.")
    
    diagnostics = pd.DataFrame(rows).sort_values(by="time").reset_index(drop=True)

    if "timestep" not in diagnostics.columns:
        times = diagnostics["time"].to_numpy(dtype=float)
        if len(times) >= 2:
            dt = np.diff(times, prepend=times[0])
            if len(dt) > 1:
                dt[0] = dt[1]
            diagnostics["timestep"] = dt
        else:
            diagnostics["timestep"] = np.nan

    return add_diagnostics_error_columns(diagnostics, config)

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

# def error_safe_log_values(values: np.ndarray, floor: float = 1e-300, ceiling: float = 1e50) -> np.ndarray:
#     values = np.asarray(values, dtype=float)
    
#     safe = np.full_like(values, np.nan, dtype=float)
#     finite = np.isfinite(values)

#     safe[finite] = np.abs(values[finite])
#     safe[finite] = np.clip(safe[finite], floor, ceiling)

#     return safe

def error_print_summary(diagnostics: pd.DataFrame, config: PlotConfig) -> None:
    diagnostics = add_diagnostics_error_columns(diagnostics, config)

    dE = diagnostics["dE_over_E0"].to_numpy(dtype=float)
    dLz = diagnostics["dLz"].to_numpy(dtype=float)
    dPx = diagnostics["dPx"].to_numpy(dtype=float)
    dPy = diagnostics["dPy"].to_numpy(dtype=float)
    dXcm = diagnostics["dRcm_x"].to_numpy(dtype=float)
    dYcm = diagnostics["dRcm_y"].to_numpy(dtype=float)

    print("Summary of Diagnostics:")
    print(f"Max |dE/E0|:", compute_finite_max(dE))
    print(f"Max |dLz|:", compute_finite_max(dLz))
    print(f"Max |dPcm_x|:", compute_finite_max(dPx))
    print(f"Max |dPcm_y|:", compute_finite_max(dPy))
    print(f"Max |dXcm|:", compute_finite_max(dXcm))
    print(f"Max |dYcm|:", compute_finite_max(dYcm))
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
               }

    summary.update(energy_drift_summary(diagnostics, config))

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
    y = np.abs(values)
    y[y <= 0.0] = np.nan

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
            positive = y[valid & (y > 0.0)]
            ymin = float(np.nanmin(positive)) if positive.size else np.finfo(float).tiny
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
    diagnostics = add_diagnostics_error_columns(diagnostics, config)
    time = diagnostics["time"].to_numpy(dtype=float)
    dE = diagnostics["dE_over_E0"].to_numpy(dtype=float)
    dL = diagnostics["dLz"].to_numpy(dtype=float)
    dPcm_x = diagnostics["dPx"].to_numpy(dtype=float)
    dPcm_y = diagnostics["dPy"].to_numpy(dtype=float)
    dXcm = diagnostics["dRcm_x"].to_numpy(dtype=float)
    dYcm = diagnostics["dRcm_y"].to_numpy(dtype=float)

    error_print_summary(diagnostics, config)

    fig = plt.figure(figsize=config.figure_size)
    gs = fig.add_gridspec(3, 4)

    ax_orbit = fig.add_subplot(gs[:, 0:2])
    plot_orbits(ax_orbit, output_df, config)

    ax_energy = fig.add_subplot(gs[0, 2])
    plot_error(ax_energy, time, dE, "Relative Energy Error", r"$|\Delta E/E|$")

    ax_angular = fig.add_subplot(gs[0, 3])
    plot_error(ax_angular, time, dL, "Angular Momentum Error", r"$|\Delta L|$")

    ax_px = fig.add_subplot(gs[1, 2])
    plot_error(ax_px, time, dPcm_x, "COM Momentum X Error", r"$|\Delta p_{\rm cm,x}|$")

    ax_py = fig.add_subplot(gs[1, 3])
    plot_error(ax_py, time, dPcm_y, "COM Momentum Y Error", r"$|\Delta p_{\rm cm,y}|$")

    ax_xcm = fig.add_subplot(gs[2, 2])
    plot_error(ax_xcm, time, dXcm, "COM Drift X Error", r"$|\Delta x_{\rm cm}|$")

    ax_ycm = fig.add_subplot(gs[2, 3])
    plot_error(ax_ycm, time, dYcm, "COM Drift Y Error", r"$|\Delta y_{\rm cm}|$")

    if save_path is not None:
        title = f"FewBodyNC Verification Suite: {save_path.parent.name} / {save_path.stem}"
    else:
        title = "FewBodyNC Verification Suite"
        
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_animation():
    data = read_output(DEFAULT_OUTPUT_PATH)
    times = data["time"].unique()
    body_ids = data["id"].unique()

    base_interval = 20

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    bodies_plot = ax.scatter([], [])
    trails = {body_id: ax.plot([], [])[0] for body_id in body_ids}

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(data["x"].min() - 0.01, data["x"].max() + 0.01)
    ax.set_ylim(data["y"].min() - 0.01, data["y"].max() + 0.01)

    def update(frame):
        time = times[frame]
        current = data[data["time"] == time]
        bodies_plot.set_offsets(current[["x", "y"]].values)
        past = data[data["time"] <= time]
        for body_id in body_ids:
            trajectory = past[past["id"] == body_id]
            trails[body_id].set_data(trajectory["x"],  trajectory["y"])
        ax.set_title(f"Time: {time:.6f}")
        return [bodies_plot, *trails.values()]

    ani = FuncAnimation(fig, update, frames=len(times), interval=base_interval, blit=False)
    # slider_ax = fig.add_axes((0.2, 0.07, 0.6, 0.03))
    # speed_slider = Slider(ax=slider_ax, label="Speed", valmin=0.25, valmax=2.0, valinit=1.0, valstep=0.25)
    # def change_speed(speed):
    #     new_interval = base_interval / speed
    #     ani._interval = new_interval
    #     ani.event_source.interval = new_interval
    # speed_slider.on_changed(change_speed)
    plt.show()

# ============================================================
# Rewrite param.txt and initial_conditions.txt
# ============================================================

def rewrite_param(dt: float, 
                  runtime: float, 
                  output_frequency: int, 
                  integrator: str, 
                  G: float = 0.000296014912,
                  pair_order: str = "canonical", 
                  param_path: Path = DEFAULT_PARAM_PATH) -> None:
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find param file: {param_path}")

    replacements = {"output_frequency": f"output_frequency {output_frequency}",
                    "runtime": f"runtime {runtime}",  
                    "timestep": f"timestep {dt}", 
                    "integrator": f"integrator {integrator}",
                    "gravitational_constant": f"gravitational_constant {G}",
                    "pair_order": f"pair_order {pair_order}"}
    ordered_keys = ["output_frequency", 
                    "runtime", 
                    "timestep", 
                    "integrator", 
                    "gravitational_constant", 
                    "pair_order"]

    param_path.write_text("\n".join(replacements[key] for key in ordered_keys) + "\n")

"""
Write initial conditions in the exact format expected by the C++ reader:
    mass x y z vx vy vz
No header row is written
"""
def rewrite_initial_conditions(rows: list[tuple[float, ...]], output_path: Path = DEFAULT_INITIAL_CONDITIONS_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for row in rows:
            if len(row) not in (7, 8):
                raise ValueError("Each initial-condition row must have 7 or 8 values: mass, x, y, z, vx, vy, vz, [radius]")
            f.write(" ".join(f"{value:.16g}" for value in row) + "\n")

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

"""
Run a convergence study using the current settings in data/param.txt
Only the timestep is changed during the sweep.
The original param.txt is restored afterward.
"""
def run_convergence_case(case_name: str, 
                         initial_conditions: list[tuple[float, ...]],  
                         dts: tuple[float, ...], 
                         runtime: float, 
                         base_dt: float,
                         base_output_frequency: int, 
                         integrator: str,
                         pair_order: str,
                         G: float) -> pd.DataFrame:
    config = PlotConfig(G=G)
    rows = []
    final_positions_by_dt = {}
    coarse_steps = max(1, int(round(runtime / base_dt)))
    aligned_runtime = coarse_steps * base_dt
    physical_output_interval = base_output_frequency * base_dt

    print("\n" + "-" * 90)
    print(f"Convergence case: {case_name}")
    print(f"integrator           = {integrator}")
    print(f"pair_order           = {pair_order}")
    print(f"runtime              = {aligned_runtime}")
    print(f"output_frequency     = {physical_output_interval}")
    print(f"G                    = {G}")
    print(f"dt ladder            = {dts}")
    print("-" * 90)

    rewrite_initial_conditions(initial_conditions)

    for dt in dts:
        refinement = base_dt / dt
        step_count = int(round(coarse_steps * refinement))
        run_runtime = step_count * dt
        output_frequency = max(1, int(round(physical_output_interval / dt)))

        print(f"\nRunning {case_name}: dt = {dt}, steps = {step_count}, final time = {run_runtime}")
        rewrite_param(dt=dt, runtime=run_runtime, output_frequency=output_frequency, integrator=integrator, G=G, pair_order=pair_order)
        param_snapshot = DEFAULT_PARAM_PATH.read_text() if DEFAULT_PARAM_PATH.exists() else ""
        clear_simulation_outputs()
        try:
            run_executable()
            output = read_output(DEFAULT_OUTPUT_PATH)
            diagnostics = load_diagnostics(output, config)
            final_time = float(output["time"].max())
            time_tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(aligned_runtime))
            if not np.isclose(final_time, aligned_runtime, rtol=0.0, atol=time_tolerance):
                raise ValueError(f"Convergence run ended at {final_time:.17g}, expected {aligned_runtime:.17g} (tolerance {time_tolerance:.17g})")
            final_positions_by_dt[dt] = compute_final_positions(output)
            diagnostics_summary = error_diagnostic_metric_summary(diagnostics, config)
            row = {
                "case": case_name,
                "dt": dt,
                "runtime": aligned_runtime,
                "steps": step_count,
                "output_frequency": output_frequency,
                "integrator": integrator,
                "pair_order": pair_order,
                "G": G,
                "self_convergence_error": np.nan,
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
                "runtime": aligned_runtime,
                "steps": step_count,
                "output_frequency": output_frequency,
                "integrator": integrator,
                "pair_order": pair_order,
                "G": G,
                "self_convergence_error": np.nan,
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
        if coarse["status"] == "success" and fine["status"] == "success":
            coarse_positions = final_positions_by_dt[coarse["dt"]]
            fine_positions = final_positions_by_dt[fine["dt"]]
            if coarse_positions is not None and fine_positions is not None:
                coarse["self_convergence_error"] = error_rms_position(coarse_positions, fine_positions)
    for i in range(len(rows) - 1):
        coarse_error = rows[i]["self_convergence_error"]
        fine_error = rows[i + 1]["self_convergence_error"]
        if np.isfinite(coarse_error) and np.isfinite(fine_error) and coarse_error > 0.0 and fine_error > 0.0:
            rows[i]["error_ratio_to_next_finer"] = coarse_error / fine_error
            rows[i]["observed_order_to_next_finer"] = finite_log2_ratio(coarse_error, fine_error)
        
    print("\nSelf-convergence ratios:")
    for row in rows:
        print(f"   dt = {row['dt']:.8g}, "
              f"error = {row['self_convergence_error']:.6e}, "
              f"ratio = {row['error_ratio_to_next_finer']:.6e}, "
              f"order = {row['observed_order_to_next_finer']:.6e}")
        
    return pd.DataFrame(rows)

def run_timestep_scaling_study(dts_multiples: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125),
                               param_path: Path = DEFAULT_PARAM_PATH,
                               output_dir: Path = DEFAULT_BENCHMARK_PLOT_DIR / "convergence") -> pd.DataFrame:
    params = read_param(param_path)
    dt = float(params["timestep"])
    runtime = float(params["runtime"])
    output_frequency = int(params["output_frequency"])
    integrator = params["integrator"]
    pair_order = params["pair_order"]
    G = float(params["gravitational_constant"])
    initial_conditions = read_initial_conditions(DEFAULT_INITIAL_CONDITIONS_PATH)

    if dt <= 0.0 or runtime <= 0.0 or output_frequency <= 0:
        raise ValueError("Convergence requires positive timestep, runtime, and output_frequency.")
    if not dts_multiples or dts_multiples[0] != 1.0:
        raise ValueError("The convergence ladder must begin with 1.0 times the current timestep.")
    if any(not np.isfinite(m) or m <= 0.0 for m in dts_multiples):
        raise ValueError("All convergence ladder multiples must be positive finite numbers.")
    for previous, current in zip(dts_multiples, dts_multiples[1:]):
        if not np.isclose(current, 0.5 * previous, rtol=0.0, atol=16.0 * np.finfo(float).eps):
            raise ValueError("Convergence ladder multiples must be strictly halve at each refinement step.")

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    readable_dir = output_dir / "readable"
    readable_dir.mkdir(parents=True, exist_ok=True)

    original_param_text = param_path.read_text()
    original_initial_conditions_text = DEFAULT_INITIAL_CONDITIONS_PATH.read_text()
    original_output = snapshot_optional_file(DEFAULT_OUTPUT_PATH)
    original_diagnostics = snapshot_optional_file(DEFAULT_DIAGNOSTICS_PATH)

    convergence_readable_columns = [
        "case",
        "dt",
        "runtime",
        "steps",
        "output_frequency",
        "integrator",
        "pair_order",
        "self_convergence_error",
        "error_ratio_to_next_finer",
        "observed_order_to_next_finer",
        "max_dE_over_E0",
        "final_dE_over_E0",
        "max_dL_over_L0",
        "max_dP",
        "max_dRcm",
        "status",
        "error",
    ]

    try: 
        print("\nConvergence test using current param.txt settings:")
        print(f"runtime              = {runtime}")
        print(f"output_frequency     = {output_frequency}")
        print(f"integrator           = {integrator}")
        print(f"pair_order           = {pair_order}")
        print(f"gravitational_constant = {G}")
        print(f"dts_multiples         = {dts_multiples}")

        dts = tuple(float(dt * multiple) for multiple in dts_multiples)
        results = run_convergence_case(case_name="CurrentParamScalingStudy",
                                       initial_conditions=initial_conditions,
                                       dts=dts,
                                       runtime=runtime,
                                       base_dt=dt,
                                       base_output_frequency=output_frequency,
                                       integrator=integrator,
                                       pair_order=pair_order,
                                       G=G)
        
        save_raw_table(results, raw_dir / "current_param_scaling_study.csv")
        readable = compact_benchmark_table(results, convergence_readable_columns)
        save_readable_table(readable, readable_dir / "current_param_scaling_study_readable.csv")

        successful = results[(results["status"] == "success") & (np.isfinite(results["self_convergence_error"])) & (results["self_convergence_error"] > 0.0)].sort_values("dt", ascending=False)
        if not successful.empty:
            plt.figure(figsize=(7, 5), constrained_layout=True)
            x = successful["dt"].to_numpy(dtype=float)
            y = successful["self_convergence_error"].to_numpy(dtype=float)
            plt.loglog(x, y, marker="o", label="Measured self-convergence")
            plt.gca().invert_xaxis()
            plt.xlabel("Timestep dt")
            plt.ylabel(f"RMS final-position self-convergence error")
            plt.title(f"Timestep convergence study: {integrator}, {pair_order}")
            plt.grid(True, which="both", ls="--", alpha=0.4)
            plt.legend()
            plt.show()
        return results
    
    finally:
        param_path.write_text(original_param_text)
        DEFAULT_INITIAL_CONDITIONS_PATH.write_text(original_initial_conditions_text)
        restore_optional_file(DEFAULT_OUTPUT_PATH, original_output)
        restore_optional_file(DEFAULT_DIAGNOSTICS_PATH, original_diagnostics)
        
        print("\nRestored original param.txt, initial_conditions.txt, output.csv, and diagnostics.csv settings")

"""
# Run every benchmark test in every selected mode and save the resulting plots.
# This function temporarily overwrites data/initial_conditions.txt and data/param.txt.
# The original files should be restored at the end of the run.

"""
def run_benchmark_suite(modes: list[dict] | None = None, output_dir: Path = DEFAULT_BENCHMARK_PLOT_DIR) -> None:
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
    original_output = snapshot_optional_file(DEFAULT_OUTPUT_PATH)
    original_diagnostics = snapshot_optional_file(DEFAULT_DIAGNOSTICS_PATH)

    if DEFAULT_PARAM_PATH.exists():
        original_param_text = DEFAULT_PARAM_PATH.read_text()
    if DEFAULT_INITIAL_CONDITIONS_PATH.exists():
        original_initial_conditions_text = DEFAULT_INITIAL_CONDITIONS_PATH.read_text()

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

                rewrite_initial_conditions(test["initial_conditions"])
                rewrite_param(dt = test["dt"], 
                            runtime = test["runtime"], 
                            output_frequency = test["output_frequency"], 
                            integrator = mode["integrator"], 
                            G = config.G,
                            pair_order = mode.get("pair_order", "canonical"))
                
                param_snapshot = DEFAULT_PARAM_PATH.read_text() if DEFAULT_PARAM_PATH.exists() else ""
                initial_conditions_snapshot = (DEFAULT_INITIAL_CONDITIONS_PATH.read_text() if DEFAULT_INITIAL_CONDITIONS_PATH.exists() else "")

                clear_simulation_outputs()
                
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
                                         failure_type=type(exc).__name__,
                                         failure_message=failure_message,
                                         config=config,
                                         param_snapshot=param_snapshot,
                                         initial_conditions_snapshot=initial_conditions_snapshot,
                                         failure_details=failure_details)
                    continue
                
                try:
                    output = read_output(DEFAULT_OUTPUT_PATH)
                    diagnostics = load_diagnostics(output, config)
                    plot_path = mode_dir / f"{safe_filename(test['name'])}.png"
                    plot_verification_suite(output_df=output, diagnostics=diagnostics, config=config, save_path=plot_path, show=False,)
                    print(f"Saved plot to {plot_path}")

                    append_benchmark_row(benchmark_rows=benchmark_rows,
                                        run_number=run_number,
                                        mode=mode,
                                        test=test,
                                        status="success",
                                        plot_path=plot_path,
                                        diagnostics=diagnostics,
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
                                         failure_type=type(exc).__name__,
                                         failure_message=failure_message,
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
            "pair_order",
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
            "plot_path",
        ]

        readable_best_columns = [column for column in readable_comparison_columns if column != "rank_in_test"]
        readable_worst_columns = readable_best_columns
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
            "median_energy_drift_fraction_of_max"
        ]        

        # Failure Summary
        failed_summary = summary[summary["status"] == "failed"].copy()
        if not failed_summary.empty:
            save_raw_table(failed_summary, raw_dir / "benchmark_failures.csv")

        # Sucess Summary
        successful_summary = summary[summary["status"] == "success"].copy()
        if successful_summary.empty:
            print("No successful benchmark runs were collected.")
            return

        # Raw Comparison Table
        comparison_columns = [
            "test",
            "mode",
            "integrator",
            "pair_order",
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
            "max_dL_over_L0",
            "final_dL_over_L0",
            "max_dP",
            "final_dP",
            "max_dRcm",
            "final_dRcm",
            "final_com_drift",
            "plot_path",
        ]

        # Comparison Table
        comparison_columns = [column for column in comparison_columns if column in summary.columns]
        comparison_table = summary[comparison_columns].copy()
        comparison_table = comparison_table.sort_values(["test", "status", "max_dE_over_E0", "max_dL_over_L0", "max_dP", "max_dRcm"], na_position="last")
        save_raw_table(comparison_table, raw_dir / "benchmark_comparison_table.csv")

        comparison_readable = add_test_rank(comparison_table, metric="max_dE_over_E0")
        comparison_readable = compact_benchmark_table(comparison_readable, readable_comparison_columns)
        save_readable_table(comparison_readable, readable_dir / "benchmark_comparison_table_readable.csv")

        # Best Runs
        best_rows = (successful_summary.sort_values(["test", "max_dE_over_E0", "max_dL_over_L0", "max_dP", "max_dRcm"]).groupby("test", as_index=False).first())
        save_raw_table(best_rows, raw_dir / "best_by_energy.csv")
        save_readable_table(compact_benchmark_table(best_rows, readable_best_columns), readable_dir / "best_by_energy_readable.csv")
        
        # Worst Runs
        worst_rows = (successful_summary.sort_values("max_dE_over_E0", ascending=False).head(10))
        save_raw_table(worst_rows, raw_dir / "worst_by_energy.csv")
        save_readable_table(compact_benchmark_table(worst_rows, readable_worst_columns), readable_dir / "worst_by_energy_readable.csv")
        
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
            successful_runs = ("status", "count")).sort_values("median_max_dE_over_E0"))
        save_raw_table(mode_rankings, raw_dir / "mode_rankings.csv")
        save_readable_table(compact_benchmark_table(mode_rankings, readable_mode_ranking_columns), readable_dir / "mode_rankings_readable.csv")

    finally:
        if original_param_text is not None:
            DEFAULT_PARAM_PATH.write_text(original_param_text)
        if original_initial_conditions_text is not None:
            DEFAULT_INITIAL_CONDITIONS_PATH.write_text(original_initial_conditions_text)
        restore_optional_file(DEFAULT_OUTPUT_PATH, original_output)
        restore_optional_file(DEFAULT_DIAGNOSTICS_PATH, original_diagnostics)
        print("\nRestored original param.txt, initial_conditions.txt, output.csv, and diagnostics.csv settings.")

# ============================================================
# Clear .csv files
# ============================================================

def clear_simulation_outputs() -> None:
    if DEFAULT_OUTPUT_PATH.exists():
        DEFAULT_OUTPUT_PATH.unlink()
    if DEFAULT_DIAGNOSTICS_PATH.exists():
        DEFAULT_DIAGNOSTICS_PATH.unlink()
