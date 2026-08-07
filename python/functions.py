import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import subprocess
import re

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
# Convergence Cases
# ============================================================

CONVERGENCE_TESTS = [
    {
        "name": "D1_BinaryShort",
        "dt_ref": 0.00625,
        "dts": (0.1, 0.05, 0.025, 0.0125),
        "runtime": 1000,
        "output_frequency": 100,
        "initial_conditions": [
            (1.0, -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,  0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
        ],
    },
    {
        "name": "D2_HierarchicalTripleShort",
        "dt_ref": 0.00125,
        "dts": (0.02, 0.01, 0.005, 0.0025),
        "runtime": 1000,
        "output_frequency": 100,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            (1e-3,  5.0, 0.0, 0.0, 0.0,  0.01089,  0.0),
        ],
    },
    {
        "name": "D3_Figure8Short",
        "dt_ref": 0.00025,
        "dts": (0.004, 0.002, 0.001, 0.0005),
        "runtime": 20,
        "output_frequency": 10,
        "initial_conditions": [
            (1.0, -0.97000436,  0.24308753, 0.0,  0.008019029,  0.007436436, 0.0),
            (1.0,  0.97000436, -0.24308753, 0.0,  0.008019029,  0.007436436, 0.0),
            (1.0,  0.0,         0.0,        0.0, -0.016038058, -0.014872872, 0.0),
        ],
    },
    {
        "name": "D4_CloseBinaryPerturberShort",
        "dt_ref": 0.000625,
        "dts": (0.01, 0.005, 0.0025, 0.00125),
        "runtime": 200,
        "output_frequency": 50,
        "initial_conditions": [
            (1.0,  -0.5, 0.0, 0.0,  0.0,    -0.012166, 0.0),
            (1.0,   0.5, 0.0, 0.0,  0.0,     0.012166, 0.0),
            (0.01,  1.2, 0.2, 0.0, -0.0020,  0.0040,   0.0),
        ],
    },
]

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
    {
        "name": "Test9_HernandezTest1",
        "dt": 0.0008718353300301555,
        "runtime": 81.3712974694812,
        "output_frequency": 20,
        "initial_conditions": [
            (0.5, -0.0595, 0.0, 0.0, 0.0, -0.0582073219045089, 0.0),
            (0.5, -0.0405, 0.0, 0.0, 0.0, -0.0187361524126806, 0.0),
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
# Pair-Order Policy Benchmark Tests
# ============================================================

PAIR_ORDER_POLICY_MODES = [
    {
        "name": "pair_order_canonical_fixed",
        "integrator": "hernandez",
        "pair_order": "canonical",
        "pair_order_policy_status": "production_default",
        "expected_effective_pair_order": "canonical",
    },
    {
        "name": "pair_order_strength_fixed",
        "integrator": "hernandez",
        "pair_order": "strength",
        "pair_order_policy_status": "optional_fixed_step_diagnostic",
        "expected_effective_pair_order": "strength",
    },
    {
        "name": "pair_order_auto_fixed",
        "integrator": "hernandez",
        "pair_order": "auto",
        "pair_order_policy_status": "conservative_auto_expected_canonical",
        "expected_effective_pair_order": "canonical",
    },
]

PAIR_ORDER_POLICY_TESTS = [
    BENCHMARK_TESTS[0],  # Binary
    BENCHMARK_TESTS[2],  # Stronger perturbed triple
    BENCHMARK_TESTS[3],  # Scattering escape
    BENCHMARK_TESTS[5],  # Close encounter
]

# ============================================================
# Energy Boundedness Tests
# ============================================================

ENERGY_BOUNDEDNESS_MODES = [
    {
        "name": "hernandez_canonical_fixed",
        "integrator": "hernandez",
        "pair_order": "canonical",
    },
]

ENERGY_BOUNDEDNESS_CASES = [
    {
        "name": "E1_LongBinary_CartesianHernandez",
        "description": "Long isolated binary boundedness test for Cartesian Hernandez canonical fixed.",
        "modes": [ENERGY_BOUNDEDNESS_MODES[0]],
        "test": {
            "name": "E1_LongBinary",
            "dt": 0.1,
            "runtime": 100000,
            "output_frequency": 1000,
            "initial_conditions": [
                (1.0, -0.5, 0.0, 0.0, 0.0, -0.012166, 0.0),
                (1.0,  0.5, 0.0, 0.0, 0.0,  0.012166, 0.0),
            ],
        },        
    },
    {
        "name": "E2_FullSolarSystem_Secular",
        "description": "Long solar-system secular boundedness comparison.",
        "modes": ENERGY_BOUNDEDNESS_MODES,
        "test": {
            "name": "E2_FullSolarSystem",
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
    },
    {
        "name": "E3_InnerPlanets_Secular",
        "description": "Inner-planets boundedness comparison.",
        "modes": ENERGY_BOUNDEDNESS_MODES,
        "test": {
            "name": "E3_InnerPlanets",
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
    }
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
    if "shadow_energy" not in df.columns:
        df["shadow_energy"] = df["total_energy"]
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
    
    diagnostics = df.sort_values(by="time").reset_index(drop=True)
    diagnostics = add_diagnostics_error_columns(diagnostics, PlotConfig())
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
    total_steps = int(np.ceil(runtime / dt))
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
                         rebound_comparison: dict | None = None,
                         config: PlotConfig | None = None,
                         param_snapshot: str = "",
                         initial_conditions_snapshot: str = "",
                         failure_details: dict | None = None) -> None:
    
    if config is None:
        config = PlotConfig()
    body_count = len(test.get("initial_conditions", []))
    row = {"run_number": run_number,
           "engine": engine,
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
           "final_dRcm": np.nan,
           "rebound_compare": False,
           "rebound_integrator": "",
           "rebound_status": "not_requested",
           "rebound_error": ""}
    
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
        row.update(energy_boundedness_summary(diagnostics, config))
    if rebound_comparison is not None:
        row.update(rebound_comparison)
    
    benchmark_rows.append(row)

def _safe_ratio(numerator: float, denominator: float, epsilon:float = 1e-300) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= epsilon:
        return float("nan")
    return float(numerator / denominator)

def finite_log2_ratio(coarse_error: float, fine_error: float) -> float:
    if not np.isfinite(coarse_error) or not np.isfinite(fine_error):
        return float("nan")
    if coarse_error <= 0.0 or fine_error <= 0.0:
        return float("nan")
    return float(np.log2(coarse_error / fine_error))

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

def tail_text(text: str | None, max_chars: int = 4000) -> str:
    if text is None:
        return ""
    return str(text)[-max_chars:]

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

def parse_bool_text(value, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}

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

def classify_energy_boundedness(max_abs_dE: float, final_abs_dE: float, drift_over_run: float, drift_fraction_of_max: float) -> str:
    if not (np.isfinite(max_abs_dE) and np.isfinite(final_abs_dE) and np.isfinite(drift_over_run) and np.isfinite(drift_fraction_of_max)):
        return "undetermined"
    if max_abs_dE <= 1e-14:
        return "bounded_like_machine_precision"
    if drift_fraction_of_max <= 0.25 and final_abs_dE <= 0.75 * max_abs_dE:
        return "bounded_like"
    return "mixed_unclear"

def energy_boundedness_summary(diagnostics: pd.DataFrame, config: PlotConfig) -> dict:
    if diagnostics is None or diagnostics.empty:
        return {"energy_drift_slope": float("nan"), 
                "energy_abs_drift_slope": float("nan"), 
                "energy_drift_over_run": float("nan"), 
                "energy_drift_fraction_of_max": float("nan"), 
                "energy_rms_dE_over_E0": float("nan"),
                "energy_boundedness_class": "undetermined"}
    
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

    if finite_abs_dE.size:
        rms_dE = float(np.sqrt(np.mean(finite_abs_dE * finite_abs_dE)))
    else:
        rms_dE = float("nan")
    
    return {"energy_drift_slope": energy_drift_slope, 
            "energy_abs_drift_slope": energy_abs_drift_slope, 
            "energy_drift_over_run": drift_over_run,
            "energy_drift_fraction_of_max": drift_fraction_of_max,
            "energy_rms_dE_over_E0": rms_dE,
            "energy_boundedness_class": classify_energy_boundedness(max_abs_dE=max_abs_dE, final_abs_dE=final_abs_dE, drift_over_run=drift_over_run, drift_fraction_of_max=drift_fraction_of_max)}

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
                   "com_vx", "com_vy", "com_vz",
                   "shadow_energy", "total_energy"]:
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

    nine_columns = ["dLx", "dLy", "dLz", "dPx", "dPy", "dPz", "dCcm_x", "dCcm_y", "dCcm_z",]
    nine = np.abs(df[nine_columns].to_numpy(dtype=float))
    df["nine_integral_of_motion_error_max"] = np.nanmax(nine, axis=1)

    if "dShadow_over_Shadow0" not in df.columns:
        df["dShadow_over_Shadow0"] = error_relative(df["shadow_energy"].to_numpy(dtype=float), config.epsilon)
    
    return df

def thin_for_plotting(time: np.ndarray, values: np.ndarray, max_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(time) <= max_points:
        return time, values
    
    indices = np.linspace(0, len(time) - 1, max_points).astype(int)
    return time[indices], values[indices]

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

    if "shadow_energy" not in diagnostics.columns:
        diagnostics["shadow_energy"] = diagnostics["total_energy"]
    if "timestep" not in diagnostics.columns:
        times = diagnostics["time"].to_numpy(dtype=float)
        if len(times) >= 2:
            dt = np.diff(times, prepend=times[0])
            if len(dt) > 1:
                dt[0] = dt[1]
            diagnostics["timestep"] = dt
        else:
            diagnostics["timestep"] = np.nan

    diagnostics = add_diagnostics_error_columns(diagnostics, config)
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
    plot_error(ax_energy, time, dE, "Relative Energy Error", r"$|\Delta E/E|$", floor=1e-20, ymin_fixed=1e-11)

    ax_angular = fig.add_subplot(gs[0, 3])
    plot_error(ax_angular, time, dL, "Angular Momentum Error", r"$|\Delta L|$", floor=1e-20, ymin_fixed=1e-20)

    ax_px = fig.add_subplot(gs[1, 2])
    plot_error(ax_px, time, dPcm_x, "COM Momentum X Error", r"$|\Delta p_{\rm cm,x}|$", floor=1e-20, ymin_fixed=1e-20)

    ax_py = fig.add_subplot(gs[1, 3])
    plot_error(ax_py, time, dPcm_y, "COM Momentum Y Error", r"$|\Delta p_{\rm cm,y}|$", floor=1e-20, ymin_fixed=1e-20)

    ax_xcm = fig.add_subplot(gs[2, 2])
    plot_error(ax_xcm, time, dXcm, "COM Drift X Error", r"$|\Delta x_{\rm cm}|$", floor=1e-20, ymin_fixed=1e-20)

    ax_ycm = fig.add_subplot(gs[2, 3])
    plot_error(ax_ycm, time, dYcm, "COM Drift Y Error", r"$|\Delta y_{\rm cm}|$", floor=1e-20, ymin_fixed=1e-20)

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

"""
Write initial conditions in the exact format expected by the C++ reader:
    mass x y z vx vy vz
No header row is written
"""
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
                         pair_order: str,
                         G: float,
                         use_diagnostics_csv: bool = True) -> pd.DataFrame:
    config = PlotConfig(G=G)
    rows = []

    print("\n" + "-" * 90)
    print(f"Convergence case: {case_name}")
    print(f"integrator           = {integrator}")
    print(f"pair_order           = {pair_order}")
    print(f"runtime              = {runtime}")
    print(f"output_frequency     = {output_frequency}")
    print(f"G                    = {G}")
    print(f"reference dt         = {dt_ref}")
    print(f"dt ladder            = {dts}")
    print("-" * 90)

    rewrite_initial_conditions(initial_conditions)

    print(f"\nRunning reference solutions: dt = {dt_ref}")
    rewrite_param(dt=dt_ref, runtime=runtime, output_frequency=output_frequency, integrator=integrator, G=G, pair_order=pair_order)
    clear_simulation_outputs()
    run_executable()
    reference_output = read_output(DEFAULT_OUTPUT_PATH)
    reference_positions = compute_final_positions(reference_output)

    for dt in dts:
        print(f"\nRunning {case_name}: dt = {dt}")
        rewrite_param(dt=dt, runtime=runtime, output_frequency=output_frequency, integrator=integrator, G=G, pair_order=pair_order)
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
                "pair_order": pair_order,
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
                "pair_order": pair_order,
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

def run_timestep_scaling_study(dt_ref: float = 0.00025,  
                               dts: tuple = (0.01, 0.005, 0.0025, 0.00125),  
                               param_path: Path = DEFAULT_PARAM_PATH,
                               use_diagnostics_csv: bool = True,
                               output_dir: Path = DEFAULT_BENCHMARK_PLOT_DIR / "convergence") -> pd.DataFrame:
    params = read_param(param_path)

    runtime = float(params.get("runtime", 1.0))
    output_frequency = int(params.get("output_frequency", 10))
    integrator = params.get("integrator", "hernandez")
    pair_order = params.get("pair_order", "canonical")
    G = float(params.get("gravitational_constant", 0.000296014912))

    initial_conditions = read_initial_conditions(DEFAULT_INITIAL_CONDITIONS_PATH)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    readable_dir = output_dir / "readable"
    readable_dir.mkdir(parents=True, exist_ok=True)

    original_param_text = param_path.read_text()
    original_initial_conditions_text = (DEFAULT_INITIAL_CONDITIONS_PATH.read_text() if DEFAULT_INITIAL_CONDITIONS_PATH.exists() else None)

    convergence_readable_columns = [
        "case",
        "dt",
        "dt_ref",
        "runtime",
        "integrator",
        "pair_order",
        "rms_final_position_error",
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

        results = run_convergence_case(case_name="CurrentParamScalingStudy",
                                       initial_conditions=initial_conditions,
                                       dt_ref=dt_ref,
                                       dts=dts,
                                       runtime=runtime,
                                       output_frequency=output_frequency,
                                       integrator=integrator,
                                       pair_order=pair_order,
                                       G=G,
                                       use_diagnostics_csv=use_diagnostics_csv)
        
        save_raw_table(results, raw_dir / "current_param_scaling_study.csv")
        readable = compact_benchmark_table(results, convergence_readable_columns)
        save_readable_table(readable, readable_dir / "current_param_scaling_study_readable.csv")

        dts_array = results["dt"].to_numpy(dtype=float)
        errors_array = results["rms_final_position_error"].to_numpy(dtype=float)

        plt.figure(figsize=(7, 5), constrained_layout=True)
        plt.loglog(dts_array, errors_array, marker="o")
        plt.gca().invert_xaxis()
        plt.xlabel("Timestep dt")
        plt.ylabel(f"Timestep convergence: {integrator}")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.show()

        return results
    
    finally:
        param_path.write_text(original_param_text)

        if original_initial_conditions_text is not None:
            DEFAULT_INITIAL_CONDITIONS_PATH.write_text(original_initial_conditions_text)
        
        print("\nRestored original param.txt and initial_conditions.txt settings")

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

    summary_columns = [
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
        print("HERNANDEZ CONVERGENCE SUITE")
        print("=" * 90)
        print("Mode:")
        print("  integrator           = hernandez")
        print("  pair_order           = canonical")
        print("  G                    = 0.000296014912")
        for test in tests:
            all_results.append(run_convergence_case(
                case_name=test["name"],
                initial_conditions=test["initial_conditions"],
                dt_ref=test["dt_ref"],
                dts=test["dts"],
                runtime=test["runtime"],
                output_frequency=test["output_frequency"],
                integrator="hernandez",
                pair_order="canonical",
                G=0.000296014912,
                use_diagnostics_csv=use_diagnostics_csv))
                
        results = pd.concat(all_results, ignore_index=True)
        save_raw_table(results, raw_dir / "convergence.csv")
        readable_columns = [
            "case",
            "dt",
            "dt_ref",
            "runtime",
            "integrator",
            "pair_order",
            "rms_final_position_error",
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
        save_readable_table(compact_benchmark_table(results, readable_columns), readable_dir / "convergence_readable.csv")

        successful = results[results["status"] == "success"].copy()
        if not successful.empty:
            summary = (successful.groupby("case", as_index=False).agg(
                median_observed_order=("observed_order_to_next_finer", "median"),
                min_observed_order=("observed_order_to_next_finer", "min"),
                max_observed_order=("observed_order_to_next_finer", "max"),
                max_rms_final_position_error=("rms_final_position_error", "max"),
                max_dE_over_E0=("max_dE_over_E0", "max"),
                successful_runs=("status", "count")))
            save_raw_table(summary, raw_dir / "convergence_summary.csv")
            save_readable_table(compact_benchmark_table(summary, summary_columns), readable_dir / "convergence_summary_readable.csv")
        
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
                            G = config.G,
                            pair_order = mode.get("pair_order", "canonical"))
                
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
            "rebound_status",
            "ratio_max_dE_over_E0_to_rebound",
            "plot_path",
        ]

        readable_best_columns = [
            "test",
            "mode",
            "integrator",
            "pair_order",
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
            "pair_order",
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
                                                    "pair_order",
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
            bounded_like_runs = ("energy_boundedness_class", lambda values: int(values.astype(str).str.contains("bounded_like", na=False).sum())),
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
                    G=config.G,
                    pair_order=mode.get("pair_order", "canonical"))
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
        "pair_order",
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
                              G=config.G,
                              pair_order=mode.get("pair_order", "canonical"))
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
