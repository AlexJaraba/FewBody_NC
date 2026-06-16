import argparse
import functions

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
# Command line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FewBodyNC plotting and verification utility.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite instead of plotting.",)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to output.csv.",)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH, help="Path to diagnostics.csv.",)
    parser.add_argument("--G", type=float, default=0.000296014912, help="Gravitational constant used for recomputed diagnostics.",)
    parser.add_argument("--use-diagnostics-csv", action="store_true", help="Use diagnostics.csv for diagnostic plots instead of recomputing from output.csv.",)
    parser.add_argument("--shadow", action="store_true", help="Plot shadow Hamiltonian error from diagnostics.csv.",)
    parser.add_argument("--convergence", action="store_true", help="Run timestep convergence study.",)

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = functions.PlotConfig(G=args.G)

    if args.benchmark:
        functions.run_benchmark_suite()
        return
    if args.shadow:
        functions.plot_shadow_hamiltonian(args.diagnostics)
        return
    if args.convergence:
        functions.run_timestep_scaling_study()
        return
    output_df = functions.read_output(args.output)
    if args.use_diagnostics_csv:
        diagnostics = functions.read_diagnostics(args.diagnostics)
        output_df = functions.read_output(args.output)
        if diagnostics is None:
            print("Falling back to recomputing diagnostics from output.csv.")
            diagnostics = functions.compute_diagnostics_from_output(output_df, config)
    else:
        diagnostics = functions.compute_diagnostics_from_output(output_df, config)

    functions.plot_verification_suite(output_df, diagnostics, config)

# ====================================================

if __name__ == "__main__":
    main()

# === DIFFERENT ARGUMENTS TO RUN ===
# Regular Plot Output: python python/plot_output.py
# Run Benchmark Test: python python/plot_output.py --benchmark
# Use Diagnostics instead of recomputing enegry: python python/plot_output.py --use-diagnostics-csv
# Plot Shadow Hamiltonian: python python/plot_output.py --shadow
# Run Convergence Study: python python/plot_output.py --convergence
# Use Different Graviational Constant: python python/plot_output.py --G 0.000296014912