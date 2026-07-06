import argparse
import functions

from pathlib import Path

"""
FewBodyNC plotting command-line entry point.

This script is intentionally small. Most of the logic lives in functions.py

Common commands:
    python python/plot_output.py
    python python/plot_output.py --benchmark
    python python/plot_output.py --use-diagnostics-csv
    python python/plot_output.py --shadow
    python python/plot_output.py --convergence
"""

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

# Define command-line options for plotting, diagnostics, convergence studies, and benchmark generation

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FewBodyNC plotting and verification utility.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite instead of plotting.",)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to output.csv.",)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS_PATH, help="Path to diagnostics.csv.",)
    parser.add_argument("--G", type=float, default=None, help="Gravitational constant used for recomputed diagnostics.",)
    parser.add_argument("--use-diagnostics-csv", action="store_true", help="Use diagnostics.csv for diagnostic plots instead of recomputing from output.csv.",)
    parser.add_argument("--shadow", action="store_true", help="Plot shadow Hamiltonian error from diagnostics.csv.",)
    parser.add_argument("--energy-boundedness", action="store_true", help="Run energy boundedness and drift suite.")
    parser.add_argument("--convergence", action="store_true", help="Run timestep convergence study.",)
    parser.add_argument("--convergence-suite", choices=["current", "cartesian-hernandez"], default="current", help="Choose which convergence suite to run when --convergence is used.")
    parser.add_argument("--adaptive-compare", action="store_true", help="Run Step 10.4 adaptive-on vs adaptive-off comparison.",)
    parser.add_argument("--adaptive-levels", type=int, default=None, help="Timestep levels to use for adaptive comparison.",)
    parser.add_argument("--adaptive-eta", type=float, default=None, help="Eta value to use for adaptive comparison.",)
    parser.add_argument("--pair-order-policy", action="store_true", help="Run pair-order policy suite.")
    parser.add_argument("--rebound-compare", action="store_true", help="Compare FewBodyNC benchmark/convergence results against a REBOUND reference run.")
    parser.add_argument("--rebound-integrator", type=str, default="whfast", help="REBOUND integrator to use for comparison, such as whfast or leapfrog.")
    parser.add_argument("--rebound-move-to-com", action="store_true", help="Call sim.move_to_com() before running REBOUND comparisons.")
    parser.add_argument("--save-rebound-reference-plots", action="store_true", help="Save standalone REBOUND reference plots during benchmark comparison.")

    return parser.parse_args()

# Dispatch command-line modes.
# Only one special mode is run at a time:
# - Benchmark Suite
# - Shadow plot
# - Convergence Study
# - Normal Verification Suite

def main() -> None:
    args = parse_args()
    params = functions.read_param(DEFAULT_PARAM_PATH)

    if args.G is None:
        G = float(params.get("gravitational_constant", 0.000296014912))
    else:
        G = args.G

    config = functions.PlotConfig(G=G)
    
    if args.pair_order_policy:
        functions.run_pair_order_policy_suite(use_diagnostics_csv=args.use_diagnostics_csv)
        return
    if args.benchmark:
        functions.run_benchmark_suite(
            use_diagnostics_csv=args.use_diagnostics_csv,
            rebound_compare=args.rebound_compare,
            rebound_integrator=args.rebound_integrator,
            rebound_move_to_com=args.rebound_move_to_com,
            save_rebound_reference_plots=args.save_rebound_reference_plots)
        return
    if args.shadow:
        functions.plot_shadow_hamiltonian(args.diagnostics)
        return
    if args.energy_boundedness:
        functions.run_energy_boundedness_suite(use_diagnostics_csv=args.use_diagnostics_csv)
        return
    if args.convergence:
        if args.convergence_suite == "cartesian-hernandez":
            functions.run_convergence_suite(use_diagnostics_csv=args.use_diagnostics_csv)
        else:
            functions.run_timestep_scaling_study(use_diagnostics_csv=args.use_diagnostics_csv)
        return
    if args.adaptive_compare:
        functions.run_adaptive_comparison_study(timestep_levels=args.adaptive_levels, timestep_eta=args.adaptive_eta)
        return
    
    output_df = functions.read_output(args.output)
    if args.use_diagnostics_csv:
        diagnostics = functions.read_diagnostics(args.diagnostics)
        required_component_columns = {"linear_momentum_x", "linear_momentum_y", "linear_momentum_z", 
                            "angular_momentum_x", "angular_momentum_y", "angular_momentum_z", 
                            "com_x", "com_y", "com_z"}
        if diagnostics is None:
            print("Falling back to recomputing diagnostics from output.csv.")
            diagnostics = functions.compute_diagnostics_from_output(output_df, config)
        else:
            missing = sorted(required_component_columns - set(diagnostics.columns))
            nonfinite = []
            for column in sorted(required_component_columns & set(diagnostics.columns)):
                values = diagnostics[column]
                if not values.notna().any():
                    nonfinite.append(column)
            if missing or nonfinite:
                print(f"diagnostics.csv is missing usable component diagnostics; missing = {missing}, nonfinite = {nonfinite}. Recomputing from output.csv.")
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
# Use Diagnostics instead of recomputing energy: python python/plot_output.py --use-diagnostics-csv
# Plot Shadow Hamiltonian: python python/plot_output.py --shadow
# Run Convergence Study: python python/plot_output.py --convergence
# Use Different Gravitational Constant: python python/plot_output.py --G 0.000296014912