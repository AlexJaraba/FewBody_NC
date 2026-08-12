import argparse
import functions

"""
FewBodyNC plotting command-line entry point.

This script is intentionally small. Most of the logic lives in functions.py

Common commands:
    python python/plot_output.py
    python python/plot_output.py --benchmark
    python python/plot_output.py --convergence
"""

# ============================================================
# Command line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FewBodyNC plotting and verification utility.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", action="store_true", help="Run benchmark suite instead of plotting.",)
    mode.add_argument("--convergence", action="store_true", help="Run timestep scaling study using the current parameters.",)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.benchmark:
        functions.run_benchmark_suite()
        return
    if args.convergence:
        functions.run_timestep_scaling_study()
        return

    params = functions.read_param(functions.DEFAULT_PARAM_PATH)
    config = functions.PlotConfig(G=params.get("gravitational_constant", 0.000296014912))
    output = functions.read_output(functions.DEFAULT_OUTPUT_PATH)
    diagnostics = functions.load_diagnostics(output, config)
    functions.plot_verification_suite(output, diagnostics, config)

# ====================================================

if __name__ == "__main__":
    main()