#include <cmath>
#include <limits>
#include <iostream>

#include "numerics/newton_solver.h"

NewtonResult Newton_Solver(
    std::function<double(double)> function,
    std::function<double(double)> d_function,
    double x0,
    double abs_tolerance,
    double rel_tolerance,
    int maxIterations)
{
    double x = x0;

    for (int i = 0; i < maxIterations; ++i) {
        const double f_x = function(x);
        const double df_x = d_function(x);

        if (!std::isfinite(f_x) || !std::isfinite(df_x)) {
            return {
                std::numeric_limits<double>::quiet_NaN(), i, false, std::numeric_limits<double>::quiet_NaN()
            };
        }

        if (std::abs(df_x) < 1e-15) {
            return {
                std::numeric_limits<double>::quiet_NaN(), i, false, std::abs(f_x)
            };
        }

        double step = f_x / df_x;

        // Safeguarded Newton Damping
        const double max_step = 1.0;
        if (std::abs(step) > max_step) {
            step = std::copysign(max_step, step);
        }

        const double x_new = x -step;

        if (!std::isfinite(x_new)) {
            return {
                std::numeric_limits<double>::quiet_NaN(), i, false, std::abs(f_x)
            };
        }

        // Relative scaled Tolerance
        const double scaled_tol = abs_tolerance + rel_tolerance * std::max(std::abs(x_new), 1.0);

        // Dual Convergence Criteria
        const bool step_converged = std::abs(step) < scaled_tol;
        const bool residual_converged = std::abs(f_x) < scaled_tol;

        if (step_converged && residual_converged) {
            return {
                x_new, i + 1, true, std::abs(f_x)
            };
        }
        x = x_new;
    }
    return {std::numeric_limits<double>::quiet_NaN(), maxIterations, false, std::numeric_limits<double>::quiet_NaN()};
}