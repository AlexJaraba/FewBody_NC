#include <cmath>
#include <limits>

#include "numerics/newton_solver.h"

NewtonResult newtonSolver(
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
            return {std::numeric_limits<double>::quiet_NaN(), i, false, std::numeric_limits<double>::quiet_NaN()};
        }
        if (std::abs(df_x) < std::numeric_limits<double>::epsilon()) {
            return {x, i, false, std::abs(f_x)};
        }

        const double step = f_x / df_x;
        const double x_new = x - step;
        if (!std::isfinite(step) || !std::isfinite(x_new)) {
            return {x, i, false, std::abs(f_x)};
        }

        const double step_tolerance = abs_tolerance + rel_tolerance * std::max(std::abs(x), std::abs(x_new));
        if (std::abs(step) <= step_tolerance) {
            const double f_new = function(x_new);
            return {x_new, i + 1, std::isfinite(f_new), std::isfinite(f_new) ? std::abs(f_new) : std::numeric_limits<double>::quiet_NaN()};
        }
        x = x_new;
    }

    const double final_residual = function(x);
    return {x, maxIterations, false, std::isfinite(final_residual) ? std::abs(final_residual) : std::numeric_limits<double>::quiet_NaN()};
}