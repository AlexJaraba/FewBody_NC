#include <cmath>
#include <limits>
#include <iostream>
#include "newton_solver.h"

NewtonResult Newton_Solver(
    std::function<double(double)> function,
    std::function<double(double)> d_function,
    double x0,
    double tolerance,
    int maxIterations)
{
    double x = x0;

    for (int i = 0; i < maxIterations; ++i) {
        double f_x = function(x);
        double df_x = d_function(x);

        if (std::abs(df_x) < 1e-12) {
            return {std::numeric_limits<double>::quiet_NaN(), i, false};
        }

        double step = f_x / df_x;
        double x_new = x - step;

        // damping
        // if (std::abs(step) > 1.0) {
        //     x_new = x - 0.5 * step;
        // }

        if (!std::isfinite(x_new)) {
            return {std::numeric_limits<double>::quiet_NaN(), i, false};
        }

        if (std::abs(x_new - x) < tolerance) {
            return {x_new, i + 1, true};
        }

        x = x_new;
    }

    return {std::numeric_limits<double>::quiet_NaN(), maxIterations, false};
}