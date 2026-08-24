#pragma once

#include <functional>

struct NewtonResult {
    double root;
    int iterations;
    bool converged;
    double residual;
};

NewtonResult newtonSolver(
    std::function<double(double)> function,
    std::function<double(double)> d_function,
    double x0,
    double abs_tolerance,
    double rel_tolerance,
    int maxIterations
);