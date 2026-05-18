#pragma once

#include <functional>

struct NewtonResult {
    double root;
    int iterations;
    bool converged;
};

NewtonResult Newton_Solver(
    std::function<double(double)> function,
    std::function<double(double)> d_function,
    double x0,
    double tolerance,
    int maxIterations
);