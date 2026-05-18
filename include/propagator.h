#pragma once

#include <vector>

struct StateVector {
    std::vector<double> r;
    std::vector<double> v;
    bool converged;
    int iterations;
};

StateVector propagate_universal(
    double mu,
    const std::vector<double>& r0,
    const std::vector<double>& v0,
    double dt
);