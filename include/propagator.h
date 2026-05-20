#pragma once

#include <vector>

struct CanonicalStateVector {
    std::vector<double> q;
    std::vector<double> p;
    bool converged;
    int iterations;
};

CanonicalStateVector propagate_universal(
    double mu_grav,
    double reduced_mass,
    const std::vector<double>& q0,
    const std::vector<double>& p0,
    double dt
);