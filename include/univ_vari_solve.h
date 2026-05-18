#pragma once

#include <vector>

struct ChiResult {
    double chi;
    int iterations;
    bool converged;
};

double stumpff_C(double z);
double stumpff_S(double z);
double norm(const std::vector<double>& v);

ChiResult solve_chi(double mu, double alpha,
                    const std::vector<double>& r0,
                    double vr, double dt,
                    double tol = 1e-8,
                    int max_iter = 100);