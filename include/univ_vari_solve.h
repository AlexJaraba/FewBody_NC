#pragma once

#include "vec3.h"

struct ChiResult {
    double chi;
    int iterations;
    bool converged;
};

double stumpff_C(double z);
double stumpff_S(double z);
double norm(const Vec3& v);

ChiResult solve_chi(double mu, double alpha,
                    const Vec3& r0,
                    double vr, double dt,
                    double abs_tol = 1e-14,
                    double rel_tol = 1e-13,
                    int max_iter = 100);