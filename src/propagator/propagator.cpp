#include <cmath>
#include <vector>
#include <iostream>

#include "../numerics/univ_vari_solve.h"
#include "propagator.h"

// dot product
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

// scalar * vector
std::vector<double> scalar_mult(double s, const std::vector<double>& v) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        result[i] = s * v[i];
    return result;
}

// vector addition
std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    return result;
}

StateVector propagate_universal(
    double mu,
    const std::vector<double>& r0,
    const std::vector<double>& v0,
    double dt)
{
    double r0_mag = norm(r0);
    double v0_mag = norm(v0);
    double vr0 = dot(r0, v0) / r0_mag;
    double alpha = 2.0 / r0_mag - (v0_mag * v0_mag) / mu;

    ChiResult chi_res = solve_chi(mu, alpha, r0, vr0, dt);

    if (!chi_res.converged) {
        return {{}, {}, false, chi_res.iterations};
    }

    double chi = chi_res.chi;
    double z = alpha * chi * chi;
    double C = stumpff_C(z);
    double S = stumpff_S(z);
    double f = 1 - (chi * chi / r0_mag) * C;
    double g = dt - (1.0 / std::sqrt(mu)) * chi * chi * chi * S;

    std::vector<double> r =
        add(scalar_mult(f, r0),
            scalar_mult(g, v0));

    double r_mag = norm(r);
    double fdot = (std::sqrt(mu) / (r_mag * r0_mag)) * chi * (z * S - 1);
    double gdot = 1 - (chi * chi / r_mag) * C;

    std::vector<double> v =
        add(scalar_mult(fdot, r0),
            scalar_mult(gdot, v0));

    return {r, v, true, chi_res.iterations};
}