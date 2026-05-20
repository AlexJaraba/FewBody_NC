#include <cmath>
#include <vector>
#include <iostream>

#include "univ_vari_solve.h"
#include "propagator.h"

// dot product
static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}

// scalar * vector
static std::vector<double> scalar_mult(double s, const std::vector<double>& v) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        result[i] = s * v[i];
    return result;
}

// vector addition
static std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    return result;
}

CanonicalStateVector propagate_universal(double mu_grav, double reduced_mass, const std::vector<double>& q0, const std::vector<double>& p0, double dt) {
    std::vector<double> v0(3);

    for (int k = 0; k < 3; ++k) {
        v0[k] = p0[k] / reduced_mass;
    }

    const double r0_mag = norm(q0);
    const double v0_mag = norm(v0);
    const double vr0 = dot(q0, v0) / r0_mag;
    const double alpha = 2.0 / r0_mag - (v0_mag * v0_mag) / mu_grav;

    ChiResult chi_res = solve_chi(mu_grav, alpha, q0, vr0, dt);

    if(!chi_res.converged) {
        return {{}, {}, false, chi_res.iterations};
    }

    const double chi = chi_res.chi;
    const double z = alpha * chi * chi;
    const double C = stumpff_C(z);
    const double S = stumpff_S(z);
    const double f = 1.0 - (chi * chi / r0_mag) * C;
    const double g = dt - (chi * chi * chi * S) / std::sqrt(mu_grav);

    std::vector<double> q = add(scalar_mult(f, q0), scalar_mult(g, v0));

    const double r_mag = norm(q);
    const double fdot = (std::sqrt(mu_grav) / (r_mag * r0_mag)) * chi * (z * S - 1.0);
    const double gdot = 1.0 - (chi * chi / r_mag) * C;

    std::vector<double> v = add(scalar_mult(fdot, q0), scalar_mult(gdot, v0));

    std::vector<double> p(3);
    for (int k = 0; k < 3; ++k) {
        p[k] = reduced_mass * v[k];
    }
    return{q, p, true, chi_res.iterations};
}