#include <cmath>
#include <iostream>

#include "univ_vari_solve.h"
#include "propagator.h"
#include "vec3.h"

CanonicalStateVector propagate_universal(double mu_grav, double reduced_mass, const Vec3& q0, const Vec3& p0, double dt) {
    Vec3 v0;

    v0 = p0 / reduced_mass;

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

    Vec3 q = (f * q0) + (g * v0);

    const double r_mag = norm(q);
    const double fdot = (std::sqrt(mu_grav) / (r_mag * r0_mag)) * chi * (z * S - 1.0);
    const double gdot = 1.0 - (chi * chi / r_mag) * C;

    Vec3 v = (fdot * q0) + (gdot * v0);
    Vec3 p = reduced_mass * v;

    return{q, p, true, chi_res.iterations};
}