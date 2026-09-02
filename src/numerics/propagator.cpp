#include <cmath>

#include "numerics/univ_vari_solve.h"
#include "numerics/propagator.h"
#include "math/vec3.h"

/* =================================================================

    Universal-variable Kepler propagator

    This file advances one relative Kepler orbit over a timestep dt.
    The Universal-variable method works for elliptic, parabolic, and hyperbolic motion when the universal anomaly solve converges.
    Inputs are relative position q, relative momentum p, reduced mass mu, and gravitational parameter G * M.

   ================================================================= */

/*
    Propagate a physical pair relative coordinate through the exact two-body Kepler map.
    The canonical momentum p is converted to relative velocity using:
        v = p / mu
    After propagation the updated velocity is converted back to canonical momentum:
        p = mu * v
*/

KeplerPropagationResult propagateUniversal(double mu_grav, double reduced_mass, const Vec3& q0, const Vec3& p0, double dt) {
    if (reduced_mass <= 0.0 || mu_grav <= 0.0 || q0.norm() < 1e-14) {
        return {{}, {}, false, 0};
    }

    const Vec3 v0 = p0 / reduced_mass;
    const double r0_mag = norm(q0);
    const double v0_mag = norm(p0) / reduced_mass;
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

    Vec3 q = (f * q0) + (g / reduced_mass) * p0;

    const double r_mag = norm(q);
    if (!q.is_finite() || !std::isfinite(r_mag) || r_mag <= 0.0) {
        return {{}, {}, false, chi_res.iterations};
    }
    double fdot = (std::sqrt(mu_grav) / (r_mag * r0_mag)) * chi * (z * S - 1.0);
    double gdot = 1.0 - (chi * chi / r_mag) * C;

    const double f_tolerance = 16.0 * std::numeric_limits<double>::epsilon();
    const double g_tolerance = 16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(dt));
    const double fdot_identity = (std::abs(g) > g_tolerance) ? (f * gdot - 1.0) / g : fdot;
    const double gdot_identity = (std::abs(f) > f_tolerance) ? (g * fdot + 1.0) / f : gdot;
    const double fdot_vel_correction = std::abs(fdot_identity - fdot) * r0_mag;
    const double gdot_vel_correction = std::abs(gdot_identity - gdot) * v0_mag;
    if (std::isfinite(gdot_identity) && (!std::isfinite(fdot_identity) || gdot_vel_correction <= fdot_vel_correction)) {
        gdot = gdot_identity;
    } else if (std::isfinite(fdot_identity)) {
        fdot = fdot_identity;
    }

    Vec3 p = (reduced_mass * fdot * q0) + (gdot * p0);
    if (!p.is_finite()) {
        return {{}, {}, false, chi_res.iterations};
    }

    return{q, p, true, chi_res.iterations};
}