#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>

#include "numerics/newton_solver.h"
#include "numerics/univ_vari_solve.h"
#include "math/vec3.h"

/* =============================================================================

    Universal anomaly solver

    This file solves for the universal anomaly chi used by the Kepler propagator.
    The Stumpff functions C(z) and S(z) allow the same formulation to work for elliptic, near-parabolic, and hyperbolic cases.
    Kepler solver failures usually indicate an overly large timestep, a near collision, or an orbit outside the safe range of the current Newton sovler.

   =============================================================================*/

/*
    Stumpff functions used by universal-variable Kepler propagation
    These functions switch formulas depending on the sign of z to avoid using elliptic expressions for hyperbolic motion and vice versa
*/

double stumpff_C(double z) {
    const double abs_z = std::abs(z);

    //Series expansion near z = 0
    if (abs_z < 1e-12) {
        const double z2 = z * z;
        const double z3 = z2 * z;
        return 0.5 - (z / 24.0) + (z2 / 720.0) - (z3 / 40320.0);
    }
    if (z > 0) {
        const double sz =  std::sqrt(z);
        return (1.0 - std::cos(sz)) / z;
    }
    const double sz = std::sqrt(-z);
    return (std::cosh(sz) - 1.0) / (-z);
}

double stumpff_S(double z) {
    const double abs_z = std::abs(z);

    // Series expansion near z = 0
    if (abs_z < 1e-12) {
        const double z2 = z * z;
        const double z3 = z2 * z;
        return (1.0 / 6.0) - (z / 120.0) + (z2 / 5040.0) - (z3 / 362880.0);
    }
    if (z > 0) {
        const double sz = std::sqrt(z);
        return (sz - std::sin(sz)) / (sz * sz * sz);
    }
    const double sz = std::sqrt(-z);
    return (std::sinh(sz) - sz) / (sz * sz * sz);
}

/*
    Solve the universal Kepler equation for chi using Newton iteration.
    The returned chi is then used to compute the Lagrange f and g functions in the propagator.
*/

ChiResult solve_chi(double mu, double alpha, const Vec3& r0, double vr, double dt, double abs_tol, double rel_tol, int max_iter) {
    const double r = norm(r0);
    if (!std::isfinite(mu) || !std::isfinite(alpha) || !std::isfinite(r) || !std::isfinite(vr) || !std::isfinite(dt) || mu <= 0.0 || r <= 0.0) {
        return {0.0, 0, false};
    }
    if (dt == 0.0) {
        return {0.0, 0, true};
    }

    const double sqrt_mu = std::sqrt(mu);
    double chi0;

    if (alpha > 1e-12) {
        chi0 = sqrt_mu * alpha * dt;
    }
    else if (alpha < -1e-12) {
        const double a = 1.0 / alpha;
        const double numerator = -2.0 * mu * alpha * dt;
        const double denominator = r * vr + std::copysign(1.0, dt) * std::sqrt(-mu * a) * (1.0 - r * alpha);
        const double fraction = numerator / denominator;

        if (std::abs(denominator) < 1e-15 || !std::isfinite(fraction) || fraction <= 0.0) {
            chi0 = sqrt_mu * dt / r;
        }
        else {
            chi0 = std::copysign(std::sqrt(-a) * std::log(fraction), dt);
        }
    }
    else {
        chi0 = sqrt_mu * dt / r;
    }

    if (!std::isfinite(chi0)) {
        chi0 = sqrt_mu * dt / r;
    }

    auto F = [&](double chi) {
        const double z = alpha * chi * chi;
        const double C = stumpff_C(z);
        const double S = stumpff_S(z);
        if (!std::isfinite(z) || !std::isfinite(C) || !std::isfinite(S)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return (r * vr / sqrt_mu) * chi * chi * C + (1.0 - r * alpha) * chi * chi * chi * S + r * chi - sqrt_mu * dt;
    };

    auto dF = [&](double chi) {
        const double z = alpha * chi * chi;
        const double C = stumpff_C(z);
        const double S = stumpff_S(z);
        if (!std::isfinite(z) || !std::isfinite(C) || !std::isfinite(S)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return (r * vr / sqrt_mu) * chi * (1.0 - z * S) + (1.0 - r * alpha) * chi * chi * C + r;
    };

    /*
        Fast path: use the reusable Newton solver first.
        If it converges with a residual below tolerance, return immediately.
    */
    NewtonResult newton_result = Newton_Solver(F, dF, chi0, abs_tol, rel_tol, max_iter);

    if (newton_result.converged && std::isfinite(newton_result.root)) {
        const double residual = std::abs(F(newton_result.root));
        const double tolerance = abs_tol + rel_tol * std::max(std::abs(newton_result.root), 1.0);
        if (std::isfinite(residual) && residual <= tolerance) {
            return {newton_result.root, newton_result.iterations, true};
        }
    }

    /*
        Safe fallback: bracketed Newton/bisection.
        The Newton step is used when it remains inside the bracket;
        otherwise the method falls back to bisection.
    */
    const double direction = (dt > 0.0) ? 1.0 : -1.0;

    auto signed_F = [&](double y) {
        return direction * F(direction * y);
    };
    auto signed_dF = [&](double y) {
        return dF(direction * y);
    };

    double lower = 0.0;
    double upper = std::max(std::abs(chi0), sqrt_mu * std::abs(dt) / r);
    upper = std::max(upper, 1e-14);

    const double f_lower = signed_F(lower);
    double f_upper = signed_F(upper);

    if (!std::isfinite(f_lower)) {
        return {newton_result.root, newton_result.iterations, false};
    }

    int bracket_iterations = 0;

    while (
        (!std::isfinite(f_upper) || f_upper < 0.0) && bracket_iterations < 100) {
        upper *= 2.0;
        f_upper = signed_F(upper);
        ++bracket_iterations;
    }

    if (!std::isfinite(f_upper) || f_upper < 0.0) {
        return { newton_result.root, newton_result.iterations + bracket_iterations, false};
    }

    double y = std::abs(chi0);

    if (!std::isfinite(y) || y <= lower || y >= upper) {
        y = 0.5 * (lower + upper);
    }

    double best_y = y;
    double best_residual = std::numeric_limits<double>::infinity();

    for (int i = 0; i < max_iter; ++i) {
        const double f_y = signed_F(y);
        if (!std::isfinite(f_y)) {
            y = 0.5 * (lower + upper);
            continue;
        }
        const double residual = std::abs(f_y);
        if (residual < best_residual) {
            best_residual = residual;
            best_y = y;
        }
        const double tolerance =
            abs_tol + rel_tol * std::max(std::abs(y), 1.0);
        if (residual <= tolerance) {
            return { direction * y, newton_result.iterations + bracket_iterations + i + 1, true};
        }
        if (f_y < 0.0) {
            lower = y;
        }
        else {
            upper = y;
        }
        const double df_y = signed_dF(y);
        double y_new = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(df_y) && std::abs(df_y) > 1e-15) {
            y_new = y - f_y / df_y;
        }
        if (!std::isfinite(y_new) || y_new <= lower || y_new >= upper) {
            y_new = 0.5 * (lower + upper);
        }
        y = y_new;
    }

    return {direction * best_y, newton_result.iterations + bracket_iterations + max_iter, false};
}
