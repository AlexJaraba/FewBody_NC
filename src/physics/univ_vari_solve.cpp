#include <cmath>
#include <limits>
#include <stdexcept>

#include "newton_solver.h"
#include "univ_vari_solve.h"
#include "vec3.h"

double stumpff_C(double z) {
    if (std::abs(z) < 1e-8) {
        // symmetric Taylor expansion
        return 0.5 - z/24.0 + z*z/720.0;
    }

    if (z > 0) {
        double sz = std::sqrt(z);
        return (1.0 - std::cos(sz)) / z;
    } else {
        double sz = std::sqrt(-z);
        return (std::cosh(sz) - 1.0) / (-z);
    }
}

double stumpff_S(double z) {
    if (std::abs(z) < 1e-8) {
        return 1.0/6.0 - z/120.0 + z*z/5040.0;
    }

    if (z > 0) {
        double sz = std::sqrt(z);
        return (sz - std::sin(sz)) / (sz*sz*sz);
    } else {
        double sz = std::sqrt(-z);
        return (std::sinh(sz) - sz) / (sz*sz*sz);
    }
}

ChiResult solve_chi(double mu, double alpha,
                    const Vec3& r0,
                    double vr, double dt,
                    double tol,
                    int max_iter)
{
    double r = std::sqrt(r0.norm2());
    double chi0;

    if (alpha > 0) {
        chi0 = std::sqrt(mu) * dt * alpha;
    } else if (alpha < 0) {
        chi0 = std::sqrt(mu) * std::abs(dt);
        if (dt < 0) chi0 = -chi0;
    } else {
        chi0 = std::sqrt(mu) * dt / r;
    }

    if (!std::isfinite(chi0)) chi0 = 0.0;

    auto F = [&](double chi) {
        double z = alpha * chi * chi;
        double C = stumpff_C(z);
        double S = stumpff_S(z);

        return (r * vr / std::sqrt(mu)) * chi * chi * C
             + (1 - r * alpha) * chi * chi * chi * S
             + r * chi
             - std::sqrt(mu) * dt;
    };

    auto dF = [&](double chi) {
        double z = alpha * chi * chi;
        double C = stumpff_C(z);
        double S = stumpff_S(z);

        return (r * vr / std::sqrt(mu)) * chi * (1 - z * S)
             + (1 - r * alpha) * chi * chi * C
             + r;
    };

    NewtonResult result = Newton_Solver(F, dF, chi0, tol, max_iter);

    return {result.root, result.iterations, result.converged};
}