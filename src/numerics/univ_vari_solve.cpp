#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>

#include "newton_solver.h"
#include "univ_vari_solve.h"

double stumpff_C(double z) {
    if (z > 0) {
        double sz = std::sqrt(z);
        return (1.0 - std::cos(sz)) / z;
    } else if (z < 0) {
        double sz = std::sqrt(-z);
        return (std::cosh(sz) - 1.0) / (-z);
    } else {
        return 0.5;
    }
}

double stumpff_S(double z) {
    if (z > 0) {
        double sz = std::sqrt(z);
        return (sz - std::sin(sz)) / (sz * sz * sz);
    } else if (z < 0) {
        double sz = std::sqrt(-z);
        return (std::sinh(sz) - sz) / (sz * sz * sz);
    } else {
        return 1.0 / 6.0;
    }
}

double norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

ChiResult solve_chi(double mu, double alpha,
                    const std::vector<double>& r0,
                    double vr, double dt,
                    double tol,
                    int max_iter)
{
    double r = norm(r0);
    double chi0;

    if (alpha > 0) {
        chi0 = std::sqrt(mu) * dt * alpha;
    } else if (alpha < 0) {
        chi0 = std::sqrt(-1.0 / alpha) *
               std::log((-2 * mu * alpha * dt) /
               (vr + std::copysign(1.0, dt) *
               std::sqrt(-mu / alpha) * (1 - r * alpha)));
    } else {
        chi0 = std::sqrt(mu) * dt / r;
    }

    if (!std::isfinite(chi0)) {
        return {std::numeric_limits<double>::quiet_NaN(), 0, false};
    }

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