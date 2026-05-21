#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>

#include "newton_solver.h"
#include "univ_vari_solve.h"

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

double norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

ChiResult solve_chi(double mu, double alpha,
                    const std::vector<double>& r0,
                    double vr, double dt,
                    double abs_tol,
                    double rel_tol,
                    int max_iter)
{
    double r = norm(r0);
    double chi0;

    if (alpha > 1e-12) {
        chi0 = std::sqrt(mu) * alpha * dt;
    }
    else if (alpha < -1e-12) {
        const double a = 1.0 / alpha;
        const double term = -2.0 * mu * alpha * dt;
        const double denom = r * vr + std::copysign(1.0, dt) * std::sqrt(-mu * a) * (1.0 - r / alpha);
        if (std::abs(denom) < 1e-15) {
            chi0 = (std::sqrt(mu) * dt) / r;
        }
        else {
            chi0 = std::copysign(std::sqrt(-a) * std::log(term / denom), dt);
        }
    }
    else {
        chi0 = std::sqrt(mu) * dt / r;
    }

    if (!std::isfinite(chi0)) {
        chi0 = (std::sqrt(mu) * dt) / r;
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

    NewtonResult result = Newton_Solver(F, dF, chi0, abs_tol, rel_tol, max_iter);

    return {result.root, result.iterations, result.converged};
}