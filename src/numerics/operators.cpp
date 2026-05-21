#include <cmath>
#include <stdexcept>
#include <iostream>

#include "operators.h"
#include "pairing.h"
#include "propagator.h"
#include "jacobi.h"
#include "univ_vari_solve.h"
#include "jacobi_transform.h"
#include "canonical_state.h"

void drift_operator(CanonicalState& state, double dt) {
    // const int N = state.Q.size();

    // for (int i = 1; i < N; ++i) {
    //     const double mu = state.mu[i];
    //     for (int k = 0; k < 3; ++k) {
    //         state.Q[i][k] += dt * state.P[i][k] / mu;
    //     }
    // }
}

void kick_operator(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) {
    const int N = state.Q.size();

    std::vector<std::vector<double>> dP(N, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> r(N, std::vector<double>(3, 0.0));
    std::vector<double> R_prev(3, 0.0);

    double M_prev = state.physical_mass[0];

    for (int i = 1; i < N; ++i) {
        std::vector<double> r_com_prev(3);
        for (int k = 0; k < 3; ++k) {
            r_com_prev[k] = R_prev[k] / M_prev;
        }

        for (int k = 0; k < 3; ++k) {
            r[i][k] = r_com_prev[k] + state.Q[i][k];
        }

        for (int k = 0; k < 3; ++k) {
            R_prev[k] += state.physical_mass[i] * r[i][k];
        }
        M_prev += state.physical_mass[i];
    }

    // Physical perturbation forces
    for (const auto& pair : pairs) {

        const int i = pair.i;
        const int j = pair.j;

        std::vector<double> dr(3);
        for (int k = 0; k < 3; ++k) {
            dr[k] = r[j][k] - r[i][k];
        }

        double r2 = 0.0;

        for (int k = 0; k < 3; ++k)
            r2 += dr[k] * dr[k];

        const double dist = std::sqrt(r2) + 1e-15;
        const double coeff = (G * state.physical_mass[i] * state.physical_mass[j]) / (dist * dist * dist);

        for (int k = 0; k < 3; ++k) {
            const double F = coeff * dr[k];
            dP[i][k] += dt * F;
            dP[j][k] -= dt * F;
        }
    }

    // Canonical momentum update
    for (int i = 1; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            state.P[i][k] += dP[i][k];
        }
    }
}

static void kepler_pair_step(CanonicalState& state, int i, double dt, double G) {
    const double mu_grav = G * state.M[i];
    const double r0 = std::sqrt(state.Q[i][0] * state.Q[i][0] + state.Q[i][1] * state.Q[i][1] + state.Q[i][2] * state.Q[i][2]);
    const double p2 = state.P[i][0] * state.P[i][0] + state.P[i][1] * state.P[i][1] + state.P[i][2] * state.P[i][2];
    const double v2 = p2 / (state.mu[i] * state.mu[i]);
    const double alpha = (2.0 / r0) - (v2 / mu_grav);
    
    CanonicalStateVector result = propagate_universal(mu_grav, state.mu[i], state.Q[i], state.P[i], dt);

    if (!result.converged) {
        std::cerr
            << "\n====================================\n"
            << "KEPLER SOLVE FAILURE\n"
            << "Jacobi Index : " << i << "\n"
            << "dt           : " << dt << "\n"
            << "r0           : " << r0 << "\n"
            << "v2           : " << v2 << "\n"
            << "alpha        : " << alpha << "\n"
            << "mu_grav      : " << mu_grav << "\n"
            << "mu_reduced   : " << state.mu[i] << "\n"
            << "M_total      : " << state.M[i] << "\n"
            << "====================================\n"
            << std::endl;
        throw std::runtime_error("Kepler solve failed.");
    }

    state.Q[i] = result.q;
    state.P[i] = result.p;
}

void kepler_operator(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) {
    for (size_t  i = 1; i < state.Q.size(); ++i) {
        kepler_pair_step(state, i, dt, G);
    }
}

void symmetric_kepler_operator(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) {
    for (size_t i = 1; i < state.Q.size(); ++i) {
        kepler_pair_step(state, i, 0.5 * dt, G);
    }

    for (int i = static_cast<int>(state.Q.size()) - 1; i >= 1; --i) {
        kepler_pair_step(state, i, 0.5 * dt, G);
    }
}