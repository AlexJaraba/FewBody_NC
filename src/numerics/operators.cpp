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
#include "perturbation_hamiltonian.h"

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
    const double eps = 1e-10;
    const int N = state.Q.size();

    for (int i = 1; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            CanonicalState plus_state = state;
            CanonicalState minus_state = state;
            plus_state.Q[i][k] += eps;
            minus_state.Q[i][k] -= eps;

            const double H_plus = compute_perturbation_hamiltonian(plus_state, pairs, G);
            const double H_minus = compute_perturbation_hamiltonian(minus_state, pairs, G);
            const double dH_dq = (H_plus - H_minus) / (2.0 * eps);

            state.P[i][k] -= dt * dH_dq;
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

    // for (int i = static_cast<int>(state.Q.size()) - 1; i >= 1; --i) {
    //     kepler_pair_step(state, i, 0.5 * dt, G);
    // }
}

void test_kepler_reversibility(CanonicalState& inital_state, double dt, double G) {
    std::cout << "\n=== KEPLER REVERSIBILITY TEST ===\n";

    CanonicalState state = inital_state;
    CanonicalState initial = state;

    //Forward
    kepler_operator(state, {}, dt, G);

    //Backward
    kepler_operator(state, {}, -dt, G);

    double max_q_error = 0.0;
    double max_p_error = 0.0;

    for (size_t i = 1; i < state.Q.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            max_q_error = std::max(max_q_error, std::abs(state.Q[i][k] - initial.Q[i][k]));
            max_p_error = std::max(max_p_error, std::abs(state.P[i][k] - initial.P[i][k]));
        }
    }

    std::cout << "Max Q Error: " << max_q_error << std::endl;
    std::cout << "Max P Error: " << max_p_error << std::endl;
}

void test_kick_reversibility(CanonicalState& inital_state, const std::vector<Pair>& pairs, double dt, double G) {
    std::cout << "\n=== KICK REVERSIBILITY TEST ===\n";

    CanonicalState state = inital_state;
    CanonicalState initial = state;

    //Forward
    kick_operator(state, pairs, dt, G);

    //Backward
    kick_operator(state, pairs, -dt, G);

    double max_p_error = 0.0;

    for (size_t i = 1; i < state.Q.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            max_p_error = std::max(max_p_error, std::abs(state.P[i][k] - initial.P[i][k]));
        }
    }

    std::cout << "Max P Error: " << max_p_error << std::endl;
}