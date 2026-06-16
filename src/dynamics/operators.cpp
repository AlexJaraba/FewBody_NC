#include <cmath>
#include <stdexcept>
#include <iostream>

#include "dynamics/operators.h"
#include "dynamics/pairing.h"
#include "dynamics/jacobi.h"
#include "numerics/univ_vari_solve.h"
#include "numerics/propagator.h"
#include "numerics/perturbation_forces.h"
#include "core/canonical_state.h"
#include "math/vec3.h"

/* =====================================================================

    Canonical evolution operators

    This file contains the primitive operators used by the symplectic compositions:
        drift_operator
        kick_operator
        kepler_operator
        symmetric_kepler_operator
    
    The Hernandez integrator is built from these operators.
    In the active Jacobi split, the Kepler operator evolves each Jacobi coordinate under its two-body Kepler Hamiltonin.
    The kick operator applies the perturbation force.

   ===================================================================== */

void drift_operator(CanonicalState& state, double dt) {
    const int N = state.Q.size();

    for (int i = 1; i < N; ++i) {
        const double mu = state.mu[i];
        state.Q[i] += (dt * state.P[i]) / mu;
    }
}

/*
    Apply perturbation kick:
        P <- P + dt * F_perturbation(Q)
    The force is the Jacobi generalized perturbation force returned by compute_perturbation_forces()
*/

void kick_operator(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) {
    auto result = compute_perturbation_forces(state, pairs, G);
    const int N = static_cast<int>(state.Q.size());

    for (int i = 1; i < N; ++i) {
        state.P[i] += dt * result.gradient[i];
    }
}

static void kepler_pair_step(CanonicalState& state, int i, double dt, double G) {
    const double mu_grav = G * state.M[i];
    const double r0 = state.Q[i].norm();
    const double p2 = state.P[i].norm2();
    const double v2 = p2 / (state.mu[i] * state.mu[i]);
    
    if (r0 < 1e-14) {
        return;
    }
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

/*
    Apply Kepler evolution to each Jacobi coordinate.
    Each Jacobi coordinate Q_k is advanced as a two-body Kepler problem using the reduced mass mu_k and enclosed mass M_k.
*/

void kepler_operator(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) {
    for (size_t  i = 1; i < state.Q.size(); ++i) {
        kepler_pair_step(state, i, dt, G);
    }
}

/*
    Symmetric Kepler operator wrapper.
    This is used by symmetric compositions so forward and backward integrations use consistent Kepler evolution.
*/

void symmetric_kepler_operator(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) {
    for (size_t i = 1; i < state.Q.size(); ++i) {
        kepler_pair_step(state, i, 0.5 * dt, G);
    }

    for (int i = static_cast<int>(state.Q.size()) - 1; i >= 1; --i) {
        kepler_pair_step(state, i, 0.5 * dt, G);
    }
}

void test_kepler_reversibility(CanonicalState& initial_state, double dt, double G) {
    std::cout << "\n=== KEPLER REVERSIBILITY TEST ===\n";

    CanonicalState state = initial_state;
    CanonicalState initial = state;

    //Forward
    kepler_operator(state, {}, dt, G);

    //Backward
    kepler_operator(state, {}, -dt, G);

    double max_q_error = 0.0;
    double max_p_error = 0.0;

    for (size_t i = 1; i < state.Q.size(); ++i) {
        Vec3 dQ = state.Q[i] - initial.Q[i];
        Vec3 dP = state.P[i] - initial.P[i];
        max_q_error = std::max(max_q_error, dQ.norm());
        max_p_error = std::max(max_p_error, dP.norm());
    }

    std::cout << "Max Q Error: " << max_q_error << std::endl;
    std::cout << "Max P Error: " << max_p_error << std::endl;
}

void test_kick_reversibility(CanonicalState& initial_state, const std::vector<Pair>& pairs, double dt, double G) {
    std::cout << "\n=== KICK REVERSIBILITY TEST ===\n";

    CanonicalState state = initial_state;
    CanonicalState initial = state;

    //Forward
    kick_operator(state, pairs, dt, G);

    //Backward
    kick_operator(state, pairs, -dt, G);

    double max_p_error = 0.0;

    for (size_t i = 1; i < state.Q.size(); ++i) {
        Vec3 dP = state.P[i] - initial.P[i];
        max_p_error = std::max(max_p_error, dP.norm());
    }

    std::cout << "Max P Error: " << max_p_error << std::endl;
}