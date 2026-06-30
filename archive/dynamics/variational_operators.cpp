#include "dynamics/variational_operators.h"
#include "core/variational_state.h"
#include "numerics/perturbation_forces.h"
#include "math/mat3.h"
#include "math/vec3.h"

/* ===================================================================

    Variational operators

    This file evolves tangent perturbations alongside the main canonical state.

    The tangent variables are:
        delta_q
        delta_p
    The variational kick uses the force Jacobian:

        delta_p <- delta_p + dt * J_force * delta_q

    These routines are currently useful for provisional tangent diagnostics, 
    but Lyapunov exponents should not yet be treated as final validation metrics.

   =================================================================== */

void variational_drift_operator(const CanonicalState& state, VariationalState& var_state, double dt) {
    const int N = static_cast<int>(var_state.delta_q.size());
    for (int i = 1; i < N; ++i) {
        var_state.delta_q[i] += (dt / state.mu[i]) * var_state.delta_p[i];
    }
}

/*
    Apply the linearized perturbation kick.
    The Hessian field in ForceResult is currently used as the force Jacobian:
        J_force = dF / dQ
    so the update is:
        delta_p += dt * J_force * delta_q
*/

void variational_kick_operator(const CanonicalState& state, VariationalState& var_state, const std::vector<Pair>& pairs, double dt, double G) {
    auto result = compute_perturbation_forces(state, pairs, G);
    const int N = static_cast<int>(var_state.delta_p.size());
    std::vector<Vec3> delta_p_new = var_state.delta_p;

    for (int i = 1; i < N; ++i) {
        Vec3 sum;
        for (int j = 1; j < N; ++j) {
            sum += result.hessian[i][j] * var_state.delta_q[j];
        }
        delta_p_new[i] += dt * sum;
    }
    var_state.delta_p = delta_p_new;
}