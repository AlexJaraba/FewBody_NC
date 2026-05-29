#include "dynamics/variational_operators.h"
#include "core/variational_state.h"
#include "numerics/perturbation_forces.h"
#include "math/mat3.h"
#include "math/vec3.h"

void variational_drift_operator(const CanonicalState& state, VariationalState& var_state, double dt) {
    const int N = static_cast<int>(var_state.delta_q.size());
    for (int i = 1; i < N; ++i) {
        var_state.delta_q[i] += (dt / state.mu[i]) * var_state.delta_p[i];
    }
}

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