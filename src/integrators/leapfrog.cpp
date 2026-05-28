#include "integrators/leapfrog.h"
#include "dynamics/operators.h"

Leapfrog::Leapfrog(const std::vector<Pair>& pairs) : pairs_(pairs) {}

void Leapfrog::step(CanonicalState& state, double dt, double G) {
    // Kick half-step
    kick_operator(state, pairs_, 0.5 * dt, G);

    // Drift full-step
    drift_operator(state, dt);

    // Kick half-step
    kick_operator(state, pairs_, 0.5 * dt, G);
}
