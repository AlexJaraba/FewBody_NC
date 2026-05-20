#include "leapfrog.h"
#include "operators.h"

void Leapfrog::step(CanonicalState& state, double dt) {
    extern double G;

    // Kick half-step
    kick_operator(state, {}, 0.5 * dt, G);

    // Kepler full-step
    kepler_operator(state, {}, dt, G);

    // Kick half-step
    kick_operator(state, {}, 0.5 * dt, G);
}
