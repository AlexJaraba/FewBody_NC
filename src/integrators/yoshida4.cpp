#include "integrators/yoshida4.h"
#include "integrators/hernandez.h"
#include "core/canonical_state.h"

Yoshida4::Yoshida4(const std::vector<Pair>& pairs) : base_integrator_(pairs) {}

void Yoshida4::step(CanonicalState& state, double dt) {
    base_integrator_.step(state, w1_ * dt);
    base_integrator_.step(state, w0_ * dt);
    base_integrator_.step(state, w1_ * dt);
}