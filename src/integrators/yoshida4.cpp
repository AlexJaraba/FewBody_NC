#include "yoshida4.h"
#include "hernandez.h"
#include "canonical_state.h"

Yoshida4::Yoshida4(const std::vector<Pair>& pairs) : base_integrator_(pairs) {}

void Yoshida4::step(CanonicalState& state, double dt) {
    base_integrator_.step(state, w1_ * dt);
    base_integrator_.step(state, w0_ * dt);
    base_integrator_.step(state, w1_ * dt);
}