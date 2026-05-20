#include "yoshida4.h"
#include "hernandez.h"
#include "canonical_state.h"

Yoshida4::Yoshida4(const std::vector<Pair>& pairs) : pairs_(pairs) {}

void Yoshida4::step(CanonicalState& state, double dt) {
    Hernandez hernandez(pairs_);
    hernandez.step(state, w1_ * dt);
    hernandez.step(state, w0_ * dt);
    hernandez.step(state, w1_ * dt);
}