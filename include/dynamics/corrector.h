#pragma once

#include <vector>

#include "pairing.h"
#include "canonical_state.h"

class SymplecticCorrector {
public:
    void apply_forward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const;
    void apply_backward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const;
private:
    void apply(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G, double sign) const;
};