#pragma once

#include <vector>

#include "body.h"
#include "pairing.h"
#include "canonical_state.h"

class SymplecticCorrector {
    public:
        SymplecticCorrector(double coefficient);
        void apply_forward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const;
        void apply_backward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const;
    
    private:
        double coefficient_;
};