#pragma once

#include <vector>

#include "integrator.h"
#include "canonical_state.h"
#include "pairing.h"

class Leapfrog : public Integrator {
public:
    explicit Leapfrog(const std::vector<Pair>& pairs);
    void step(CanonicalState& state, double dt) override;
private:
    std::vector<Pair> pairs_;
};