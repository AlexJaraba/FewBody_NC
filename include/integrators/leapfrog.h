#pragma once

#include <vector>

#include "integrators/integrator.h"
#include "core/canonical_state.h"
#include "dynamics/pairing.h"

class Leapfrog : public Integrator {
public:
    explicit Leapfrog(const std::vector<Pair>& pairs);
    void step(CanonicalState& state, double dt) override;
private:
    std::vector<Pair> pairs_;
};