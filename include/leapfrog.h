#pragma once

#include <vector>

#include "integrator.h"
#include "canonical_state.h"

class Leapfrog : public Integrator {
public:
    void step(CanonicalState& state, double dt) override;
};