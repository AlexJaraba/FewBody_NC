#pragma once

#include "core/canonical_state.h"

class Integrator {
public:
    virtual void step(CanonicalState& state, double dt) = 0;
    virtual ~Integrator() = default;
};