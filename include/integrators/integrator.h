#pragma once

#include "core/canonical_state.h"

class Integrator {
public:
    virtual void step(CanonicalState& state, double dt, double G) = 0;
    virtual ~Integrator() = default;
};