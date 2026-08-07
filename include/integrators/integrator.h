#pragma once

#include <vector>

#include "core/canonical_state.h"
#include "core/body.h"

class Integrator {
public:
    virtual void step(std::vector<Body>& bodies, double dt, double G) = 0;
    virtual ~Integrator() = default;
};