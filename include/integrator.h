#pragma once

#include <vector>
#include "body.h"

class Integrator {
public:
    virtual void step(std::vector<Body>& bodies, double dt) = 0;
    virtual ~Integrator() = default;
};