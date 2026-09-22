#pragma once

#include <vector>
#include "dynamics/extern_potentials.h"
#include "core/body.h"

class Integrator {
public:
    virtual void step(std::vector<Body>& bodies, double dt, double G, const ExternalPotential* external_potential) = 0;
    virtual ~Integrator() = default;
};