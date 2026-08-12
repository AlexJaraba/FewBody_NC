#pragma once

#include <vector>

#include "integrators/integrator.h"

class Leapfrog : public Integrator {
public:
    void step(std::vector<Body>& bodies, double dt, double G) override;
};