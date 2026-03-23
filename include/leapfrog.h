#pragma once

#include <vector>
#include "integrator.h"

class Leapfrog : public Integrator {
public:
    void step(std::vector<Body>& bodies, double dt) override;
};