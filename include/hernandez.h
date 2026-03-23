#pragma once

#include <vector>

#include "body.h"
#include "integrator.h"

class Hernandez : public Integrator {
public:
    void step(std::vector<Body>& bodies, double dt);
};