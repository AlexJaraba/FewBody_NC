#pragma once

#include <vector>

#include "core/body.h"

struct HernandezPairMapResult {
    bool converged;
    int iterations;
};

HernandezPairMapResult propagatePairKepler(std::vector<Body>& bodies, int i, int j, double dt, double G);