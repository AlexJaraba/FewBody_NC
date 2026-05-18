#pragma once

#include <vector>

#include "body.h"
#include "pairing.h"
#include "jacobi.h"

void drift_operator(
    std::vector<Body>& bodies,
    double dt
);

void kick_operator(
    std::vector<Body>& bodies,
    const std::vector<Pair>& pairs,
    double dt,
    double G
);

void kepler_operator(
    std::vector<Body>& bodies,
    const std::vector<Pair>& pairs,
    double dt,
    double G
);

void symmetric_kepler_operator(
    std::vector<Body>& bodies,
    const std::vector<Pair>& pairs,
    double dt,
    double G
);