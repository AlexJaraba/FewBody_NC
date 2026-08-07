#pragma once

#include <vector>

#include "core/body.h"
#include "dynamics/pairing.h"

struct TwoBodyState;

struct PairKeplerMapResult {
    bool converged =  false;
    int iterations = 0;
};

void advance_pair_COM(TwoBodyState& pair_state, double timestep);

[[nodiscard]] PairKeplerMapResult apply_exact_two_body_flow(std::vector<Body>& bodies, const Pair& pair, double timestep, double gravitational_constant);