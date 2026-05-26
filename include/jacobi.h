#pragma once

#include <vector>

#include "body.h"
#include "canonical_state.h"

CanonicalState compute_jacobi_state(const std::vector<Body>& bodies);

void reconstruct_bodies(const CanonicalState& state, std::vector<Body>& bodies);
std::vector<std::vector<double>> reconstruct_cartesian_position(const CanonicalState& state);