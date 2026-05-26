#pragma once

#include <vector>

#include "body.h"
#include "canonical_state.h"
#include "vec3.h"
#include "reconstruction.h"

CanonicalState compute_jacobi_state(const std::vector<Body>& bodies);

void reconstruct_bodies(const CanonicalState& state, std::vector<Body>& bodies);