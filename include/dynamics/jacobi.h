#pragma once

#include <vector>

#include "core/body.h"
#include "core/canonical_state.h"
#include "core/reconstruction.h"
#include "math/vec3.h"

CanonicalState compute_jacobi_state(const std::vector<Body>& bodies);

void reconstruct_bodies(const CanonicalState& state, std::vector<Body>& bodies);