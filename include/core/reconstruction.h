#pragma once

#include <vector>

#include "vec3.h"
#include "canonical_state.h"

std::vector<Vec3> reconstruct_cartesian_positions(const CanonicalState& state);
std::vector<Vec3> reconstruct_cartesian_velocities(const CanonicalState& state);