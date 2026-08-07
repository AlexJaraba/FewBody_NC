#pragma once

#include <vector>

#include "core/body.h"
#include "math/vec3.h"

[[nodiscard]] std::vector<Vec3> calculate_newtonian_acceleration(const std::vector<Body>& bodies, double gravitational_constant);