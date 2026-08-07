#pragma once

#include <vector>

#include "math/vec3.h"

struct KeplerPropagationResult {
    Vec3 relative_position;
    Vec3 relative_momentum;
    bool converged = false;
    int iterations = 0;
};

[[nodiscard]] KeplerPropagationResult propagate_two_body_state(double gravitational_parameter, 
    double reduced_mass, const Vec3& initial_relative_position, const Vec3& initial_relative_momentum, double timestep);