#pragma once

#include <vector>

#include "core/body.h"
#include "math/vec3.h"

struct SystemDiagnostics {
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    double total_energy = 0.0;
    Vec3 linear_momentum;
    Vec3 angular_momentum;
    Vec3 COM_position;
    Vec3 COM_velocity;
    double timestep = 0.0;
};

[[nodiscard]] SystemDiagnostics calculate_system_diagnostics(const std::vector<Body>& bodies, double gravitational_constant, double timestep);