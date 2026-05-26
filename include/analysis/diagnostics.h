#pragma once

#include <vector>

#include "core/body.h"
#include "dynamics/pairing.h"
#include "math/vec3.h"

struct Diagnostics {
    double kinetic_energy;
    double potential_energy;
    double total_energy;
    double shadow_energy;
    double angular_momentum;
    double com_drift;
    double linear_momentum;
    double timestep;
};

Diagnostics compute_diagnostics(const std::vector<Body>& bodies, double G, double dt);