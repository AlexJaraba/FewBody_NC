#pragma once

#include <vector>

#include "math/vec3.h"

struct KeplerPropagationResult {
    Vec3 q;
    Vec3 p;
    bool converged;
    int iterations;
};

KeplerPropagationResult propagateUniversal(
    double mu_grav,
    double reduced_mass,
    const Vec3& q0,
    const Vec3& p0,
    double dt
);