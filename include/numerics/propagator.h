#pragma once

#include <vector>

#include "vec3.h"

struct CanonicalStateVector {
    Vec3 q;
    Vec3 p;
    bool converged;
    int iterations;
};

CanonicalStateVector propagate_universal(
    double mu_grav,
    double reduced_mass,
    const Vec3& q0,
    const Vec3& p0,
    double dt
);