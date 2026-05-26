#pragma once

#include <vector>

#include "math/vec3.h"

struct CanonicalState{
    std::vector<Vec3> Q;
    std::vector<Vec3> P;
    Vec3 com_position;
    Vec3 com_velocity;
    std::vector<double> mu;
    std::vector<double> M;
    std::vector<double> physical_mass;
};