#pragma once

#include <vector>

#include "math/mat3.h"
#include "math/vec3.h"

struct ForceResult {
    double potential = 0.0;
    std::vector<Vec3> gradient;
    std::vector<std::vector<Mat3>> hessian;
};