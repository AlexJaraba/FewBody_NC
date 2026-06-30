#pragma once

#include <vector>

#include "math/vec3.h"

struct VariationalState {
    std::vector<Vec3> delta_q;
    std::vector<Vec3> delta_p;
};