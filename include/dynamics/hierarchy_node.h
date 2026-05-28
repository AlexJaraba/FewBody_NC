#pragma once

#include <memory>

#include "core/body.h"
#include "math/vec3.h"

struct HierarchyNode {
    // Leaf Body index
    int body_index = -1;
    
    //Children
    std::shared_ptr<HierarchyNode> left;
    std::shared_ptr<HierarchyNode> right;

    // Aggregate properties
    double total_mass = 0.0;
    Vec3 com_position;
    Vec3 com_velocity;

    // True if physical particle
    bool is_leaf() const {
        return !left && !right;
    }
};