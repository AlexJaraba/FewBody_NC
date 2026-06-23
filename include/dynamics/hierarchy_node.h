#pragma once

#include <memory>
#include <vector>

#include "math/vec3.h"

struct HierarchyNode {
    int node_id = -1;
    int parent_id = -1;

    // Physical body index for leaves only
    // Internal cluster nodes keep this as -1
    int body_index = -1;

    int depth = 0;
    
    //Children
    std::shared_ptr<HierarchyNode> left;
    std::shared_ptr<HierarchyNode> right;

    // All physical body indices contained in this node
    std::vector<int> body_indices;

    // Aggregate properties
    double total_mass = 0.0;
    Vec3 com_position;
    Vec3 com_velocity;

    // Data describing the merge that created this internal node
    // Leaves keep these as 0
    double internal_separation = 0.0;
    double internal_strength = 0.0;

    bool is_leaf() const {
        return !left && !right;
    }
    bool is_internal() const {
        return !is_leaf();
    }
    bool is_binary() const {
        return left && right;
    }
    
    int leaf_count() const {
        return static_cast<int>(body_indices.size());
    }
};