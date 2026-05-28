#include <limits>
#include <cmath>

#include "dynamics/hierarchy_tree.h"

HierarchyTree::HierarchyTree(const std::vector<Body>& bodies) {
    std::vector<int> indices;
    for (size_t i = 0; i < bodies.size(); ++i) {
        indices.push_back(i);
    }
    root = buildTree(bodies, indices);
}

std::shared_ptr<HierarchyNode> HierarchyTree::buildTree(const std::vector<Body>& bodies, std::vector<int> indices) {
    // Leaf Node
    if (indices.size() == 1) {
        auto node = std::make_shared<HierarchyNode>();
        const int i = indices[0];
        node->body_index = i;
        node->total_mass = bodies[i].mass;
        node->com_position = bodies[i].position;
        node->com_velocity = bodies[i].velocity;
        return node;
    }

    if (indices.size() == 2) {
        auto node = std::make_shared<HierarchyNode>();
        auto left_child = std::make_shared<HierarchyNode>();
        auto right_child = std::make_shared<HierarchyNode>();

        const int i = indices[0];
        const int j = indices[1];

        left_child->body_index = i;
        left_child->total_mass = bodies[i].mass;
        left_child->com_position = bodies[i].position;
        left_child->com_velocity = bodies[i].velocity;

        right_child->body_index = j;
        right_child->total_mass = bodies[j].mass;
        right_child->com_position = bodies[j].position;
        right_child->com_velocity = bodies[j].velocity;

        node->left = left_child;
        node->right = right_child;

        node->total_mass = bodies[i].mass + bodies[j].mass;
        node->com_position = (bodies[i].position * bodies[i].mass + bodies[j].position * bodies[j].mass) / node->total_mass;
        node->com_velocity = (bodies[i].velocity * bodies[i].mass + bodies[j].velocity * bodies[j].mass) / node->total_mass;

        return node;
    }

    // Strongest Pair
    double best_strength = -1.0;
    int best_i = -1, best_j = -1;
    for (size_t a = 0; a < indices.size(); ++a) {
        for (size_t b = a + 1; b < indices.size(); ++b) {
            const int i = indices[a];
            const int j = indices[b];

            Vec3 dr = bodies[i].position - bodies[j].position;
            const double r2 = dr.norm2();
            const double strength = (bodies[i].mass * bodies[j].mass) / (r2 + 1e-15);

            if (strength > best_strength) {
                best_strength = strength;
                best_i = i;
                best_j = j;
            }
        }
    }

    // Children
    auto node = std::make_shared<HierarchyNode>();
    std::vector<int> left_indices, right_indices;
    left_indices.push_back(best_i);
    left_indices.push_back(best_j);

    for (int idx : indices) {
        if (idx != best_i && idx != best_j) {
            right_indices.push_back(idx);
        }
    }

    node->left = buildTree(bodies, left_indices);
    
    if (!right_indices.empty()) {
        node->right = buildTree(bodies, right_indices);
    }

    node->total_mass = 0.0;
    node->com_position = Vec3();
    node->com_velocity = Vec3();

    auto accumulate = [&](std::shared_ptr<HierarchyNode> child) {
        if (child) {
            node->total_mass += child->total_mass;
            node->com_position += child->com_position * child->total_mass;
            node->com_velocity += child->com_velocity * child->total_mass;
        }
    };

    accumulate(node->left);
    accumulate(node->right);

    if (node->total_mass > 0.0) {
        node->com_position /= node->total_mass;
        node->com_velocity /= node->total_mass;
    }

    return node;
}