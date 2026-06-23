#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "dynamics/hierarchy_tree.h"

namespace {
    constexpr double HIERARCHY_DISTANCE_FLOOR = 1.0e-300;
    constexpr double HIERARCHY_VALIDATE_TOL = 1.0e-12;
}

HierarchyTree::HierarchyTree(const std::vector<Body>& bodies) {
    if (bodies.empty()) {
        root = nullptr;
        return;
    }

    std::vector<std::shared_ptr<HierarchyNode>> active_nodes;
    active_nodes.reserve(bodies.size());

    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        active_nodes.push_back(make_leaf(bodies, i));
    }

    while (active_nodes.size() > 1) {
        double best_strength = -std::numeric_limits<double>::infinity();
        std::size_t best_a = 0;
        std::size_t best_b = 1;

        for (std::size_t a = 0; a < active_nodes.size(); ++a) {
            for (std::size_t b = a + 1; b < active_nodes.size(); ++b) {
                const double strength = cluster_strength(active_nodes[a], active_nodes[b]);

                if (strength > best_strength) {
                    best_strength = strength;
                    best_a = a;
                    best_b = b;
                }
            }
        }

        std::shared_ptr<HierarchyNode> left = active_nodes[best_a];
        std::shared_ptr<HierarchyNode> right = active_nodes[best_b];

        // Deterministic left/right ordering
        if (min_body_index(right) < min_body_index(left)) {
            std::swap(left, right);
        }

        std::shared_ptr<HierarchyNode> merged = merge_nodes(left, right);

        const std::size_t high = std::max(best_a, best_b);
        const std::size_t low = std::min(best_a, best_b);

        active_nodes.erase(active_nodes.begin() + static_cast<long>(high));
        active_nodes.erase(active_nodes.begin() + static_cast<long>(low));
        active_nodes.push_back(merged);
    }

    root = active_nodes.front();
    assign_depths(root, 0);
    validate(static_cast<int>(bodies.size()));
    }

bool HierarchyTree::empty() const {
    return root == nullptr;
}

int HierarchyTree::node_count() const {
    return static_cast<int>(nodes_.size());
}
int HierarchyTree::leaf_count() const {
    if (!root) {
        return 0;
    }
    return root->leaf_count();
}

const std::vector<std::shared_ptr<HierarchyNode>>& HierarchyTree::nodes() const {
    return nodes_;
}

std::vector<int> HierarchyTree::leaf_body_indices() const {
    std::vector<int> indices;
    collect_leaf_indices(root, indices);
    std::sort(indices.begin(), indices.end());
    return indices;
}

void HierarchyTree::validate(int expected_leaf_count) const {
    if (expected_leaf_count < 0) {
        throw std::runtime_error("HierarchyTree::validate received a negative expected leaf count.");
    }
    if (expected_leaf_count == 0) {
        if (root) {
            throw std::runtime_error("HierarchyTree validation failed: non-null root for empty tree.");
        }
        return;
    }
    if (!root) {
        throw std::runtime_error("HierarchyTree validation failed: null root for non-empty tree.");
    }

    std::vector<bool> seen_bodies(static_cast<std::size_t>(expected_leaf_count), false);
    const int counted_leaves = validate_node(root, seen_bodies);

    if (counted_leaves != expected_leaf_count) {
        throw std::runtime_error("HierarchyTree validation failed: wrong number of leaves.");
    }
    
    for (bool seen : seen_bodies) {
        if (!seen) {
            throw std::runtime_error("HierarchyTree validation failed: missing body index.");
        }
    }
}

std::shared_ptr<HierarchyNode> HierarchyTree::make_leaf(const std::vector<Body>& bodies, int body_index) {
    if (body_index < 0 || body_index >= static_cast<int>(bodies.size())) {
        throw std::runtime_error("HierarchyTree::make_leaf received an invalid body index.");
    }

    const Body& body = bodies[body_index];

    auto node = std::make_shared<HierarchyNode>();
    node->node_id = next_node_id_++;
    node->parent_id = -1;
    node->body_index = body_index;
    node->body_indices = {body_index};
    node->total_mass = body.mass;
    node->com_position = body.position;
    node->com_velocity = body.velocity;
    node->internal_separation = 0.0;
    node->internal_strength = 0.0;
    nodes_.push_back(node);

    return node;
}

std::shared_ptr<HierarchyNode> HierarchyTree::merge_nodes(const std::shared_ptr<HierarchyNode>& left, const std::shared_ptr<HierarchyNode>& right) {
    if (!left || !right) {
        throw std::runtime_error("HierarchyTree::merge_nodes received a null child.");
    }

    auto node = std::make_shared<HierarchyNode>();
    node->node_id = next_node_id_++;
    node->parent_id = -1;
    node->body_index = -1;
    node->left = left;
    node->right = right;
    left->parent_id = node->node_id;
    right->parent_id = node->node_id;
    node->body_indices = merged_body_indices(left, right);
    node->total_mass = left->total_mass + right->total_mass;

    if (node->total_mass <= 0.0) {
        throw std::runtime_error("HierarchyTree::merge_nodes produced non-positive total mass.");
    }

    node->com_position = (left->com_position * left->total_mass + right->com_position * right->total_mass) / node->total_mass;
    node->com_velocity = (left->com_velocity * left->total_mass + right->com_velocity * right->total_mass) / node->total_mass;

    const Vec3 separation_vector = left->com_position - right->com_position;

    node->internal_separation = separation_vector.norm();
    node->internal_strength = cluster_strength(left, right);

    nodes_.push_back(node);

    return node;
}

double HierarchyTree::cluster_strength(const std::shared_ptr<HierarchyNode>& a, const std::shared_ptr<HierarchyNode>& b) {
    if (!a || !b) {
        throw std::runtime_error("HierarchyTree::cluster_strength received a null node.");
    }

    const Vec3 dr = a->com_position - b->com_position;
    const double r2 = std::max(dr.norm2(), HIERARCHY_DISTANCE_FLOOR);

    return (a->total_mass * b->total_mass) / r2;
}

int HierarchyTree::min_body_index(const std::shared_ptr<HierarchyNode>& node) {
    if (!node || node->body_indices.empty()) {
        throw std::runtime_error("HierarchyTree::min_body_index received an empty node.");
    }

    return *std::min_element(node->body_indices.begin(), node->body_indices.end());
}

std::vector<int> HierarchyTree::merged_body_indices(const std::shared_ptr<HierarchyNode>& a, const std::shared_ptr<HierarchyNode>& b) {
    if (!a || !b) {
        throw std::runtime_error("HierarchyTree::merged_body_indices received a null node.");
    }

    std::vector<int> indices;
    indices.reserve(a->body_indices.size() + b->body_indices.size());

    indices.insert(indices.end(), a->body_indices.begin(), a->body_indices.end());
    indices.insert(indices.end(), b->body_indices.begin(), b->body_indices.end());

    std::sort(indices.begin(), indices.end());

    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    return indices;
}

void HierarchyTree::assign_depths(const std::shared_ptr<HierarchyNode>& node, int depth) {
    if (!node) {
        return;
    }

    node->depth = depth;

    assign_depths(node->left, depth + 1);
    assign_depths(node->right, depth + 1);
}

void HierarchyTree::collect_leaf_indices(const std::shared_ptr<HierarchyNode>& node, std::vector<int>& indices) const {
    if (!node) {
        return;
    }
    if (node->is_leaf()) {
        indices.push_back(node->body_index);
        return;
    }

    collect_leaf_indices(node->left, indices);
    collect_leaf_indices(node->right, indices);
}

int HierarchyTree::validate_node(const std::shared_ptr<HierarchyNode>& node, std::vector<bool>& seen_bodies) const {
    if (!node) {
        throw std::runtime_error("HierarchyTree validation failed: encountered null node.");
    }
    if (node->node_id < 0) {
        throw std::runtime_error("HierarchyTree validation failed: node has invalid id.");
    }
    if (node->is_leaf()) {
        if (node->body_index < 0 || node->body_index >= static_cast<int>(seen_bodies.size())) {
            throw std::runtime_error("HierarchyTree validation failed: leaf has invalid body index.");
        }
        if (seen_bodies[static_cast<std::size_t>(node->body_index)]) {
            throw std::runtime_error("HierarchyTree validation failed: duplicate body index.");
        }

        seen_bodies[static_cast<std::size_t>(node->body_index)] = true;

        if (node->body_indices.size() != 1 || node->body_indices.front() != node->body_index) {
            throw std::runtime_error("HierarchyTree validation failed: leaf body_indices mismatch.");
        }
        return 1;
    }
    if (!node->left || !node->right) {
        throw std::runtime_error("HierarchyTree validation failed: internal node is not binary.");
    }

    const int left_count = validate_node(node->left, seen_bodies);
    const int right_count = validate_node(node->right, seen_bodies);
    const int total_count = left_count + right_count;

    if (total_count != static_cast<int>(node->body_indices.size())) {
        throw std::runtime_error("HierarchyTree validation failed: internal body count mismatch.");
    }

    const double child_mass_sum = node->left->total_mass + node->right->total_mass;

    if (std::abs(node->total_mass - child_mass_sum) > HIERARCHY_VALIDATE_TOL * std::max(1.0, std::abs(child_mass_sum))) {
        throw std::runtime_error("HierarchyTree validation failed: internal mass mismatch.");
    }
    return total_count;
}