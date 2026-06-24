#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <functional>

#include "dynamics/hierarchy_tree.h"

namespace {
    constexpr double HIERARCHY_DISTANCE_FLOOR = 1.0e-300;
    constexpr double HIERARCHY_VALIDATE_TOL = 1.0e-12;
    
    Pair normalized_pair(int i, int j) {
        if (i < j) {
            return Pair{i, j};
        }
        return Pair{j, i};
    }
    bool same_pair(const Pair& a, const Pair& b) {
        return a.i == b.i && a.j == b.j;
    }
    bool contains_pair(const std::vector<Pair>& pairs, const Pair& target) {
        for (const Pair& pair : pairs) {
            if (same_pair(pair, target)) {
                return true;
            }
        }
        return false;
    }
    std::vector<Pair> make_all_pairs_from_body_count(int body_count) {
        std::vector<Pair> pairs;
        for (int i = 0; i < body_count; ++i) {
            for (int j = i + 1; j < body_count; ++j) {
                pairs.push_back(Pair{i, j});
            }
        }
        return pairs;
    }
    void append_recursive_selected_lear_pairs_from_node(const std::shared_ptr<HierarchyNode>& node, const std::vector<Pair>& selected_pairs, std::vector<Pair>& ordered_pairs) {
        if (!node || node->is_leaf()) {
            return;
        }
        append_recursive_selected_lear_pairs_from_node(node->left, selected_pairs, ordered_pairs);
        append_recursive_selected_lear_pairs_from_node(node->right, selected_pairs, ordered_pairs);

        if (node->is_binary() && node->left->is_leaf() && node->right->is_leaf()) {
            const Pair pair = normalized_pair(node->left->body_index, node->right->body_index);
            if (contains_pair(selected_pairs, pair) && !contains_pair(ordered_pairs, pair)) {
                ordered_pairs.push_back(pair);
            }
        }
    }
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

// Node Count
int HierarchyTree::node_count() const {
    return static_cast<int>(nodes_.size());
}

// Leaf Count
int HierarchyTree::leaf_count() const {
    if (!root) {
        return 0;
    }
    return root->leaf_count();
}

// Nodes
const std::vector<std::shared_ptr<HierarchyNode>>& HierarchyTree::nodes() const {
    return nodes_;
}

// Leaf Body Indices
std::vector<int> HierarchyTree::leaf_body_indices() const {
    std::vector<int> indices;
    collect_leaf_indices(root, indices);
    std::sort(indices.begin(), indices.end());
    return indices;
}

// Leaf Binary Canidates
std::vector<HierarchyBinaryCandidate> HierarchyTree::leaf_binary_candidates(const std::vector<Body>& bodies, const HierarchySelectionCriteria& criteria) const {
    if (!root) {
        return {};
    }
    if (static_cast<int>(bodies.size()) != leaf_count()) {
        throw std::runtime_error("HierarchyTree::leaf_binary_candidates received a body count that does not match the tree.");
    }

    std::vector<HierarchyBinaryCandidate> candidates;
    std::function<void(const std::shared_ptr<HierarchyNode>&)> visit = [&](const std::shared_ptr<HierarchyNode>& node) {
        if (!node) {
            return;
        }
        if (node->is_binary() && node->left->is_leaf() && node->right->is_leaf()) {
            const int i = node->left->body_index;
            const int j = node->right->body_index;

            Pair pair{std::min(i, j), std::max(i, j)};

            const Vec3 internal_dr = bodies[pair.i].position - bodies[pair.j].position;
            const double internal_separation = internal_dr.norm();
            const double internal_strength = pair_strength(bodies, pair);
            const double safe_internal_separation = std::max(internal_separation, std::sqrt(HIERARCHY_DISTANCE_FLOOR));

            double nearest_external_separation = std::numeric_limits<double>::infinity();
            double strongest_external_strength = 0.0;

            for (int k = 0; k < static_cast<int>(bodies.size()); ++k) {
                if (k == pair.i || k == pair.j) {
                    continue;
                }
                const Vec3 dr_i = bodies[pair.i].position - bodies[k].position;
                const Vec3 dr_j = bodies[pair.j].position - bodies[k].position;

                nearest_external_separation = std::min(nearest_external_separation, dr_i.norm());
                nearest_external_separation = std::min(nearest_external_separation, dr_j.norm());

                Pair external_i{std::min(pair.i, k), std::max(pair.i, k)};
                Pair external_j{std::min(pair.j, k), std::max(pair.j, k)};

                strongest_external_strength = std::max(strongest_external_strength, pair_strength(bodies, external_i));
                strongest_external_strength = std::max(strongest_external_strength, pair_strength(bodies, external_j));
            }
            const double separation_ratio = std::isfinite(nearest_external_separation) ? nearest_external_separation / safe_internal_separation : std::numeric_limits<double>::infinity();
            const double strength_ratio = strongest_external_strength > 0.0 ? internal_strength / strongest_external_strength : std::numeric_limits<double>::infinity();
            const bool accepted = separation_ratio >= criteria.min_separation_ratio && strength_ratio >= criteria.min_strength_ratio;

            HierarchyBinaryCandidate candidate;
            candidate.pair = pair;
            candidate.node_id = node->node_id;
            candidate.internal_separation = internal_separation;
            candidate.nearest_external_separation = nearest_external_separation;
            candidate.separation_ratio = separation_ratio;
            candidate.internal_strength = internal_strength;
            candidate.strongest_external_strength = strongest_external_strength;
            candidate.strength_ratio = strength_ratio;
            candidate.accepted = accepted;

            candidates.push_back(candidate);
        }
        visit(node->left);
        visit(node->right);
    };
    visit(root);
    return candidates;
}

// Selected Leaf Pairs
std::vector<Pair> HierarchyTree::selected_leaf_pairs(const std::vector<Body>& bodies, const HierarchySelectionCriteria& criteria) const {
    const std::vector<HierarchyBinaryCandidate> candidates = leaf_binary_candidates(bodies, criteria);

    std::vector<Pair> selected_pairs;

    for (const HierarchyBinaryCandidate& candidate : candidates) {
        if (candidate.accepted) {
            selected_pairs.push_back(candidate.pair);
        }
    }
    return canonicalize_pairs_preserve_order(selected_pairs);
}

// Recursive Selected Leaf Pairs
std::vector<Pair> HierarchyTree::recursive_selected_leaf_pairs(const std::vector<Body>& bodies, const HierarchySelectionCriteria& criteria) const {
    if (!root) {
        return {};
    }
    if (static_cast<int>(bodies.size()) != leaf_count()) {
        throw std::runtime_error("HierarchyTree::resursive_selected_leaf_pairs received a body count rhar does not match the tree.");
    }
    
    const std::vector<Pair> selected_pairs = selected_leaf_pairs(bodies, criteria);
    std::vector<Pair> ordered_pairs;

    append_recursive_selected_lear_pairs_from_node(root, selected_pairs, ordered_pairs);

    return canonicalize_pairs_preserve_order(ordered_pairs);
}

// Recursive Hernandez Pair Order
std::vector<Pair> HierarchyTree::recursive_hernandez_pair_order(const std::vector<Body>& bodies, const HierarchySelectionCriteria& criteria) const {
    if (!root) {
        return {};
    }
    if (static_cast<int>(bodies.size()) != leaf_count()) {
        throw std::runtime_error("HierarchyTree::recursive_hernandez_order received a body count that does not match the tree.");
    }

    std::vector<Pair> ordered_pairs = recursive_selected_leaf_pairs(bodies, criteria);
    const std::vector<Pair> all_pairs = make_all_pairs_from_body_count(static_cast<int>(bodies.size()));

    for (const Pair& pair : all_pairs) {
        if (!contains_pair(ordered_pairs, pair)) {
            ordered_pairs.push_back(pair);
        }
    }

    return canonicalize_pairs_preserve_order(ordered_pairs);
}

// Validate
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

// Make Leaf
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

// Merge Nodes
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

// Cluster Strength
double HierarchyTree::cluster_strength(const std::shared_ptr<HierarchyNode>& a, const std::shared_ptr<HierarchyNode>& b) {
    if (!a || !b) {
        throw std::runtime_error("HierarchyTree::cluster_strength received a null node.");
    }

    const Vec3 dr = a->com_position - b->com_position;
    const double r2 = std::max(dr.norm2(), HIERARCHY_DISTANCE_FLOOR);

    return (a->total_mass * b->total_mass) / r2;
}

// Min Body Index
int HierarchyTree::min_body_index(const std::shared_ptr<HierarchyNode>& node) {
    if (!node || node->body_indices.empty()) {
        throw std::runtime_error("HierarchyTree::min_body_index received an empty node.");
    }

    return *std::min_element(node->body_indices.begin(), node->body_indices.end());
}

// Merged Body Indices
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

// Assign Depths
void HierarchyTree::assign_depths(const std::shared_ptr<HierarchyNode>& node, int depth) {
    if (!node) {
        return;
    }

    node->depth = depth;

    assign_depths(node->left, depth + 1);
    assign_depths(node->right, depth + 1);
}

// Collect Leaf Indices
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

// Validate Node
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