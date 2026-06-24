#pragma once

#include <memory>
#include <vector>

#include "core/body.h"
#include "dynamics/hierarchy_node.h"
#include "dynamics/pairing.h"

struct HierarchySelectionCriteria {
    // External separation / internal binary separation
    // Larger means the binary is spartially isolated
    double min_separation_ratio = 5.0;

    // Internal pair strength / strongest external pair strength
    // Larger means the binary dominates its perturbations
    double min_strength_ratio = 10.0;
};

struct HierarchyBinaryCandidate {
    Pair pair;
    int node_id = -1;
    double internal_separation = 0.0;
    double nearest_external_separation = 0.0;
    double separation_ratio = 0.0;
    double internal_strength = 0.0;
    double strongest_external_strength = 0.0;
    double strength_ratio = 0.0;
    bool accepted = false;
};

class HierarchyTree {
public:
    std::shared_ptr<HierarchyNode> root;
    explicit HierarchyTree(const std::vector<Body>& bodies);

    bool empty() const;

    int node_count() const;
    int leaf_count() const;

    const std::vector<std::shared_ptr<HierarchyNode>>& nodes() const;

    std::vector<int> leaf_body_indices() const;
    std::vector<HierarchyBinaryCandidate> leaf_binary_candidates(const std::vector<Body>& bodies, const HierarchySelectionCriteria& criteria) const;
    std::vector<Pair> selected_leaf_pairs(const std::vector<Body>& bodies, const HierarchySelectionCriteria& criteria) const;
    std::vector<Pair> recursive_selected_leaf_pairs(const std::vector<Body>& bodies, const HierarchySelectionCriteria& criteria) const;
    std::vector<Pair> recursive_hernandez_pair_order(const std::vector<Body>& bodies, const HierarchySelectionCriteria& criteria) const;

    void validate(int expected_leaf_count) const;
private:
    std::vector<std::shared_ptr<HierarchyNode>> nodes_;
    int next_node_id_ = 0;

    std::shared_ptr<HierarchyNode> make_leaf(const std::vector<Body>& bodies, int body_index);
    std::shared_ptr<HierarchyNode> merge_nodes(const std::shared_ptr<HierarchyNode>& left, const std::shared_ptr<HierarchyNode>& right);

    static double cluster_strength(const std::shared_ptr<HierarchyNode>& a, const std::shared_ptr<HierarchyNode>& b);
    static int min_body_index(const std::shared_ptr<HierarchyNode>& node);
    static std::vector<int> merged_body_indices(const std::shared_ptr<HierarchyNode>& a, const std::shared_ptr<HierarchyNode>& b);

    void assign_depths(const std::shared_ptr<HierarchyNode>& node, int depth);
    void collect_leaf_indices(const std::shared_ptr<HierarchyNode>& node, std::vector<int>& indices) const;

    int validate_node(const std::shared_ptr<HierarchyNode>& node, std::vector<bool>& seen_bodies) const;
};