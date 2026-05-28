#include "integrators/hernandez_recursive.h"
#include "dynamics/operators.h"

void recursive_hernandez_step(std::shared_ptr<HierarchyNode> node, CanonicalState& state, double dt, double G) {
    if (!node) {
        return;
    }
    if (node->is_leaf()) {
        return;
    }

    recursive_hernandez_step(node->left, state, 0.5 * dt, G);
    recursive_hernandez_step(node->right, state, 0.5 * dt, G);

    if (node->left && node->right && node->left->is_leaf() && node->right->is_leaf()) {
        const int i = node->left->body_index;
        const int j = node->right->body_index;
        
        Pair pair{i, j};
        std::vector<Pair> pairs;
        pairs.push_back(pair);

        kepler_operator(state, pairs, dt, G);
    }

    recursive_hernandez_step(node->right, state, 0.5 * dt, G);
    recursive_hernandez_step(node->left, state, 0.5 * dt, G);
}