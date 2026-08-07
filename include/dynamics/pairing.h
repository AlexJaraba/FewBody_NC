#pragma once

#include <vector>

#include "core/body.h"

struct Pair {
    int i = -1;
    int j = -1;
};

[[nodiscard]] Pair order_pair_indices(int i, int j);
[[nodiscard]] double pair_force_scale(const std::vector<Body>& bodies, const Pair& pair);
[[nodiscard]] std::vector<Pair> build_all_unique_duplicate_pairs(const std::vector<Pair>& pairs);
[[nodiscard]] std::vector<Pair> sort_and_remove_duplicate_pairs(const std::vector<Pair>& pairs);
[[nodiscard]] std::vector<Pair> preserve_order_and_remove_duplicate_pairs(const std::vector<Pair>& pairs);
[[nodiscard]] std::vector<Pair> order_pairs_by_force_scale(const std::vector<Pair>& pairs, const std::vector<Body>& bodies);