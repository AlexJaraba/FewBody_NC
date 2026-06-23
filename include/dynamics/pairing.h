#pragma once

#include <vector>

#include "core/body.h"

struct Pair {
    int i;
    int j;
};

bool is_kepler_pair(int i, int j, const std::vector<Pair>& pairs);

bool pair_exists(int i, int j, const std::vector<Pair>& pairs);

Pair canonicalize_pair(int i, int j);

double pair_strength(const std::vector<Body>& bodies, const Pair& pair);
double strongest_pair_strength_ratio(const std::vector<Pair>& pairs, const std::vector<Body>& bodies);

std::vector<Pair> canonicalize_pairs(const std::vector<Pair>& pairs);
std::vector<Pair> reverse_pairs(const std::vector<Pair>& pairs);
std::vector<Pair> canonicalize_pairs_preserve_order(const std::vector<Pair>& pairs);
std::vector<Pair> order_pairs_by_strength(const std::vector<Pair>& pairs, const std::vector<Body>& bodies);