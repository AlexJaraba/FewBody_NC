#pragma once

#include <vector>

#include "body.h"

struct Pair {
    int i;
    int j;
};

bool is_kepler_pair(int i, int j, const std::vector<Pair>& pairs);

Pair canonicalize_pair(int i, int j);

std::vector<Pair> canonicalize_pairs(const std::vector<Pair>& pairs);

std::vector<Pair> reverse_pairs(const std::vector<Pair>& pairs);