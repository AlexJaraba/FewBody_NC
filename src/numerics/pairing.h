#pragma once

#include <vector>
#include "body.h"

struct Pair {
    int i;
    int j;
};

std::vector<Pair> build_kepler_pairs(const std::vector<Body>& bodies, double G);

bool is_kepler_pair(int i, int j, const std::vector<Pair>& pairs);