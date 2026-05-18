#pragma once

#include <vector>
#include "body.h"

struct Pair {
    int i;
    int j;
};

bool is_kepler_pair(int i, int j, const std::vector<Pair>& pairs);