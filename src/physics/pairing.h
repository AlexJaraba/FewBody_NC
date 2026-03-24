#pragma once

#include <vector>
#include "body.h"

struct Pair {
    int i, j;

    Pair(int a, int b) {
        if (a < b) {
            i = a;
            j = b;
        } else {
            i = b;
            j = a;
        }
    }

    bool operator==(const Pair& other) const {
        return (i == other.i && j == other.j) || (i == other.j && j == other.i);
    }
};

std::vector<Pair> build_kepler_pairs(const std::vector<Body>& bodies, double G);

bool is_kepler_pair(int i, int j, const std::vector<Pair>& pairs);