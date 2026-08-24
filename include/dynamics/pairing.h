#pragma once

#include <vector>

#include "core/body.h"

struct Pair {
    int i;
    int j;
};

Pair canonicalizePair(int i, int j);

double pairStrength(const std::vector<Body>& bodies, const Pair& pair);

std::vector<Pair> canonicalizePairs(const std::vector<Pair>& pairs);
std::vector<Pair> canonicalizePairsPreserveOrder(const std::vector<Pair>& pairs);
std::vector<Pair> orderPairsStrength(const std::vector<Pair>& pairs, const std::vector<Body>& bodies);