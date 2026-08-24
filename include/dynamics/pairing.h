#pragma once

#include <vector>

#include "core/body.h"

struct Pair {
    int i;
    int j;
};

Pair canonicalizePair(int i, int j);

<<<<<<< Updated upstream
double pair_strength(const std::vector<Body>& bodies, const Pair& pair);
double strongest_pair_strength_ratio(const std::vector<Pair>& pairs, const std::vector<Body>& bodies);
=======
double pairStrength(const std::vector<Body>& bodies, const Pair& pair);
>>>>>>> Stashed changes

std::vector<Pair> canonicalizePairs(const std::vector<Pair>& pairs);
std::vector<Pair> canonicalizePairsPreserveOrder(const std::vector<Pair>& pairs);
std::vector<Pair> orderPairsStrength(const std::vector<Pair>& pairs, const std::vector<Body>& bodies);