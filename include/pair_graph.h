#pragma once

#include <vector>

#include "body.h"
#include "pairing.h"

struct PairGraph {
    std::vector<Pair> kepler_pairs;
    std::vector<Pair> perturbation_pairs;
};

PairGraph build_hierarchical_pair_graph(const std::vector<Body>& bodies);
