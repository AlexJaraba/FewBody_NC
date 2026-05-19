#include <cmath>

#include "pair_graph.h"

PairGraph build_hierarchical_pair_graph(const std::vector<Body>& bodies) {
    PairGraph graph;
    const int N = bodies.size();

    for (int i = 1; i < N; ++i) {
        double best_strength = -1.0;
        int best_j = 0;

        for (int j = 0; j < i; ++j) {
            double r2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                double dr = bodies[i].position[k] - bodies[j].position[k];
                r2 += dr * dr;
            }
            double strength = (bodies[i].mass * bodies[j].mass) / r2;
            if (strength > best_strength) {
                best_strength = strength;
                best_j = j;
            }
        }
        graph.kepler_pairs.push_back({best_j, i});
    }

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            bool is_kepler = false;
            for (const auto& p : graph.kepler_pairs) {
                if ((p.i == i && p.j == j) || (p.i == j && p.j == i)) {
                    is_kepler = true;
                    break;
                }
            }
            if (!is_kepler) {
                graph.pertubation_pairs.push_back({i, j});
            }
        }
    }
    return graph;
}