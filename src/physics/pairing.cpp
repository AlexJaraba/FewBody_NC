#include <algorithm>
#include <cmath>
#include <vector>

#include "pairing.h"
#include "../include/body.h"

struct PairCandidate {
    int i, j;
    double score;
};

std::vector<Pair> build_kepler_pairs(const std::vector<Body>& bodies, double G)
{
    int N = bodies.size();
    std::vector<PairCandidate> candidates;

    // --- Build all candidates ---
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Vec3 dr = bodies[j].position - bodies[i].position;
            double r = dr.norm();

            // PURELY POSITIONAL metric (time-symmetric)
            double score = std::round(r * 1e12) * 1e-12;

            candidates.push_back({i, j, score});
        }
    }

    // --- Deterministic sort ---
    std::sort(candidates.begin(), candidates.end(),
        [](const PairCandidate& a, const PairCandidate& b) {

            if (std::abs(a.score - b.score) > 1e-15)
                return a.score < b.score;

            if (a.i != b.i)
                return a.i < b.i;

            return a.j < b.j;
        });

    // --- Greedy selection ---
    std::vector<bool> used(N, false);
    std::vector<Pair> pairs;

    for (const auto& c : candidates) {
        if (!used[c.i] && !used[c.j]) {
            pairs.emplace_back(c.i, c.j);
            used[c.i] = true;
            used[c.j] = true;
        }
    }

    // --- Final sort (guarantee order) ---
    std::sort(pairs.begin(), pairs.end(),
        [](const Pair& a, const Pair& b) {
            if (a.i != b.i) return a.i < b.i;
            return a.j < b.j;
        });

    return pairs;
}


bool is_kepler_pair(int i, int j, const std::vector<Pair>& pairs)
{
    for (const auto& p : pairs) {
        if ((p.i == i && p.j == j) || (p.i == j && p.j == i))
            return true;
    }
    return false;
}