#include <algorithm>
#include <cmath>

#include "pairing.h"

std::vector<Pair> build_kepler_pairs(const std::vector<Body>& bodies, double G)
{
    int N = bodies.size();

    struct Candidate {
        int i, j;
        double strength;
    };

    std::vector<Candidate> candidates;

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {

            double dx = bodies[i].position[0] - bodies[j].position[0];
            double dy = bodies[i].position[1] - bodies[j].position[1];
            double dz = bodies[i].position[2] - bodies[j].position[2];

            double r2 = dx*dx + dy*dy + dz*dz + 1e-12;

            double strength = G * bodies[i].mass * bodies[j].mass / r2;

            candidates.push_back({i, j, strength});
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.strength > b.strength;
              });

    std::vector<bool> used(N, false);
    std::vector<Pair> pairs;

    for (const auto& c : candidates) {
        if (!used[c.i] && !used[c.j]) {
            pairs.push_back({c.i, c.j});
            used[c.i] = true;
            used[c.j] = true;
        }
    }

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