#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dynamics/pairing.h"

Pair canonicalizePair(int i, int j) {
    if (i < j) {
        return {i, j};
    } else {
        return {j, i};
    }
}

double pairStrength(const std::vector<Body>& bodies, const Pair& pair) {
    if (pair.i < 0 || pair.j < 0 || pair.i >= static_cast<int>(bodies.size()) || pair.j >= static_cast<int>(bodies.size())) {
        throw std::runtime_error("pair_strength received an out-of-range pair.");
    }
    const std::size_t i = static_cast<std::size_t>(pair.i);
    const std::size_t j = static_cast<std::size_t>(pair.j);
    const Vec3 dr = bodies[i].position - bodies[j].position;
    const double r2 = dr.norm2();
    if (!std::isfinite(r2) || r2 <= 0.0) {
        throw std::runtime_error("pair_strength requires a finite, non-zero pair separation.");
    }

    return (bodies[pair.i].mass * bodies[pair.j].mass) / r2;
}

std::vector<Pair> canonicalizePairs(const std::vector<Pair>& pairs) {
    std::vector<Pair> canonical;
    canonical.reserve(pairs.size());

    for (const auto& p : pairs) {
        canonical.push_back(canonicalizePair(p.i, p.j));
    }

    std::sort(canonical.begin(), canonical.end(), [](const Pair& a, const Pair& b) {
        if (a.i != b.i)
            return a.i < b.i;

        return a.j < b.j;
    });

    canonical.erase(std::unique(canonical.begin(), canonical.end(), [](const Pair& a, const Pair& b) {
        return a.i == b.i && a.j == b.j;
    }), canonical.end());

    return canonical;
}

std::vector<Pair> canonicalizePairsPreserveOrder(const std::vector<Pair>& pairs) {
    std::vector<Pair> canonical;
    canonical.reserve(pairs.size());

    for (const Pair& pair : pairs) {
        const Pair current = canonicalizePair(pair.i, pair.j);

        bool already_exists = false;

        for (const Pair& existing : canonical) {
            if (existing.i == current.i && existing.j == current.j) {
                already_exists = true;
                break;
            }
        }

        if (!already_exists) {
            canonical.push_back(current);
        }
    }

    return canonical;
}

std::vector<Pair> orderPairsStrength(const std::vector<Pair>& pairs, const std::vector<Body>& bodies) {
    struct PairWithStrength {
        Pair pair;
        double strength;
    };

    const std::vector<Pair> unique_pairs = canonicalizePairs(pairs);

    std::vector<PairWithStrength> ranked_pairs;
    ranked_pairs.reserve(unique_pairs.size());

    for (const Pair& pair : unique_pairs) {
        ranked_pairs.push_back({pair, pairStrength(bodies, pair)});
    }

    std::stable_sort(ranked_pairs.begin(), ranked_pairs.end(), [](const PairWithStrength& a, const PairWithStrength& b) {return a.strength > b.strength;});
    std::vector<Pair> ordered_pairs;
    ordered_pairs.reserve(ranked_pairs.size());

    for (const PairWithStrength& ranked_pair : ranked_pairs) {
        ordered_pairs.push_back(ranked_pair.pair);
    }

    return ordered_pairs;
}