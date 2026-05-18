#include <algorithm>
#include <cmath>

#include "pairing.h"

bool is_kepler_pair(int i, int j, const std::vector<Pair>& pairs)
{
    Pair p = canonicalize_pair(i, j);
    
    for (const auto& p : pairs) {
        if ((p.i == i && p.j == j) || (p.i == j && p.j == i))
            return true;
    }
    return false;
}

Pair canonicalize_pair(int i, int j) {
    if (i < j) {
        return {i, j};
    } else {
        return {j, i};
    }
}

std::vector<Pair> canonicalize_pairs(const std::vector<Pair>& pairs) {
    std::vector<Pair> canonical;

    canonical.reserve(pairs.size());

    for (const auto& p : pairs) {
        canonical.push_back(canonicalize_pair(p.i, p.j));
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

std::vector<Pair> reverse_pairs(const std::vector<Pair>& pairs) {
    std::vector<Pair> reversed = pairs;

    std::reverse(reversed.begin(), reversed.end());

    return reversed;
}