#include <algorithm>
#include <cmath>

#include "pairing.h"

bool is_kepler_pair(int i, int j, const std::vector<Pair>& pairs)
{
    for (const auto& p : pairs) {
        if ((p.i == i && p.j == j) || (p.i == j && p.j == i))
            return true;
    }
    return false;
}