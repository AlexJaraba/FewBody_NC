#include <stdexcept>

#include "integrators/hb15.h"
#include "integrators/hb15_pair_map.h"
#include "dynamics/pairing.h"

HB15::HB15(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs(fixed_pairs)) {}

void HB15::step(std::vector<Body>& bodies, double dt, double G) {
    for (const Pair& pair : pairs_) {
        HB15PairMapResult result = apply_hb15_pair_kepler_map(bodies, pair.i, pair.j, dt, G);
        if (!result.converged) {
            throw std::runtime_error("HB15 pair Kepler map failed to converge.");
        }
    }
}

const std::vector<Pair>& HB15::pairs() const {
    return pairs_;
}