#include <stdexcept>

#include "integrators/hb15.h"
#include "integrators/hb15_pair_map.h"
#include "dynamics/pairing.h"

namespace {
    void apply_checked_pair_map(std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        HB15PairMapResult result = apply_hb15_pair_kepler_map(bodies, pair.i, pair.j, dt, G);
        if (!result.converged) {
            throw std::runtime_error("HB15 pair Kepler map failed to converge.");
        }
    }
}

HB15::HB15(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs(fixed_pairs)) {}

void HB15::step(std::vector<Body>& bodies, double dt, double G) {
    const double half_dt = 0.5 * dt;
    for (const Pair& pair : pairs_) {
        apply_checked_pair_map(bodies, pair, half_dt, G);
    }
    for (auto it = pairs_.rbegin(); it != pairs_.rend(); ++it) {
        apply_checked_pair_map(bodies, *it, half_dt, G);
    }
}

const std::vector<Pair>& HB15::pairs() const {
    return pairs_;
}