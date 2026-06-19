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
    void apply_kinetic_correction_drift(std::vector<Body>& bodies, double drift_dt) {
        for (Body& body : bodies) {
            body.position += drift_dt * body.velocity;
            body.updateMomentumFromVelocity();
        }
    }
    double hb15_kinetic_correction_coefficient(std::size_t body_count) {
        if (body_count < 2) {
            return 0.0;
        }
        return -static_cast<double>(body_count - 2);
    }
}

HB15::HB15(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs(fixed_pairs)) {}

void HB15::step(std::vector<Body>& bodies, double dt, double G) {
    const double correction_coefficient = hb15_kinetic_correction_coefficient(bodies.size());
    const double correction_half_dt = 0.5 * correction_coefficient * dt;
    const double pair_half_dt = 0.5 * dt;

    apply_kinetic_correction_drift(bodies, correction_half_dt);

    for (const Pair& pair : pairs_) {
        apply_checked_pair_map(bodies, pair, pair_half_dt, G);
    }
    for (auto it = pairs_.rbegin(); it != pairs_.rend(); ++it) {
        apply_checked_pair_map(bodies, *it, pair_half_dt, G);
    }

    apply_kinetic_correction_drift(bodies, correction_half_dt);
}

const std::vector<Pair>& HB15::pairs() const {
    return pairs_;
}