#pragma once

#include <vector>

#include "core/body.h"
#include "core/canonical_state.h"
#include "numerics/composition.h"
#include "integrators/integrator.h"
#include "integrators-helper/hernandez/hernandez_cartesian_core.h"
#include "dynamics/pairing.h"

struct HernandezPairLevelSchedule;

class Hernandez : public Integrator {
public:
    explicit Hernandez(const std::vector<Pair>& fixed_pairs);

    // Jacobi/Canonical Path
    void step(CanonicalState& state, double dt, double G);

    // Cartesian pairwise-Kepler path
    void step(std::vector<Body>& bodies, double dt, double G);
    void apply_pair_group(std::vector<Body>& bodies, const std::vector<Pair>& active_pairs, double dt, double G) const;
    void step_block(std::vector<Body>& bodies, const HernandezPairLevelSchedule& schedule, double dt, double G) const;
    const std::vector<Pair>& pairs() const;

private:
    std::vector<Pair> pairs_;
    SymmetricComposition composition_;
    HernandezCartesianCore cartesian_core_;
};