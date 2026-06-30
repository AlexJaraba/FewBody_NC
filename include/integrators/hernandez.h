#pragma once

#include <vector>

#include "core/body.h"
#include "core/canonical_state.h"
#include "numerics/composition.h"
#include "integrators/integrator.h"
#include "integrators-helper/hernandez/body_stepper.h"
#include "dynamics/pairing.h"

struct HernandezPairLevelSchedule;

class Hernandez : public Integrator {
public:
    explicit Hernandez(const std::vector<Pair>& fixed_pairs);

    // Canonical-state Path: Used when the solver evolves a CanonicalState.
    void step(CanonicalState& state, double dt, double G) override;
    void step_canonical(CanonicalState& state, double dt, double G);

    // Body-state Path: Used when the solver evolves physical Body objects directly.
    void step(std::vector<Body>& bodies, double dt, double G);
    void step_bodies(std::vector<Body>& bodies, double dt, double G);
    void apply_pair_group(std::vector<Body>& bodies, const std::vector<Pair>& active_pairs, double dt, double G) const;
    void step_block(std::vector<Body>& bodies, const HernandezPairLevelSchedule& schedule, double dt, double G) const;
    const std::vector<Pair>& pairs() const;

private:
    std::vector<Pair> pairs_;
    SymmetricComposition canonical_composition_;
    HernandezBodyStepper body_stepper_;
};