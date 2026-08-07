#pragma once

#include <vector>

#include "core/body.h"
#include "core/canonical_state.h"
#include "numerics/composition.h"
#include "integrators/integrator.h"
#include "integrators-helper/hernandez/body_stepper.h"
#include "dynamics/pairing.h"

class Hernandez : public Integrator {
public:
    explicit Hernandez(const std::vector<Pair>& fixed_pairs);

    // Body-state Path: Used when the solver evolves physical Body objects directly.
    void step(std::vector<Body>& bodies, double dt, double G) override;
    const std::vector<Pair>& pairs() const;

private:
    HernandezBodyStepper body_stepper_;
};