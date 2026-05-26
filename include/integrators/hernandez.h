#pragma once

#include <vector>

#include "core/body.h"
#include "numerics/composition.h"
#include "integrators/integrator.h"
#include "dynamics/pairing.h"
#include "dynamics/corrector.h"

class Hernandez : public Integrator {
public:
    explicit Hernandez(const std::vector<Pair>& fixed_pairs);

    void step(CanonicalState& state, double dt);

private:
    std::vector<Pair> pairs_;
    SymmetricComposition composition_;
};