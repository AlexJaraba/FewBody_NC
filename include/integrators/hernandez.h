#pragma once

#include <vector>

#include "body.h"
#include "composition.h"
#include "integrator.h"
#include "pairing.h"
#include "corrector.h"

class Hernandez : public Integrator {
public:
    explicit Hernandez(const std::vector<Pair>& fixed_pairs);

    void step(CanonicalState& state, double dt);

private:
    std::vector<Pair> pairs_;
    SymmetricComposition composition_;
};