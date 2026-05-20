#pragma once

#include <cmath>
#include <vector>

#include "integrator.h"
#include "pairing.h"
#include "canonical_state.h"

class Yoshida4 : public Integrator {
public:
    explicit Yoshida4(const std::vector<Pair>& pairs);
    void step(CanonicalState& state, double dt) override;
private:
    std::vector<Pair> pairs_;
    const double w1_= 1.0 / (2.0 - std::cbrt(2.0));
    const double w0_ = -std::cbrt(2.0) / (2.0 - std::cbrt(2.0));
};