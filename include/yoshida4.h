#pragma once

#include <cmath>
#include <vector>

#include "integrator.h"
#include "pairing.h"

class Yoshida4 : public Integrator {
public:
    explicit Yoshida4(const std::vector<Pair>& pairs);
    void step(std::vector<Body>& bodies, double dt) override;
private:
    std::vector<Pair> pairs_;
    const double w1_= 1.0 / (2.0 - std::cbrt(2.0));
    const double w0_ = -std::cbrt(2.0) / (2.0 - std::cbrt(2.0));
};