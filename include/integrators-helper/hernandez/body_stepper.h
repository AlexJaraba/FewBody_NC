#pragma once

#include <vector>

#include "core/body.h"
#include "dynamics/pairing.h"

class ExternalPotential;
class HernandezBodyStepper {
public:
    explicit HernandezBodyStepper(const std::vector<Pair>& fixed_pairs);

    void step(std::vector<Body>& bodies, double dt, double G, const ExternalPotential* external_potential);
    const std::vector<Pair>& pairs() const;

private:
    std::vector<Pair> pairs_;
};