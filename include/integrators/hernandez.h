#pragma once

#include <vector>

#include "integrators/integrator.h"
#include "integrators-helper/hernandez/body_stepper.h"
#include "dynamics/pairing.h"

class Hernandez : public Integrator {
public:
    explicit Hernandez(const std::vector<Pair>& fixed_pairs);

    void step(std::vector<Body>& bodies, double dt, double G) override;
    const std::vector<Pair>& pairs() const;

private:
    HernandezBodyStepper body_stepper_;
};