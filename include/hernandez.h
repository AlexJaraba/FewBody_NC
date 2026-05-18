#pragma once

#include <vector>

#include "body.h"
#include "integrator.h"
#include "pairing.h"

class Hernandez : public Integrator {
public:
    explicit Hernandez(const std::vector<Pair>& fixed_pairs);

    void step(std::vector<Body>& bodies, double dt) override;

private:
    std::vector<Pair> pairs_;
};