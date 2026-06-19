#pragma once

#include <vector>

#include "core/body.h"
#include "dynamics/pairing.h"

class HB15 {
public:
    explicit HB15(const std::vector<Pair>& fixed_pairs);

    void step(std::vector<Body>& bodies, double dt, double G);

    const std::vector<Pair>& pairs() const;

private:
    std::vector<Pair> pairs_;
    
};