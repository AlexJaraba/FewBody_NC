#pragma once

#include <vector>

#include "core/body.h"
#include "dynamics/pairing.h"

struct HernandezPairLevelSchedule;

class HernandezBodyStepper {
public:
    explicit HernandezBodyStepper(const std::vector<Pair>& fixed_pairs);

    void step(std::vector<Body>& bodies, double dt, double G);
    void apply_pair_group(std::vector<Body>& bodies, const std::vector<Pair>& active_pairs, double dt, double G) const;
    void step_block(std::vector<Body>& bodies, const HernandezPairLevelSchedule& schedule, double dt, double G) const;

    const std::vector<Pair>& pairs() const;

private:
    std::vector<Pair> pairs_;
    
};