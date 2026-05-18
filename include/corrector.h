#pragma once

#include <vector>

#include "body.h"
#include "pairing.h"

class SymplecticCorrector {
    public:
        SymplecticCorrector(double coefficient);
        void apply_forward(std::vector<Body>& bodies, const std::vector<Pair>& pairs, double dt, double G) const;
        void apply_backward(std::vector<Body>& bodies, const std::vector<Pair>& pairs, double dt, double G) const;
    
    private:
        double coefficient_;
};