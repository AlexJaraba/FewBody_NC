#pragma once

#include <functional>
#include <vector>

#include "body.h"
#include "pairing.h"
#include "corrector.h"

enum class OperatorType {
    DRIFT,
    KICK,
    KEPLER,
};

struct CompositionStep {
    OperatorType type;
    double coefficient;
};

class SymmetricComposition {
    public:
        SymmetricComposition(const std::vector<CompositionStep>& steps);
        void set_corrector(const SymplecticCorrector& corrector);
        void execute(std::vector<Body>& bodies, const std::vector<Pair>& pairs, double dt, double G) const;
        
        const std::vector<CompositionStep>& steps() const;

    private:
        std::vector<CompositionStep> steps_;
        bool has_corrector_ = false;
        SymplecticCorrector corrector_{0.0};
};