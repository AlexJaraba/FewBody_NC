#pragma once

#include <functional>
#include <vector>

#include "body.h"
#include "pairing.h"
#include "corrector.h"
#include "canonical_state.h"

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
        void execute(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const;
        
        const std::vector<CompositionStep>& steps() const;

    private:
        std::vector<CompositionStep> steps_;
        bool has_corrector_ = false;
        SymplecticCorrector corrector_;
};